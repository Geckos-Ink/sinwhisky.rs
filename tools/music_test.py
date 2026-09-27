#!/usr/bin/env python3
"""Reduce encoded bitrate, then apply SinWhisky to raise sample rate for listening.

Python standard library only. FFmpeg (with libmp3lame) and Rust/Cargo are required.
The input is read-only; an existing output is never overwritten.
"""
import argparse
import array
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MP3_BITRATES = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)


def audio_info(ffmpeg, path):
    # FFmpeg prints input metadata without decoding when no output is provided.
    result = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
                            capture_output=True, text=True)
    lines = result.stderr.splitlines()
    audio = next((line for line in lines if re.search(r"Stream #0:\d+.*Audio:", line)), None)
    if audio is None:
        raise ValueError(f"Cannot inspect input audio: {result.stderr.strip()}")
    rate = re.search(r"\b(\d+) Hz\b", audio)
    bitrate = re.search(r"\b([\d.]+) kb/s\b", audio)
    if bitrate is None:
        bitrate = re.search(r"bitrate:\s*([\d.]+) kb/s", result.stderr)
    if rate is None:
        raise ValueError("Cannot determine the audio sample rate")
    return int(rate.group(1)), float(bitrate.group(1)) if bitrate else None


def rust_binary(override):
    if override:
        path = pathlib.Path(override).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Rust executable is missing or not executable: {path}")
        return path
    cargo = shutil.which("cargo")
    if not cargo:
        raise ValueError("Cargo was not found; load ~/.cargo/env or set SINWHISKY_BIN")
    subprocess.run([cargo, "build", "--offline", "--release", "--manifest-path", str(ROOT / "Cargo.toml")], check=True)
    result = subprocess.run([cargo, "metadata", "--offline", "--no-deps", "--format-version", "1",
                             "--manifest-path", str(ROOT / "Cargo.toml")], capture_output=True, text=True, check=True)
    import json
    return pathlib.Path(json.loads(result.stdout)["target_directory"]) / "release" / "sinwhisky"


def interpolate_block(binary, pcm, channels, left_frames, count, final, factor, mode, gain):
    """Halo samples retain the same adjacent-triplet fit/blend across block boundaries."""
    begin = left_frames * factor
    end = (left_frames + count) * factor - (factor - 1 if final else 0)
    predictions = []
    for ch in range(channels):
        samples = pcm[ch::channels]
        data = " ".join(format(sample * gain / 32768.0, ".17g") for sample in samples)
        result = subprocess.run([str(binary), str(factor - 1), "1", "1", mode],
                                input=data, capture_output=True, text=True, check=True)
        output = array.array("d", map(float, result.stdout.split()))
        expected = (len(samples) - 1) * factor + 1
        if len(output) != expected:
            raise ValueError("Rust resampler returned an unexpected output length")
        predictions.append(output[begin:end])
    interleaved = array.array("d")
    clipped = 0
    peak = 0.0
    for frame in zip(*predictions):
        for value in frame:
            if not math.isfinite(value):
                raise ValueError("Rust resampler returned a non-finite sample")
            peak = max(peak, abs(value))
            if abs(value) > 1.0:
                clipped += 1
            interleaved.append(value)
    if sys.byteorder != "little":
        interleaved.byteswap()
    return interleaved.tobytes(), clipped, peak


def stream_resample(ffmpeg, source, destination, binary, rate, target_rate, channels,
                    factor, mode, gain, selection):
    decode_command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *selection,
                      "-i", str(source), "-map", "0:a:0", "-ac", str(channels),
                      "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    encode_command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                      "-f", "f64le", "-ar", str(rate * factor), "-ac", str(channels),
                      "-i", "pipe:0", "-ar", str(target_rate)]
    if destination.suffix.lower() == ".wav":
        encode_command += ["-c:a", "pcm_s16le", "-rf64", "auto"]
    else:
        encode_command += ["-c:a", "flac", "-sample_fmt", "s32"]
    encode_command.append(str(destination))
    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=decode_log)
        encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stderr=encode_log)
        try:
            chunk = rate * 2  # Two seconds, plus two-frame halos on either side.
            frame_bytes = channels * 2

            def read_frames(count):
                data = decoder.stdout.read(count * frame_bytes)
                if len(data) % frame_bytes:
                    raise ValueError("Decoder returned an incomplete PCM frame")
                values = array.array("h")
                values.frombytes(data)
                if sys.byteorder != "little":
                    values.byteswap()
                return values, len(data) < count * frame_bytes

            pending, eof = read_frames(chunk + 2)
            left = array.array("h")
            frames_written = 0
            total_clipped = 0
            peak = 0.0
            while pending:
                if not eof and len(pending) // channels < chunk + 2:
                    extra, eof = read_frames(chunk + 2 - len(pending) // channels)
                    pending.extend(extra)
                available = len(pending) // channels
                count = min(chunk, available if eof else available - 2)
                final = eof and count == available
                block = left + pending[:(count + 2) * channels]
                data, clipped, block_peak = interpolate_block(binary, block, channels,
                    len(left) // channels, count, final, factor, mode, gain)
                encoder.stdin.write(data)
                total_clipped += clipped
                peak = max(peak, block_peak)
                frames_written += count
                # Keep preceding samples in unmodified PCM units, not synthesized output.
                left = (left + pending[:count * channels])[-2 * channels:]
                del pending[:count * channels]
            if frames_written == 0:
                raise ValueError("The selected audio is empty")
            encoder.stdin.close()
            decoder.stdout.close()
            if decoder.wait() != 0 or encoder.wait() != 0:
                decode_log.seek(0)
                encode_log.seek(0)
                raise ValueError((decode_log.read() + encode_log.read()).decode(errors="replace").strip())
            return frames_written, total_clipped, peak
        finally:
            for process in (decoder, encoder):
                if process.poll() is None:
                    process.terminate()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("downscale_bitrate", type=float, help="encoded bitrate fraction: 0 < factor <= 1; 1 skips lossy re-encoding")
    parser.add_argument("upscale", type=float, help="sample-rate multiplier > 1 (e.g. 2 or 1.5)")
    parser.add_argument("output", type=pathlib.Path, help="new .wav or .flac file")
    parser.add_argument("--mode", choices=("uniform", "angular"), default="uniform")
    parser.add_argument("--start", type=float, default=0, help="excerpt start in seconds")
    parser.add_argument("--seconds", type=float, help="excerpt duration; omit for the full track")
    parser.add_argument("--gain", type=float, default=0.8, help="shared amplitude gain in (0, 1]; default provides headroom")
    parser.add_argument("--channels", type=int, choices=(1, 2), default=2, help="decoded mono/stereo channels")
    parser.add_argument("--source-bitrate", type=float, help="source bitrate in kbps if metadata is missing or requires an override")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--binary", default=os.environ.get("SINWHISKY_BIN"))
    args = parser.parse_args()
    if not math.isfinite(args.downscale_bitrate) or not 0 < args.downscale_bitrate <= 1:
        parser.error("downscale_bitrate must satisfy 0 < value <= 1")
    if not math.isfinite(args.upscale) or args.upscale <= 1:
        parser.error("upscale must be finite and > 1")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("start must be finite and >= 0")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be finite and > 0")
    if not math.isfinite(args.gain) or not 0 < args.gain <= 1:
        parser.error("gain must satisfy 0 < value <= 1")
    if args.source_bitrate is not None and (not math.isfinite(args.source_bitrate) or args.source_bitrate <= 0):
        parser.error("source-bitrate must be finite and > 0")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input file does not exist: {source}")
    if output.exists() or output == source:
        parser.error("output must be a new file, distinct from the input")
    if output.suffix.lower() not in (".wav", ".flac"):
        parser.error("use .wav or .flac to retain the raised sample rate")
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        parser.error("FFmpeg was not found; install it or set FFMPEG=/path/to/ffmpeg")
    source_rate, bitrate = audio_info(ffmpeg, source)
    bitrate = args.source_bitrate if args.source_bitrate is not None else bitrate
    target_bitrate = None
    if args.downscale_bitrate < 1:
        if bitrate is None:
            parser.error("source bitrate is unavailable; specify --source-bitrate in kbps")
        requested = bitrate * args.downscale_bitrate
        supported = [value for value in MP3_BITRATES if value <= requested]
        if not supported:
            parser.error(f"requested {requested:g} kbps is below the supported MP3 minimum of 8 kbps")
        target_bitrate = supported[-1]
        print(f"Encoded bitrate: {bitrate:g} kbps → requested {requested:g} → MP3 {target_bitrate} kbps", flush=True)
    else:
        print("downscale_bitrate=1: decode the source without lossy re-encoding", flush=True)
    binary = rust_binary(args.binary)
    selection = ["-ss", str(args.start)]
    if args.seconds is not None:
        selection += ["-t", str(args.seconds)]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the output for atomic, no-overwrite hard-link installation.
    with tempfile.TemporaryDirectory(prefix=".sinwhisky-music-", dir=output.parent) as temp:
        temp = pathlib.Path(temp)
        if target_bitrate is not None:
            degraded = temp / "degraded.mp3"
            print("Encoding the selected audio at the reduced bitrate…", flush=True)
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                            *selection, "-i", str(source), "-map", "0:a:0", "-ac", str(args.channels),
                            "-c:a", "libmp3lame", "-b:a", f"{target_bitrate}k", str(degraded)], check=True)
            rate, actual_bitrate = audio_info(ffmpeg, degraded)
            print(f"Degraded audio: {rate} Hz, {actual_bitrate or target_bitrate:g} kbps", flush=True)
            decode_source, decode_selection = degraded, []
        else:
            rate = source_rate
            decode_source, decode_selection = source, selection
        target_rate = round(rate * args.upscale)
        if target_rate <= rate:
            parser.error("upscale is too close to 1 to increase the integer sample rate")
        factor = math.ceil(args.upscale)
        print(f"SinWhisky {args.mode}: {rate} Hz → {target_rate} Hz; gain={args.gain:g}", flush=True)
        if factor != args.upscale:
            print(f"Fractional upscale: Rust interpolates {factor}x, then FFmpeg converts to {target_rate} Hz", flush=True)
        staged = temp / output.name
        count, clipped, peak = stream_resample(ffmpeg, decode_source, staged, binary, rate,
            target_rate, args.channels, factor, args.mode, args.gain, decode_selection)
        os.link(staged, output)  # Fails if another process created the output meanwhile.
    print(f"Wrote {output} ({count/rate:.3f} seconds, {target_rate} Hz, {args.channels} channels)")
    print(f"Rust output peak={peak:.6f}; samples above full scale={clipped}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"test_music: {error}", file=sys.stderr)
        raise SystemExit(2)
