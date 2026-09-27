#!/usr/bin/env python3
"""Local listening experiment: FFmpeg decoding, NumPy anti-alias filter, Rust resampling.

Requires numpy and an FFmpeg binary with libmp3lame. Audio stays in the output directory.
This measures one excerpt, not perceptual quality or codec compression performance.
"""
import argparse
import json
import pathlib
import subprocess
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_wav(path):
    with wave.open(str(path), "rb") as audio:
        assert audio.getsampwidth() == 2, "expected PCM16"
        rate = audio.getframerate()
        data = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        return data.reshape(-1, audio.getnchannels()).astype(float) / 32768, rate


def write_wav(path, data, rate):
    # A shared fixed gain is applied before analysis; no per-method normalization.
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(data.shape[1])
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(np.rint(np.clip(data, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())


def metrics(reference, output):
    error = reference - output
    rmse = float(np.sqrt(np.mean(error ** 2)))
    power = float(np.mean(reference ** 2))
    return {"rmse": rmse, "max_absolute": float(np.max(np.abs(error))),
            "snr_db": float(10 * np.log10(power / (rmse ** 2))) if power > 0 and rmse > 0 else None,
            "peak": float(np.max(np.abs(output))), "clipped_samples": int(np.sum(np.abs(output) >= 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "target/release/sinwhisky")
    parser.add_argument("--start", type=float, default=60)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()
    if not np.isfinite(args.start) or args.start < 0 or not np.isfinite(args.seconds) or not 0 < args.seconds <= 60 or args.factor not in (2, 3, 4):
        parser.error("start >= 0, seconds in (0, 60], factor in {2,3,4} required")
    dest = args.output_dir.resolve()
    # An existing output must be empty to avoid overwriting the source or previous trials.
    if dest.exists() and any(dest.iterdir()):
        parser.error("output directory must be new or empty")
    dest.mkdir(parents=True, exist_ok=True)

    def ffmpeg(*options):
        subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", *map(str, options)], check=True)

    ffmpeg("-ss", args.start, "-i", args.input.resolve(), "-t", args.seconds,
           "-ar", "44100", "-c:a", "pcm_s16le", dest / "decoded.wav")
    decoded, rate = read_wav(dest / "decoded.wav")
    if len(decoded) < 1024:
        raise SystemExit("Excerpt is too short for filtering and 256-frame edge exclusion")
    reference = decoded * 0.8  # fixed headroom shared by every method
    q = args.factor
    taps = np.arange(-128, 129)
    cutoff = 0.45 / q  # cycles per original sample; below new Nyquist
    kernel = 2 * cutoff * np.sinc(2 * cutoff * taps) * np.hamming(len(taps))
    kernel /= np.sum(kernel)
    padded = np.pad(reference, ((128, 128), (0, 0)), mode="reflect")
    filtered = np.column_stack([np.convolve(padded[:, ch], kernel, mode="valid") for ch in range(reference.shape[1])])
    low = filtered[::q]
    nout = (len(low) - 1) * q + 1
    reference = reference[:nout]
    filtered = filtered[:nout]
    write_wav(dest / "reference.wav", reference, rate)
    write_wav(dest / "bandlimited.wav", filtered, rate)
    write_wav(dest / "low-rate.wav", low, rate // q)
    outputs = {"linear": np.column_stack([np.interp(np.arange(nout) / q, np.arange(len(low)), low[:, ch]) for ch in range(low.shape[1])])}
    for mode in ("uniform", "angular"):
        channels = []
        for ch in range(low.shape[1]):
            process = subprocess.run([str(args.binary.resolve()), str(q - 1), "1", "1", mode],
                input="\n".join(format(x, ".17g") for x in low[:, ch]), text=True, capture_output=True, check=True)
            channels.append(np.fromstring(process.stdout, sep=" "))
        outputs[mode] = np.column_stack(channels)
    report = {
        "start_seconds": args.start, "frames": nout, "channels": reference.shape[1],
        "source_rate": rate, "reduced_rate": rate // q, "factor": q, "shared_gain": 0.8,
        "filter": "257-tap symmetric Hamming-windowed sinc, cutoff=0.45/factor cycles/sample, reflect padding",
        "metrics": {},
        "notes": ["Metrics before output PCM16 quantization; exclude 256 edge frames.",
                  "Bandlimited reference isolates interpolation error; source reference includes discarded high frequencies.",
                  "MP3 bitrate controls use the full-rate reference, independently of resampling.",
                  "No perceptual quality claim; no circle-codec bitrate is measured."]}
    for name, output in outputs.items():
        assert output.shape == reference.shape
        assert np.isfinite(output).all()
        np.testing.assert_array_equal(output[::q], low)
        report["metrics"][name] = {
            "vs_source": metrics(reference[256:-256], output[256:-256]),
            "vs_bandlimited": metrics(filtered[256:-256], output[256:-256])}
        write_wav(dest / f"{name}.wav", output, rate)
    for bitrate in (64, 128):
        mp3 = dest / f"bitrate-{bitrate}k.mp3"
        wav = dest / f"bitrate-{bitrate}k.wav"
        ffmpeg("-i", dest / "reference.wav", "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", mp3)
        ffmpeg("-i", mp3, "-c:a", "pcm_s16le", wav)
        data, decoded_rate = read_wav(wav)
        assert decoded_rate == rate and data.shape == reference.shape
        report["metrics"][f"mp3_{bitrate}k"] = {"vs_source": metrics(reference[256:-256], data[256:-256]), "file_bytes": mp3.stat().st_size}
    (dest / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
