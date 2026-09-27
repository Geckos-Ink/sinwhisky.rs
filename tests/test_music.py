"""Optional end-to-end tests for the music wrapper (FFmpeg and release Rust binary)."""
import array
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
FFMPEG = shutil.which(os.environ.get("FFMPEG", "ffmpeg"))
BINARY = pathlib.Path(os.environ.get("SINWHISKY_BIN", ROOT / "target/release/sinwhisky")).resolve()


@unittest.skipUnless(FFMPEG and BINARY.is_file(), "FFmpeg and the release Rust binary are required")
class MusicScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sinwhisky-music-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = pathlib.Path(self.temp.name)
        self.source = self.folder / "input music.wav"
        self.rate = 8000
        self.frames = 17003  # Cross the wrapper's two-second block boundary.
        self.pcm = array.array("h")
        for i in range(self.frames):
            self.pcm.extend((round(7000 * math.sin(i * 0.13)), round(5000 * math.cos(i * 0.19))))
        with wave.open(str(self.source), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(self.rate)
            audio.writeframes(self.pcm.tobytes() if sys.byteorder == "little" else self.little(self.pcm))
        self.env = dict(os.environ, FFMPEG=FFMPEG, SINWHISKY_BIN=str(BINARY))

    @staticmethod
    def little(values):
        copy = array.array(values.typecode, values)
        copy.byteswap()
        return copy.tobytes()

    def run_script(self, downscale="1", upscale="4", *options):
        output = self.folder / "output music.wav"
        command = [str(ROOT / "test_music.sh"), str(self.source), downscale, upscale, str(output), *options]
        return subprocess.run(command, env=self.env, capture_output=True, text=True), output

    @staticmethod
    def read(path):
        with wave.open(str(path), "rb") as audio:
            info = (audio.getframerate(), audio.getnchannels(), audio.getnframes())
            pcm = array.array("h", audio.readframes(audio.getnframes()))
            if sys.byteorder != "little":
                pcm.byteswap()
            return info, pcm

    def test_blocks_match_whole_signal_in_both_modes(self):
        for mode in ("uniform", "angular"):
            with self.subTest(mode=mode):
                result, output = self.run_script("1", "4", "--mode", mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                info, pcm = self.read(output)
                self.assertEqual(info, (32000, 2, (self.frames - 1) * 4 + 1))
                for ch in range(2):
                    data = " ".join(format(x * 0.8 / 32768, ".17g") for x in self.pcm[ch::2])
                    direct = subprocess.run([str(BINARY), "3", "1", "1", mode], input=data,
                                            capture_output=True, text=True, check=True)
                    expected = [round(max(-1, min(32767/32768, float(x))) * 32768) for x in direct.stdout.split()]
                    self.assertEqual(len(pcm[ch::2]), len(expected))
                    self.assertLessEqual(max(abs(a-b) for a, b in zip(pcm[ch::2], expected)), 1)
                output.unlink()

    def test_fractional_upscale_keeps_duration_and_sets_rate(self):
        result, output = self.run_script("1", "1.5")
        self.assertEqual(result.returncode, 0, result.stderr)
        info, _ = self.read(output)
        self.assertEqual(info[:2], (12000, 2))
        self.assertLess(abs(info[2] / info[0] - self.frames / self.rate), 2 / self.rate)
        self.assertIn("Fractional upscale", result.stdout)

    def test_bitrate_reduction_then_resampling(self):
        result, output = self.run_script("0.5", "2", "--seconds", "0.5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MP3 128 kbps", result.stdout)  # Stereo PCM16 at 8 kHz is 256 kbps.
        info, _ = self.read(output)
        self.assertEqual(info[:2], (16000, 2))
        self.assertGreater(info[2], 7000)
        self.assertLess(info[2], 9000)

    def test_invalid_factors_and_existing_output_are_rejected(self):
        for downscale, upscale in [("0", "2"), ("1.1", "2"), ("nan", "2"), ("1", "1"), ("1", "inf")]:
            with self.subTest(downscale=downscale, upscale=upscale):
                result, output = self.run_script(downscale, upscale)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())
        output = self.folder / "output music.wav"
        output.write_bytes(b"preserve existing output")
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), b"preserve existing output")


if __name__ == "__main__":
    unittest.main()
