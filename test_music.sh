#!/usr/bin/env bash
# Usage: ./test_music.sh INPUT DOWNSCALE_BITRATE UPSCALE OUTPUT.wav [options]
# Alternatively set INPUT_MUSIC, DOWNSCALE_BITRATE, UPSCALE, OUTPUT_MUSIC and run with no arguments.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"

# Configuration defaults for the environment-based invocation.
input_music="${INPUT_MUSIC:-}"
downscale_bitrate="${DOWNSCALE_BITRATE:-0.25}"
upscale="${UPSCALE:-2}"
output_music="${OUTPUT_MUSIC:-}"

if ! command -v cargo >/dev/null 2>&1 && [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python 3 was not found. Set PYTHON=/path/to/python3.\n' >&2
    exit 2
fi
if [[ $# -eq 0 && -n "$input_music" && -n "$output_music" ]]; then
    set -- "$input_music" "$downscale_bitrate" "$upscale" "$output_music"
fi
exec "$python_bin" "$script_dir/tools/music_test.py" "$@"
