#!/usr/bin/env bash
# Usage: ./test_music.sh INPUT DOWNSCALE_BITRATE UPSCALE OUTPUT.wav [options]
# Alternatively set INPUT_MUSIC, DOWNSCALE_BITRATE, UPSCALE, OUTPUT_MUSIC and run with no arguments.
#
# Equalization, frequency isolation and compression (PAPER_EQ_COMPRESSION.md):
#   ./test_music.sh quick    INPUT OUTPUT_DIR [--start 60] [--seconds 5] [eq_audio options]
#   ./test_music.sh divide   INPUT OUTPUT_DIR [...]   bands, tonal and residual tracks
#   ./test_music.sh eq       INPUT OUTPUT_DIR [...]   divider followed by band gains
#   ./test_music.sh extrude  INPUT OUTPUT_DIR [...]   one band isolated, plus the remainder
#   ./test_music.sh compress INPUT OUTPUT_DIR [...]   SWL2 streams, bytes against FLAC
# `quick` runs all four on a short excerpt, plus a parametric stream to listen to.
# OUTPUT_DIR must not exist and must be outside the repository. Options such as
# --edges 250,2000, --gains-db 6,0,-12, --band 40,250, --tolerance 0,16,
# --search fast|full, --parametric 16, --upper 2000 go to examples/eq_audio.rs.
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

case "${1:-}" in
    quick|divide|eq|extrude|compress)
        command="$1"
        shift
        if [[ $# -lt 2 ]]; then
            printf 'Usage: %s %s INPUT OUTPUT_DIR [--start S] [--seconds D] [options]\n' "$0" "$command" >&2
            exit 2
        fi
        input="$1"
        output="$2"
        shift 2
        start=60
        seconds=5
        passed=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --start) start="$2"; shift 2 ;;
                --seconds) seconds="$2"; shift 2 ;;
                *) passed+=("$1"); shift ;;
            esac
        done
        if [[ "$command" == quick ]]; then
            command=all
            passed=(--parametric 16 ${passed[@]+"${passed[@]}"})
        fi
        ffmpeg="${FFMPEG:-ffmpeg}"
        ffprobe="${FFPROBE:-$(dirname -- "$(command -v "$ffmpeg" || echo ffmpeg)")/ffprobe}"
        for tool in "$ffmpeg" "$ffprobe" cargo; do
            if ! command -v "$tool" >/dev/null 2>&1; then
                printf '%s was not found (set FFMPEG/FFPROBE or load Cargo).\n' "$tool" >&2
                exit 2
            fi
        done
        if [[ ! -f "$input" ]]; then
            printf 'Input not found: %s\n' "$input" >&2
            exit 2
        fi
        if [[ -e "$output" ]]; then
            printf 'Output already exists: %s\n' "$output" >&2
            exit 2
        fi
        cargo build --offline --release --quiet --example eq_audio --manifest-path "$script_dir/Cargo.toml"
        binary="${CARGO_TARGET_DIR:-$script_dir/target}/release/examples/eq_audio"
        stream="$("$ffprobe" -v error -select_streams a:0 -show_entries stream=sample_rate,channels -of csv=p=0 "$input")"
        rate="${stream%%,*}"
        channels="${stream##*,}"
        temporary="$(mktemp -d)"
        trap 'rm -rf "$temporary"' EXIT
        "$ffmpeg" -hide_banner -loglevel error -nostdin -ss "$start" -t "$seconds" -i "$input" \
            -f s16le -acodec pcm_s16le -ar "$rate" -ac "$channels" "$temporary/input.raw"
        "$binary" "$command" "$temporary/input.raw" "$channels" "$rate" "$output" ${passed[@]+"${passed[@]}"}
        # Listening reference and, for byte counts, the lossless FLAC control.
        "$ffmpeg" -hide_banner -loglevel error -nostdin -f s16le -ar "$rate" -ac "$channels" \
            -i "$temporary/input.raw" "$output/source.wav"
        if [[ "$command" == all || "$command" == compress ]]; then
            "$ffmpeg" -hide_banner -loglevel error -nostdin -f s16le -ar "$rate" -ac "$channels" \
                -i "$temporary/input.raw" -c:a flac "$output/source.flac"
            flac_bytes="$(wc -c < "$output/source.flac" | tr -d ' ')"
            pcm_bytes="$(wc -c < "$temporary/input.raw" | tr -d ' ')"
            printf 'FLAC control: %s bytes (%s%% of %s PCM bytes)\n' "$flac_bytes" \
                "$(awk -v f="$flac_bytes" -v p="$pcm_bytes" 'BEGIN { printf "%.2f", 100 * f / p }')" "$pcm_bytes"
        fi
        printf 'Wrote %s\n' "$output"
        exit 0
        ;;
esac

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python 3 was not found. Set PYTHON=/path/to/python3.\n' >&2
    exit 2
fi
if [[ $# -eq 0 && -n "$input_music" && -n "$output_music" ]]; then
    set -- "$input_music" "$downscale_bitrate" "$upscale" "$output_music"
fi
exec "$python_bin" "$script_dir/tools/music_test.py" "$@"
