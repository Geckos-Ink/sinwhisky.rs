#!/usr/bin/env python3
"""Optional live C/Python comparison; the Rust crate itself has no Python dependency."""
import argparse
import importlib.util
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_root", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "target/release/sinwhisky")
    parser.add_argument("--write-fixtures", action="store_true", help="Regenerate frozen Python-reference vectors after C/Rust validation")
    args = parser.parse_args()
    legacy = args.legacy_root.resolve()
    spec = importlib.util.spec_from_file_location("whisky_reference", legacy / "v2/whisky.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise SystemExit("A C compiler is required; comparison was NOT run")
    rng = np.random.default_rng(42)
    cases = [
        [0.0, 1.0, 0.0, -1.0, 0.0],
        np.sin(np.linspace(0, 4 * np.pi, 40)),
        2 * np.arange(10) - 3.0,
        [5.0] * 8,
        rng.standard_normal(50),
        [0.0, 100.0, 0.0],
    ]
    rows = ["# zoom\tneighbors\taperture\tinput_f64\tpython_output_f64"]
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix="sinwhisky-cross-") as temp:
        cli = pathlib.Path(temp) / "sw_cli"
        subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                        str(legacy / "sinwhisky_c/sinwhisky.c"), str(legacy / "sinwhisky_c/sw_cli.c"),
                        "-lm", "-o", str(cli)], check=True)
        for case_index, samples in enumerate(cases):
            for zoom in (0, 1, 3, 6):
                for neighbors in (1, 2, 3):
                    for aperture in (0, 1):
                        data = " ".join(format(float(x), ".17g") for x in samples)
                        options = [str(zoom), str(neighbors), str(aperture)]
                        rust = np.fromstring(subprocess.run([str(args.binary.resolve()), *options], input=data, text=True, capture_output=True, check=True).stdout, sep=" ")
                        c = np.fromstring(subprocess.run([str(cli), *options], input=data, text=True, capture_output=True, check=True).stdout, sep=" ")
                        py = reference.sinwhisky_resample(samples, zoom, neighbors=neighbors, use_aperture=bool(aperture))
                        np.testing.assert_allclose(rust, py, atol=1e-9, rtol=1e-9)
                        np.testing.assert_allclose(rust, c, atol=2e-4, rtol=2e-4)
                        comparisons += 1
                        if case_index in (0, 2, 3, 5) and zoom == 3:
                            rows.append("\t".join([*options, data, " ".join(format(x, ".17g") for x in py)]))
    if args.write_fixtures:
        path = ROOT / "tests/fixtures/uniform.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n")
    print(f"Passed {comparisons} Rust/C/Python comparisons")


if __name__ == "__main__":
    main()
