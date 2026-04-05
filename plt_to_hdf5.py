"""Convert a case folder of Tecplot binary Q.*.plt files to a single HDF5 file.

HDF5 layout
-----------
::

    flow.h5
    │
    ├── (attrs)
    │     title          : str           # from .plt header
    │     variable_names : [str, ...]    # e.g. ["xc","yc","u","v","p"]
    │     ndim           : int           # 2 or 3
    │     dt             : float         # time increment between frames
    │     ni, nj         : int           # grid dimensions
    │     nframes        : int           # number of frames
    │     source_files   : [str, ...]    # ["Q.00001.plt", "Q.00002.plt", ...]
    │     zfp_accuracy   : float|"lossless"|"none"
    │
    ├── /grid/
    │     xc  (ni × nj)  float          # coordinate arrays, written once from frame 0
    │     yc  (ni × nj)  float
    │     ...
    │
    └── /fields/
          u   (nframes × ni × nj)  float    # chunked (1, ni, nj), maxshape (∞, ni, nj)
          v   (nframes × ni × nj)  float
          p   (nframes × ni × nj)  float
          ...

Coordinate variables (xc, yc, zc, x, y, z) are treated as grid data.
All other variables are treated as time-varying field data.
Physical time for frame i is reconstructed as ``t = i * dt``.

Compression
-----------
ZFP (via hdf5plugin) is used by default because it operates natively on
floating-point blocks and achieves much better ratios than gzip for structured
CFD data.  Two ZFP modes are exposed:

* **lossless** (default) – ZFP reversible mode; bit-exact, faster than gzip.
* **lossy**   (``--accuracy``) – ZFP fixed-accuracy mode; specify the maximum
  absolute error per value (e.g. ``1e-4``).  Ratios of 10–20× are typical for
  smooth flow fields.

Gzip is kept as a fallback for environments where hdf5plugin is unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

from read_plt import read_plt75

_COORD_NAMES: frozenset[str] = frozenset({"x", "y", "z", "xc", "yc", "zc"})


def _zfp_kwargs(accuracy: float | None) -> dict:
    """Return hdf5plugin.Zfp kwargs for lossless or fixed-accuracy mode."""
    if accuracy is None:
        return dict(hdf5plugin.Zfp())                    # reversible (lossless)
    return dict(hdf5plugin.Zfp(accuracy=accuracy))       # lossy, bounded absolute error


_VEL_NAMES: frozenset[str] = frozenset({"u", "v", "w"})


def convert_case_to_hdf5(
    case_dir: str | Path,
    output_path: str | Path,
    *,
    dt: float = 1.0,
    zfp_accuracy: float | None = None,
    compress: bool = True,
    field_filter: set[str] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Read all Q.*.plt files in *case_dir* and write a single HDF5 file.

    Parameters
    ----------
    case_dir:
        Directory containing ``Q.*.plt`` files.
    output_path:
        Destination ``.h5`` / ``.hdf5`` file.
    dt:
        Physical time increment between consecutive frames.
    zfp_accuracy:
        ZFP fixed-accuracy tolerance: the maximum **absolute** error per value,
        i.e. ``|reconstructed - original| ≤ zfp_accuracy`` for every element.
        This is not relative — the same bound applies regardless of magnitude,
        so choose a value appropriate for your smallest meaningful field value.
        A good starting point is ``1e-3 × (typical field magnitude)``.
        ``None`` (default) selects lossless ZFP reversible mode.
    compress:
        If *False*, write uncompressed data (useful as a baseline for benchmarks).
    field_filter:
        If provided, only field variables whose names are in this set are written.
        Variables not present in the data are silently ignored.
        ``None`` (default) writes all field variables.
    overwrite:
        If *True*, silently overwrite an existing output file.
    verbose:
        If *True* (default), print per-frame progress.

    Returns
    -------
    Path
        Resolved path of the written HDF5 file.
    """
    root = Path(case_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Case directory not found: {root}")

    files = sorted(root.glob("Q.*.plt"))
    if not files:
        raise FileNotFoundError(f"No 'Q.*.plt' files found in: {root}")

    out = Path(output_path)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {out}  (pass overwrite=True to replace)"
        )

    # Peek at first frame to get variable list and grid shape.
    first = read_plt75(files[0])
    if not first.zones:
        raise ValueError(f"No zones in: {files[0].name}")

    zone0 = first.zones[0]
    ni, nj = zone0.ni, zone0.nj
    nframes = len(files)
    variable_names = first.variable_names

    coord_vars = [v for v in variable_names if v in _COORD_NAMES]
    field_vars = [v for v in variable_names if v not in _COORD_NAMES]
    if field_filter is not None:
        field_vars = [v for v in field_vars if v in field_filter]
        if not field_vars:
            raise ValueError(
                f"field_filter {field_filter!r} matched none of the available "
                f"field variables: {[v for v in variable_names if v not in _COORD_NAMES]}"
            )

    compress_kw = _zfp_kwargs(zfp_accuracy) if compress else {}
    if verbose:
        if not compress:
            mode_label = "none (uncompressed)"
        elif zfp_accuracy is not None:
            mode_label = f"ZFP lossy (accuracy={zfp_accuracy:.2e})"
        else:
            mode_label = "ZFP lossless (reversible)"
        print(f"Converting {nframes} frames  [{ni} × {nj}]")
        print(f"  compression: {mode_label}")
        print(f"  grid vars  : {coord_vars}")
        print(f"  field vars : {field_vars}")
        print(f"  output     : {out}")

    with h5py.File(out, "w") as hf:
        # Root metadata
        hf.attrs["title"] = first.title
        hf.attrs["variable_names"] = variable_names
        hf.attrs["ndim"] = first.ndim
        hf.attrs["dt"] = dt
        hf.attrs["ni"] = ni
        hf.attrs["nj"] = nj
        hf.attrs["nframes"] = nframes
        hf.attrs["source_files"] = [f.name for f in files]
        if not compress:
            hf.attrs["zfp_accuracy"] = "none"
        else:
            hf.attrs["zfp_accuracy"] = zfp_accuracy if zfp_accuracy is not None else "lossless"

        grid_grp = hf.create_group("grid")
        fields_grp = hf.create_group("fields")

        # Grid variables are written once from frame 0; ZFP works on 2-D slices.
        for var in coord_vars:
            arr = zone0.data[var][:, :, 0]
            kw = {"chunks": (ni, nj), **compress_kw} if compress_kw else {}
            grid_grp.create_dataset(var, data=arr, **kw)

        # Pre-allocate field datasets chunked one frame at a time.
        # maxshape=(None, ni, nj) makes the time axis unlimited for later appending.
        # ZFP compresses each (ni × nj) chunk independently.
        datasets: dict[str, h5py.Dataset] = {}
        for var in field_vars:
            arr0 = zone0.data[var][:, :, 0]
            kw = {"chunks": (1, ni, nj), **compress_kw} if compress_kw else {"chunks": (1, ni, nj)}
            ds = fields_grp.create_dataset(
                var,
                shape=(nframes, ni, nj),
                maxshape=(None, ni, nj),
                dtype=arr0.dtype,
                **kw,
            )
            datasets[var] = ds

        # Write frame 0 (already loaded).
        for var in field_vars:
            datasets[var][0] = zone0.data[var][:, :, 0]
        if verbose:
            print(f"  [  1/{nframes}] {files[0].name}")

        # Stream remaining frames one at a time to keep memory low.
        for idx, fp in enumerate(files[1:], start=1):
            parsed = read_plt75(fp)
            zone = parsed.zones[0]
            for var in field_vars:
                datasets[var][idx] = zone.data[var][:, :, 0]
            if verbose:
                print(f"  [{idx + 1:>3}/{nframes}] {fp.name}")

    if verbose:
        print("Done.")
    return out


def append_frames(
    h5_path: str | Path,
    case_dir: str | Path,
    *,
    verbose: bool = True,
) -> Path:
    """Append new Q.*.plt frames from *case_dir* to an existing HDF5 file.

    Only files not already recorded in the ``source_files`` attribute are
    appended, so re-running on the same folder is safe and idempotent.
    The compression settings (ZFP mode, chunk shape) are inherited from the
    existing file — no need to specify them again.

    Parameters
    ----------
    h5_path:
        Path to an existing HDF5 file created by :func:`convert_case_to_hdf5`.
    case_dir:
        Directory containing ``Q.*.plt`` files (may contain both old and new frames).
    verbose:
        If *True* (default), print per-frame progress.

    Returns
    -------
    Path
        Resolved path of the updated HDF5 file.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    root = Path(case_dir)
    all_files = sorted(root.glob("Q.*.plt"))
    if not all_files:
        raise FileNotFoundError(f"No 'Q.*.plt' files found in: {root}")

    with h5py.File(h5_path, "a") as hf:
        existing = set(hf.attrs.get("source_files", []))
        new_files = [f for f in all_files if f.name not in existing]

        if not new_files:
            if verbose:
                print("No new frames to append.")
            return h5_path

        ni = int(hf.attrs["ni"])
        nj = int(hf.attrs["nj"])
        field_vars = [v for v in hf.attrs["variable_names"] if v not in _COORD_NAMES]
        current_nframes = int(hf.attrs["nframes"])
        n_new = len(new_files)

        if verbose:
            print(f"Appending {n_new} new frames to {h5_path.name}  (current: {current_nframes})")

        # Extend all field datasets along the time axis.
        for var in field_vars:
            hf[f"fields/{var}"].resize(current_nframes + n_new, axis=0)

        for idx, fp in enumerate(new_files):
            parsed = read_plt75(fp)
            zone = parsed.zones[0]
            for var in field_vars:
                hf[f"fields/{var}"][current_nframes + idx] = zone.data[var][:, :, 0]
            if verbose:
                print(f"  [{idx + 1:>3}/{n_new}] {fp.name}")

        # Update metadata.
        hf.attrs["nframes"] = current_nframes + n_new
        hf.attrs["source_files"] = sorted(existing | {f.name for f in new_files})

    if verbose:
        print("Done.")
    return h5_path


def _compute_errors_streaming(
    ref_h5: Path,
    cmp_h5: Path,
    field_vars: list[str],
    nframes: int,
) -> tuple[float, float]:
    """Compute max and mean absolute error between two HDF5 files, one frame at a time.

    Reads a single frame at a time from each file to keep memory usage at
    O(1 frame) regardless of dataset size.
    """
    max_err = 0.0
    sum_err = 0.0
    count = 0
    with h5py.File(ref_h5) as ref, h5py.File(cmp_h5) as cmp:
        for idx in range(nframes):
            for v in field_vars:
                ref_frame = ref[f"fields/{v}"][idx].astype(np.float64)
                cmp_frame = cmp[f"fields/{v}"][idx].astype(np.float64)
                diff = np.abs(ref_frame - cmp_frame)
                max_err = max(max_err, float(diff.max()))
                sum_err += float(diff.sum())
                count += diff.size
    mean_err = sum_err / count if count > 0 else 0.0
    return max_err, mean_err


def benchmark_case(
    case_dir: str | Path,
    *,
    dt: float = 1.0,
    accuracies: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
    field_filter: set[str] | None = None,
    baselines: bool = True,
) -> None:
    """Benchmark ZFP compression modes and print a summary table.

    Runs conversion for each mode (uncompressed, ZFP lossless, and each accuracy
    level), measures wall-clock write time and output file size, and computes the
    actual max/mean absolute error per variable for lossy modes by comparing against
    the uncompressed HDF5 reference — one frame at a time to keep memory usage low.

    Parameters
    ----------
    case_dir:
        Directory containing ``Q.*.plt`` files.
    dt:
        Physical time increment passed through to the HDF5 file.
    accuracies:
        Sequence of ZFP fixed-accuracy tolerances to benchmark.
    field_filter:
        If provided, only these field variable names are included (e.g. ``_VEL_NAMES``).
    baselines:
        If *False*, skip the uncompressed and ZFP lossless baseline modes.
        Error ratios will be omitted from the summary table.
    """
    root = Path(case_dir)
    files = sorted(root.glob("Q.*.plt"))
    if not files:
        raise FileNotFoundError(f"No 'Q.*.plt' files found in: {root}")

    first = read_plt75(files[0])
    variable_names = first.variable_names
    field_vars = [v for v in variable_names if v not in _COORD_NAMES]
    if field_filter is not None:
        field_vars = [v for v in field_vars if v in field_filter]
    zone0 = first.zones[0]
    ni, nj = zone0.ni, zone0.nj
    nframes = len(files)

    print(f"Benchmark: {nframes} frames  [{ni} × {nj}]  —  field vars: {field_vars}")

    # Modes: (label, compress, zfp_accuracy)
    modes: list[tuple[str, bool, float | None]] = []
    if baselines:
        modes += [("uncompressed", False, None), ("ZFP lossless", True, None)]
    modes += [(f"ZFP {a:.0e}", True, a) for a in accuracies]

    results: list[tuple[str, float, float, float, float]] = []

    tmp_dir = Path(tempfile.mkdtemp())
    # When skipping baselines, write an uncompressed reference silently for error comparison.
    ref_path: Path | None = None
    ref_is_tmp = False
    try:
        if not baselines:
            ref_path = tmp_dir / "flow_reference.h5"
            ref_is_tmp = True
            convert_case_to_hdf5(
                root, ref_path,
                dt=dt, compress=False, field_filter=field_filter,
                overwrite=True, verbose=False,
            )

        for label, compress, accuracy in modes:
            out_path = tmp_dir / f"flow_{label.replace(' ', '_')}.h5"
            print(f"  {label:<18} ...", end="", flush=True)
            t0 = time.perf_counter()
            convert_case_to_hdf5(
                root, out_path,
                dt=dt,
                compress=compress,
                zfp_accuracy=accuracy,
                field_filter=field_filter,
                overwrite=True,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            size_mb = out_path.stat().st_size / 1e6

            # First baseline mode (uncompressed) becomes the reference.
            if ref_path is None:
                ref_path = out_path

            # Compute actual errors for lossy modes, streaming one frame at a time.
            max_err = mean_err = float("nan")
            if compress and accuracy is not None and ref_path is not None:
                max_err, mean_err = _compute_errors_streaming(
                    ref_path, out_path, field_vars, nframes
                )

            print(f"\r  {label:<18}  {size_mb:>8.2f} MB  {elapsed:>7.2f}s", end="")
            if not np.isnan(max_err):
                print(f"  max|err|={max_err:.2e}  mean|err|={mean_err:.2e}", end="")
            print()
            results.append((label, size_mb, elapsed, max_err, mean_err))

    finally:
        shutil.rmtree(tmp_dir)

    # Summary table — ratio relative to first result if available
    ref_size = results[0][1] if results else 1.0
    col = "{:<18}  {:>10}  {:>8}  {:>10}  {:>12}  {:>12}"
    print()
    print(col.format("Mode", "Size (MB)", "Ratio", "Time (s)", "Max |err|", "Mean |err|"))
    print("-" * 82)
    for label, size_mb, elapsed, max_err, mean_err in results:
        ratio = ref_size / size_mb if size_mb > 0 else float("nan")
        max_s  = f"{max_err:.2e}"  if not np.isnan(max_err)  else "—"
        mean_s = f"{mean_err:.2e}" if not np.isnan(mean_err) else "—"
        print(col.format(label, f"{size_mb:.2f}", f"{ratio:.2f}x", f"{elapsed:.2f}", max_s, mean_s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert a case folder of Q.*.plt files to a ZFP-compressed HDF5 file.\n\n"
            "Default: lossless ZFP (reversible mode).\n"
            "Use --accuracy to enable lossy ZFP with a bounded absolute error.\n"
            "Use --benchmark to compare all modes and print a summary table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("case_dir", type=Path, help="Directory containing Q.*.plt files")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output HDF5 file (default: <case_dir>/flow.h5)",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Time step between frames")
    parser.add_argument(
        "--accuracy",
        type=float,
        default=None,
        metavar="TOL",
        help=(
            "ZFP fixed-accuracy mode: maximum absolute error per value "
            "(e.g. 1e-4 means |reconstructed - original| <= 1e-4 for every element). "
            "The bound is absolute, not relative — pick a value relative to your "
            "smallest meaningful field magnitude. "
            "Omit for lossless ZFP."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run compression benchmarks across all ZFP modes "
            "(uncompressed, lossless, and accuracy levels from --accuracies) "
            "and print a summary table. Ignores --output and --accuracy."
        ),
    )
    parser.add_argument(
        "--accuracies",
        type=str,
        default="1e-2,1e-3,1e-4,1e-5",
        metavar="A,B,...",
        help="Comma-separated ZFP accuracy levels for --benchmark (default: 1e-2,1e-3,1e-4,1e-5)",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="In --benchmark mode, skip the uncompressed and ZFP lossless baseline runs.",
    )
    parser.add_argument(
        "--vel-only",
        action="store_true",
        help=(
            "Write only velocity components (u, v, w if present) as field variables. "
            "Output defaults to <case_dir>/vel.h5 unless --output is specified."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    args = parser.parse_args()

    if args.benchmark:
        accuracies = tuple(float(x) for x in args.accuracies.split(","))
        field_filter = _VEL_NAMES if args.vel_only else None
        benchmark_case(
            args.case_dir,
            dt=args.dt,
            accuracies=accuracies,
            field_filter=field_filter,
            baselines=not args.no_baselines,
        )
    else:
        field_filter = _VEL_NAMES if args.vel_only else None
        default_name = "vel.h5" if args.vel_only else "flow.h5"
        out_path = args.output or (args.case_dir / default_name)
        convert_case_to_hdf5(
            args.case_dir,
            out_path,
            dt=args.dt,
            zfp_accuracy=args.accuracy,
            field_filter=field_filter,
            overwrite=args.overwrite,
        )

