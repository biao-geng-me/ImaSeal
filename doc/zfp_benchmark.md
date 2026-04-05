# HDF5 Compression Benchmark — ZFP via hdf5plugin

**Date:** 2026-04-04  
**Dataset:** `env1/path1` — 272 frames, 2048 × 1024 grid, ~20 GB raw `.plt`  
**Tool:** `plt_to_hdf5.py`  
**Compression:** ZFP via `hdf5plugin`, chunked `(1, ni, nj)` — one block per frame

---

## Table 1 — All field variables (`u, v, p, div, vor, BL, GC`)

| Mode         | Size (MB) | Ratio  | Time (s) | Max \|err\| | Mean \|err\| |
|--------------|----------:|-------:|---------:|------------:|-------------:|
| Uncompressed |    15,989 |  1.00× |      93s |           — |            — |
| ZFP lossless |     7,123 |  2.24× |     139s |           — |            — |
| `1e-02`      |       289 |   55×  |      45s |    5.80e-03 |     7.73e-05 |
| `5e-02`      |       184 |   87×  |      40s |    2.34e-02 |     2.67e-04 |
| `1e-01`      |       134 |  119×  |      39s |    4.67e-02 |     4.38e-04 |
| `2e-01`      |        95 |  167×  |      37s |    9.35e-02 |     6.77e-04 |
| `4e-01`      |        70 |  228×  |      35s |    1.86e-01 |     8.92e-04 |

---

## Table 2 — Velocity only (`u, v`)

| Accuracy | Size (MB) | Time (s) | Max \|err\| | Mean \|err\| |
|----------:|----------:|---------:|------------:|-------------:|
| `1e-01`  |        83 |      18s |    4.66e-02 |     9.85e-04 |
| `2e-01`  |        59 |      17s |    9.27e-02 |     1.67e-03 |
| `4e-01`  |        41 |      16s |    1.86e-01 |     2.33e-03 |

---

## Discussion

**ZFP lossless is not worth it here.** At 2.24× compression it is slower than
uncompressed due to compression overhead exceeding I/O savings on this storage.
It would benefit more on slower network filesystems.

**Lossy ZFP is very effective for this dataset.** Large compression ratios
(55–228×) are driven by variables like `div`, `vor`, `BL`, and `GC` which are
either near-zero or near-binary across most of the domain and compress
extremely well. `u` and `v` are the hardest variables to compress — velocity-only
storage is only ~60% smaller than the full-field at the same accuracy, confirming
that the other 5 variables contribute disproportionately little to total file size.

**More compression = faster write**, not slower — ZFP writes fewer bytes so I/O
savings dominate the small extra compute cost.

**The ZFP accuracy bound is absolute**, not relative: `accuracy=1e-1` guarantees
`|reconstructed − original| ≤ 0.1` for every value. Actual max errors observed
were roughly half the tolerance; mean errors were 2–3 orders of magnitude smaller,
indicating errors are spatially localised near sharp gradients.

## Recommended settings

| Use case | Accuracy | Size (all vars) | Notes |
|---|---|---|---|
| Full archive | `1e-02` | 289 MB | Max error < 6e-3, safe for all analysis |
| Working / ML dataset | `1e-01` | 134 MB | Mean error < 5e-4, fast to load |
| Velocity-only (RL/sim) | `1e-01` | 83 MB | Compact, mean error < 1e-3 |
