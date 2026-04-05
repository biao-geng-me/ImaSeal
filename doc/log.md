## 2026-3-22

Tried the plt read code in the dmd code, but it couldn't read the `#!TDV75` format. Tried to hack it but it just couldn't work (the format is very different). Instead, asked Copilot to generate a read function based on the original Fortran write function. It did an amazing job.

## 2026-3-23

Vibe-coded the first playable version of `whisker_array_simulator.py` based on Matplotlib. Barely touched the code, never checked the code. Testing and iterating was all I did.

## 2026-4-4

Asked Copilot to write `plt_to_hdf5.py` to convert the `Q.*.plt` case folders into a single compressed HDF5 file. After some back and forth on compression options, settled on ZFP via `hdf5plugin` — it operates natively on floating-point blocks and is much better than gzip for structured CFD data. The script supports lossless ZFP (default) and lossy fixed-accuracy mode, extensible datasets for appending new frames later (which Copilot didn't use initially), and a `--vel-only` flag for writing only velocity components.

Ran benchmarks on `env1/path1` (272 frames, 2048×1024, ~20 GB raw). Lossy ZFP at `accuracy=1e-1` compresses the full field to 134 MB (119×) with a mean error under `5e-4`. Velocity-only at the same accuracy is 83 MB. Results are documented in `doc/zfp_benchmark.md`. 

