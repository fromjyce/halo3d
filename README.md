# halo3d

A 3D 7-point Laplacian/Jacobi stencil solver decomposed across MPI ranks in
a Cartesian topology, exchanging halo (ghost-cell) data each step via
derived MPI datatypes, with communication overlapped against computation.

**Novel angle:** the halo exchange is built entirely on `MPI_Type_create_subarray`
against the padded local array — no manual pack/unpack buffers, for either
the per-iteration halo exchange or the final output gather — combined with a
non-blocking exchange that overlaps each rank's core-cell computation
against the in-flight halo messages. See `docs/DESIGN.md` for the two hard
parts and the alternatives considered but not taken.

## Features

- **3D Cartesian decomposition**: `MPI_Cart_create` process grid (explicit
  `PX PY PZ` or auto-factorized via `MPI_Dims_create`), with a divisibility
  guard that aborts with a clear error instead of silently mis-decomposing
- **Derived-datatype halo exchange**: `MPI_Type_create_subarray` describes
  each of the 6 non-contiguous face slabs directly against the local
  array's existing strides — built once at setup, reused every iteration
- **Non-blocking overlap**: `Isend`/`Irecv` posted up front; core cells
  (≥2 cells from every face, no ghost dependency) compute while halo
  messages are in flight; boundary-shell cells compute after `Waitall`
- **1D stepping-stone variant** (`jacobi2d_mpi.c`): 2D 5-point stencil,
  blocking `MPI_Sendrecv` on contiguous rows — the deliberate contrast that
  motivates derived datatypes once decomposition goes fully 3D
- **Bitwise-identical correctness**: validated against a serial reference
  with `cmp` (zero tolerance, not an approximate check) across rank counts
  and process-grid shapes, including degenerate flat shapes that exercise
  the no-overlap-possible code path
- **Communication-overhead instrumentation**: `MPI_Wtime` around each
  `MPI_Waitall`, reported as % of per-step wall time, maxed across ranks

## Architecture

```
Global grid (NX x NY x NZ)
        ↓
MPI_Cart_create — 3D process grid (PX x PY x PZ)
        ↓
Per-rank local subgrid + 1-cell ghost layer
        ↓
Post Irecv/Isend (6 faces, MPI_Type_create_subarray)
   ↳ overlapped with → core-cell Jacobi update (no ghost dependency)
        ↓
Waitall(recv) → boundary-shell Jacobi update (uses arrived halo)
        ↓
Waitall(send) → double-buffer swap → next iteration
        ↓
Point-to-point gather (subarray-typed) → output field
```

## Installation

```bash
git clone https://github.com/fromjyce/halo3d.git
cd halo3d

# Requires an MPI implementation
brew install open-mpi                          # macOS
# apt install libopenmpi-dev openmpi-bin        # Debian/Ubuntu

make                                            # builds src/*.c into .build/
```

## Configuration

No config file — everything is a CLI argument, deliberately (see
`docs/DESIGN.md` for the uniform-decomposition tradeoff this implies):

- Grid size: `NX NY NZ`
- Iteration count: `NITERS`
- Process grid: optional `PX PY PZ` (otherwise auto-factorized); must evenly
  divide the grid or the solver aborts rather than silently mis-decomposing

## Usage

### Correctness harness
```bash
tests/run_correctness.sh
```

### Benchmark
```bash
bench/run_halo_overhead.sh
```

### Run a variant directly
```bash
mpirun -n 8 .build/jacobi3d_mpi 64 64 64 200 out.bin         # auto process-grid
mpirun -n 8 .build/jacobi3d_mpi 64 64 64 200 out.bin 2 2 2    # explicit PX PY PZ
```

## Tech Stack

- **Language**: C (C11)
- **Parallelism**: MPI (Open MPI 5.0.10 tested; MPICH not yet cross-tested — see `docs/DESIGN.md`)
- **Domain decomposition**: `MPI_Cart_create`, `MPI_Cart_shift`, `MPI_Dims_create`
- **Halo exchange**: `MPI_Type_create_subarray`, `MPI_Isend`/`MPI_Irecv`, `MPI_Waitall`
- **Build**: Make
- **Testing**: bash harness + `cmp` (bitwise-identical, zero-tolerance correctness check)

## Evaluation Metrics

- **Correctness**: bitwise-identical (`cmp`) output vs. serial reference
  across 1–8 ranks and 6 process-grid shapes — full matrix in `docs/RESULTS.md`
- **Communication overhead**: recv-wait / send-wait as % of per-step wall
  time — see `docs/RESULTS.md`; laptop-only, explicitly labeled as not a
  scaling study
- **TODO: measure** — MPICH comparison, multi-node strong/weak scaling,
  roofline (see `docs/DESIGN.md` → Future Work)

## Repo Layout

```
src/            serial references + MPI variants
tests/          correctness harness (bitwise-identical vs serial, across rank counts/shapes)
bench/          benchmark script + raw committed output under bench/results/
docs/DESIGN.md  what it does, the hard parts, design decisions and alternatives, limitations
docs/RESULTS.md measured numbers with the exact reproducing command
```

## Contact

If you come across any issues, have suggestions for improvement, or want to
discuss further enhancements, feel free to contact me at
[jaya2004kra@gmail.com](mailto:jaya2004kra@gmail.com). Your feedback is
greatly appreciated.

## License

All the code and resources in this repository are licensed under the GNU
General Public License. You are free to use, modify, and distribute the
code under the terms of this license. However, I do not take responsibility
for the accuracy or reliability of the programs.
