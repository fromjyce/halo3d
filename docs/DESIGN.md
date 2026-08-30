# halo3d — technical design

## What it does

A 7-point Laplacian Jacobi stencil solved over a 3D grid, decomposed across
MPI ranks in a Cartesian topology. Each rank owns a rectangular block of the
grid plus a 1-cell ghost layer on every face; each iteration, ranks exchange
their boundary layers with their face neighbors (the "halo exchange"), then
update every owned cell as the average of its 6 face neighbors.

Two variants are implemented, each the validation target for the one after it:

| Variant | Decomposition | Exchange | File |
|---|---|---|---|
| S0 serial reference (2D, 3D) | none | none | `src/serial2d.c`, `src/serial3d.c` |
| S1 | 2D, 1D (rows split along y) | blocking `MPI_Sendrecv`, contiguous rows | `src/jacobi2d_mpi.c` |
| S2 | 3D, split in x/y/z via `MPI_Cart_create` | non-blocking `Isend`/`Irecv` + derived datatypes, overlapped with compute | `src/jacobi3d_mpi.c` |

S1 exists as a deliberate stepping stone: with the domain split only along
y and x fastest-varying in memory, a halo row is a single contiguous run of
doubles, so a plain `MPI_Sendrecv(..., MPI_DOUBLE, ...)` is enough. S2 removes
that convenience by splitting in all three dimensions, which is what actually
motivates derived datatypes (see below).

## The hard part

Not the stencil loop — averaging 6 neighbors is one line. The two things that
are actually hard:

**1. Describing a non-contiguous halo face without manual packing.**
Once the domain is split in x, y, *and* z, an x-face or y-face halo is a 2D
slab embedded in a 3D array with a stride — not contiguous. The naive fix is
to pack it into a scratch buffer by hand (loop, copy into a flat buffer, send
the buffer, loop again on the receive side to unpack). `src/jacobi3d_mpi.c`
instead builds one `MPI_Type_create_subarray` per face at setup:

```c
MPI_Type_create_subarray(3, sizes, subsizes, start, MPI_ORDER_C, MPI_DOUBLE, &send_t[f]);
```

`sizes` describes the full padded local array, `subsizes`/`start` carve out
the 1-cell-thick face slab directly against that array's existing strides.
Committed once, reused for all `niters` iterations — no per-iteration pack/
unpack loop, no scratch buffer, no host-side memory traffic beyond the field
itself. MPI's own implementation is responsible for gathering the strided
data at the wire (whatever it does internally, it's day-one-optimized MPI
library code, not a loop I wrote once and never profiled).

Same trick used for the output gather (`jacobi3d_mpi.c`'s final assembly):
each rank's block is received directly into its final position in the global
array via a per-rank subarray receive type, instead of receiving into a flat
buffer and unpacking with index arithmetic on rank 0.

*Alternative considered:* manual packing. Simpler to reason about in
isolation, and in some MPI implementations/network combinations it's
actually faster than derived datatypes — this is implementation-dependent
and contested in the HPC literature, not a settled question. I did not
build an A/B harness against a hand-packed variant (out of scope for this
pass — see Future Work); the derived-datatype choice here is justified by
code clarity and by matching the project brief's stated "hard part," not by
a measured win.

**2. Overlapping communication with computation without a data race.**
The 6 face exchanges take real wall time (network/shared-memory latency,
not FLOPs). A rank's "core" cells — anything at least 2 cells from every
face — don't depend on any ghost data at all, so they can be computed while
the halo exchange for *this* iteration is still in flight. The scheme:

1. Post all 6 `MPI_Irecv` (into the ghost regions, via the recv subarray types).
2. Post all 6 `MPI_Isend` (from the interior boundary layer, via the send subarray types).
3. Compute the core (no dependency on the in-flight messages).
4. `MPI_Waitall` the receives.
5. Compute the boundary shell (the cells the just-arrived halo unblocks).
6. `MPI_Waitall` the sends, *then* swap the double-buffer pointers.

Step 6's placement is the correctness-critical part: the send buffer is
`u_old`, which is about to become `u_new` (the write target) on the next
iteration. If the swap happened before the sends finished draining, the
network layer could still be reading `u_old` while the next iteration
started overwriting it through the new `u_new` alias. Waiting on the sends
before swapping costs a small amount of overlap opportunity but removes
that race outright.

*Alternative considered:* persistent requests (`MPI_Send_init`/`MPI_Recv_init`,
`MPI_Startall`) to amortize request setup across iterations, and/or
pipelining so iteration *n+1*'s sends can start before iteration *n*'s are
fully drained (would need triple-buffering, not double). Not built — see
Future Work. The measured `sendwait_pct` in `docs/RESULTS.md` is the
concrete cost of *not* doing this.

## Why bitwise-identical correctness is achievable here

The acceptance bar (bitwise-identical output to the serial reference, at
every rank count and process-grid shape) is only achievable because the
per-cell Jacobi update has a *fixed operation order* that doesn't depend on
decomposition:

```c
double sum = u_old[west] + u_old[east] + u_old[south] + u_old[north] + u_old[down] + u_old[up];
u_new[c] = sum / 6.0;
```

This is a purely local computation — no global reduction, no order-dependent
floating-point accumulation across ranks. Every rank computes exactly the
sum-then-divide any serial run would, for any cell it owns, regardless of
how many other ranks exist or how the domain is sliced. That's what makes
`cmp` a valid correctness check instead of a "close enough" tolerance check.
(This would *not* hold for, say, a distributed dot-product or a norm — those
require picking a fixed global reduction order to get bitwise reproducibility,
which OpenMPI's default reduction algorithms don't guarantee across different
rank counts.)

The boundary condition field itself is also decomposition-independent by
construction: it's `1 + x + 2y + 3z` evaluated at each cell's *global*
coordinate (`src/common.h`), not something derived from neighbor state.

One prior bug worth recording (fixed before any bench numbers were taken):
the first boundary field was `sin(πx)·sin(πy)·sin(πz)`. Every boundary cell
of a `[0,1]^3` cube has at least one coordinate equal to 0 or 1, and
`sin(0) = sin(π) = 0`, so that field is identically zero on the *entire*
boundary — the whole simulation silently computed all zeros, and "checksums
match" would have been a trivially true, meaningless check. Caught by
eyeballing the checksum output (`1.09e-14` instead of a plausible O(1e3)
value for a 32³ grid) before writing any test that depended on it.

## Known limitations

- **Uniform decomposition only.** `NX % dims[0] == 0` (and similarly for
  y, z) is required and enforced with an explicit abort; there's no
  block-cyclic or remainder-handling path for grid sizes that don't divide
  evenly by the process grid. Real HPC codes often make the same
  simplification; a general version would need per-rank-variable local
  sizes threaded through the datatype construction.
- **Ghost width is fixed at 1**, matching the 7-point stencil. A higher-order
  stencil (e.g. a 4th-order finite difference needing 2 neighbors per axis)
  would need a wider halo and the subarray `subsizes` would change
  accordingly — not parameterized here.
- **No periodic boundaries.** `MPI_Cart_create` is called with
  `periods = {0,0,0}`; edge ranks get `MPI_PROC_NULL` neighbors via
  `MPI_Cart_shift`, which MPI silently no-ops sends/receives to. Correct,
  but periodic domains (common in turbulence/CFD codes) aren't exercised.
- **Fixed boundary values only (Dirichlet)** — no source term, no
  Neumann/mixed conditions, no time-varying boundary. This is a systems
  project about decomposition and communication, not a numerics project; see
  "Why bitwise-identical correctness is achievable" above for why the
  boundary field was chosen the way it was.
- **Output gather is not overlapped or scalable** — a serialized
  point-to-point loop on rank 0 at the very end of the run, fine for
  writing a `32³`–`128³` field once, would not fine for a production-scale
  checkpoint path (that would want parallel I/O, e.g. MPI-IO or HDF5).

## Future work (not built, scoped out of this pass)

- A/B harness: derived-datatype halo exchange vs. hand-packed, across
  OpenMPI and MPICH, per the parent project brief's S3 stage. This is the
  one place the brief explicitly says not to assume a winner.
- `MPI_Neighbor_alltoallw` single-call exchange in place of 6 explicit
  Isend/Irecv pairs (S3).
- Persistent requests / iteration-pipelined sends (see "hard part #2" above).
- A temporal-blocking / wavefront-diamond variant (the project brief's
  "novel angle" list) — not attempted; this pass stops at S2.
- Real multi-node strong/weak scaling. Everything measured in
  `docs/RESULTS.md` is single-laptop, oversubscribed MPI ranks sharing 
  physical cores — see the caveat there. It answers "is the parallel
  decomposition correct and does overlap reduce measured wait time," not
  "does this scale."

## Toolchain (versions actually used)

- macOS 26.6.2, arm64 (Apple Silicon)
- Apple clang 21.0.0 (`cc`, `mpicc` wraps the same compiler)
- Open MPI 5.0.10 (Homebrew `open-mpi` formula) — MPICH was not
  cross-tested in this pass (see Future Work: the brief calls for
  benchmarking both, since behavior differs; only OpenMPI was used here)
- `mpirun --oversubscribe` used throughout for local correctness runs,
  since the laptop has fewer physical cores than the rank counts tested
  (up to 8). This is a correctness-only run mode — see `docs/RESULTS.md`
  for why it is not a scaling measurement.

## How to reproduce

```
tests/run_correctness.sh       # builds everything, bitwise-identity checks
bench/run_halo_overhead.sh     # communication-overhead numbers, writes bench/results/
```
