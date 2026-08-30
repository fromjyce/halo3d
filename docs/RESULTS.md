# halo3d — results

All numbers below come from committed raw output in `bench/results/` and
`tests/run_correctness.sh`'s own output — nothing here is hand-typed without
a command that reproduces it. Re-run the exact commands to regenerate.

## Correctness

```
$ tests/run_correctness.sh
```

Bitwise-identical (`cmp`, zero tolerance) output vs. the serial reference,
across:

- **S1** (2D, 1D row decomposition, blocking `MPI_Sendrecv`): 1, 2, 4 ranks.
- **S2** (3D, full Cartesian decomposition, subarray halos, non-blocking +
  overlap): 1, 2, 4, 8 ranks, across process-grid shapes `1x1x1`, `2x1x1`,
  `2x2x1`, `2x2x2`, and two degenerate flat shapes `8x1x1` / `1x1x8` that
  exercise the no-core-region code path (local subdomain has fewer than 3
  cells on an axis, so the "core cells computed while halo is in flight"
  optimization has nothing to do that iteration — the boundary-shell pass
  covers 100% of owned cells and it still has to be correct).
- The divisibility guard (`NX % PX == 0` etc.) correctly rejects an
  incompatible process grid (`16³` grid against a `3x1x1` grid for 4 ranks)
  instead of producing silently wrong output.

Last run: **all cases passed.**

## Communication overhead, S2, single laptop

```
$ NX=64 NY=64 NZ=64 NITERS=200 bench/run_halo_overhead.sh
```

Raw output: `bench/results/halo_overhead_20260830_124354.txt`

| ranks | step time (ms, max over ranks) | recv-wait (% of step, max) | send-wait (% of step, max) |
|---|---|---|---|
| 1 | 0.637 | 0.02 | 0.02 |
| 2 | 0.334 | 5.24 | 0.06 |
| 4 | 0.173 | 6.21 | 0.09 |
| 8 | 0.209 | 57.07 | 0.28 |

Re-running this script on the same laptop moves individual percentages by a
few points (this machine isn't a quiet benchmark box — no core pinning, other
processes running) but the qualitative shape is stable across repeats: near-
zero recv-wait at 1 rank, a jump once cross-rank halo exchange exists at all,
and a much larger jump at 8 ranks where the machine's physical cores are
oversubscribed 2:1. Treat single-run percentages as indicative, not precise.

What's measured, precisely (see `src/jacobi3d_mpi.c`): wall time inside
`MPI_Waitall` for the 6 halo receives, and separately for the 6 sends,
as a percentage of that rank's total step time, maxed across ranks via
`MPI_Reduce(..., MPI_MAX, ...)`.

**What this is not:** a scaling study. This is one Apple Silicon laptop
running `mpirun --oversubscribe` — at 8 ranks, ranks are sharing physical
cores, and the "network" is entirely shared-memory (OpenMPI's `sm`/`self`
BTL over the same machine, not a real interconnect). The clear trend —
recv-wait eating a larger fraction of a shrinking step time as rank count
rises — is the expected direction (fixed per-message latency against a
shrinking per-rank compute budget as the domain per rank gets smaller), but
the absolute percentages would look different on a real multi-node cluster
with an actual network fabric and no core oversubscription. Per the parent
project brief: laptop numbers prove the decomposition is correct and that
overlap gives *some* measured benefit relative to a fully synchronous
exchange; they are explicitly not reported as a scaling result.

`send-wait` staying small and flat while `recv-wait` grows is consistent
with the overlap scheme actually doing its job on the send side (sends are
posted early and their completion is rarely the bottleneck) while receive
completion — which gates the boundary-shell computation — is the part that
scales with rank count, as expected for a surface-to-volume-bound halo
exchange.

## TODO: measure

- MPICH comparison (brief calls for benchmarking both OpenMPI and MPICH;
  only OpenMPI was run in this pass).
- Multi-node strong/weak scaling (needs a real allocation — laptop-only
  numbers above are not a substitute; see `docs/DESIGN.md` Future Work).
- Roofline (arithmetic intensity vs. achieved GFLOP/s) — not collected.
- Derived-datatype vs. manual-packing A/B — not built this pass.
