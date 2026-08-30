#!/usr/bin/env bash
# Measures communication overhead as a fraction of per-step wall time for
# the S2 3D solver, across rank counts, on a fixed problem size. This is
# the number docs/RESULTS.md quotes -- re-run this script to reproduce it.
#
# What is actually measured (see src/jacobi3d_mpi.c): wall time inside the
# two MPI_Waitall calls (waiting for halo receives, then waiting for sends
# to drain before the buffer swap), as a percentage of total step time,
# maxed over ranks via MPI_Reduce(MAX). This is a laptop, oversubscribed,
# shared-memory-transport measurement -- see docs/RESULTS.md for the
# "what this number is not" caveat.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/.build"
MPIRUN=${MPIRUN:-mpirun}
BIN="$BUILD/jacobi3d_mpi"

if [ ! -x "$BIN" ]; then
    echo "build missing, run tests/run_correctness.sh first (it builds this binary)" >&2
    exit 1
fi

NX=${NX:-64}
NY=${NY:-64}
NZ=${NZ:-64}
NITERS=${NITERS:-200}
OUT="$ROOT/bench/results/halo_overhead_$(date +%Y%m%d_%H%M%S).txt"

echo "# halo3d S2 communication-overhead bench" > "$OUT"
echo "# host: $(uname -a)" >> "$OUT"
echo "# command per row: mpirun --oversubscribe -n P $BIN $NX $NY $NZ $NITERS /dev/null" >> "$OUT"
echo "# grid=${NX}x${NY}x${NZ} niters=$NITERS" >> "$OUT"
printf "%-6s %-14s %-24s %-24s\n" "ranks" "step_ms(max)" "recvwait_pct(max)" "sendwait_pct(max)" >> "$OUT"

for P in 1 2 4 8; do
    LINE=$("$MPIRUN" --oversubscribe -n "$P" "$BIN" "$NX" "$NY" "$NZ" "$NITERS" /dev/null 2>&1 >/dev/null | tail -1)
    STEP=$(echo "$LINE" | grep -oE 'step_ms\(max\)=[0-9.]+' | cut -d= -f2)
    RW=$(echo "$LINE" | grep -oE 'recvwait_pct_of_step\(max\)=[0-9.]+' | cut -d= -f2)
    SW=$(echo "$LINE" | grep -oE 'sendwait_pct_of_step\(max\)=[0-9.]+' | cut -d= -f2)
    printf "%-6s %-14s %-24s %-24s\n" "$P" "$STEP" "$RW" "$SW" | tee -a "$OUT"
done

echo "raw output written to $OUT"
