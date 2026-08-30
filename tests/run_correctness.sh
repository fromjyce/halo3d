#!/usr/bin/env bash
# Correctness harness: build every variant, run the serial references once
# as golden output, then run every MPI variant across several rank counts
# and process-grid shapes and require byte-exact match via cmp.
#
# This is the harness the project's global rule #1 requires before any
# feature work: if this script fails, nothing downstream (bench numbers,
# README claims) is trustworthy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
BUILD="$ROOT/.build"
mkdir -p "$BUILD"

CC=${CC:-cc}
MPICC=${MPICC:-mpicc}
MPIRUN=${MPIRUN:-mpirun}
CFLAGS="-O2 -Wall -Wextra -std=c11"

echo "== building =="
"$CC" $CFLAGS -o "$BUILD/serial2d" "$SRC/serial2d.c" -lm
"$CC" $CFLAGS -o "$BUILD/serial3d" "$SRC/serial3d.c" -lm
"$MPICC" $CFLAGS -o "$BUILD/jacobi2d_mpi" "$SRC/jacobi2d_mpi.c" -lm
"$MPICC" $CFLAGS -o "$BUILD/jacobi3d_mpi" "$SRC/jacobi3d_mpi.c" -lm

WORK="$BUILD/correctness"
mkdir -p "$WORK"
FAIL=0

echo "== S1: 2D 5-point, 1D decomposition, blocking Sendrecv =="
"$BUILD/serial2d" 16 16 10 "$WORK/s2d_gold.bin" > /dev/null
for P in 1 2 4; do
    "$MPIRUN" --oversubscribe -n "$P" "$BUILD/jacobi2d_mpi" 16 16 10 "$WORK/s2d_p$P.bin" > /dev/null 2>&1
    if cmp -s "$WORK/s2d_gold.bin" "$WORK/s2d_p$P.bin"; then
        echo "  P=$P: OK (bitwise identical)"
    else
        echo "  P=$P: FAIL (diverged from serial reference)"
        FAIL=1
    fi
done

echo "== S2: 3D 7-point, Cartesian decomposition, subarray halos, non-blocking + overlap =="
"$BUILD/serial3d" 32 32 32 10 "$WORK/s3d_gold.bin" > /dev/null
declare -a SHAPES=("1 1 1 1" "2 2 1 1" "4 2 2 1" "8 2 2 2" "8 8 1 1" "8 1 1 8")
for shape in "${SHAPES[@]}"; do
    read -r P PX PY PZ <<< "$shape"
    "$MPIRUN" --oversubscribe -n "$P" "$BUILD/jacobi3d_mpi" 32 32 32 10 \
        "$WORK/s3d_p${P}_${PX}x${PY}x${PZ}.bin" "$PX" "$PY" "$PZ" > /dev/null 2>&1
    if cmp -s "$WORK/s3d_gold.bin" "$WORK/s3d_p${P}_${PX}x${PY}x${PZ}.bin"; then
        echo "  P=$P grid=${PX}x${PY}x${PZ}: OK (bitwise identical)"
    else
        echo "  P=$P grid=${PX}x${PY}x${PZ}: FAIL (diverged from serial reference)"
        FAIL=1
    fi
done

echo "== S2: divisibility guard rejects an invalid process grid =="
if "$MPIRUN" --oversubscribe -n 4 "$BUILD/jacobi3d_mpi" 16 16 16 6 "$WORK/bad.bin" 3 1 1 > /dev/null 2>&1; then
    echo "  FAIL: invalid grid was accepted"
    FAIL=1
else
    echo "  OK (rejected as expected)"
fi

if [ "$FAIL" -ne 0 ]; then
    echo "CORRECTNESS HARNESS: FAILED"
    exit 1
fi
echo "CORRECTNESS HARNESS: PASSED"
