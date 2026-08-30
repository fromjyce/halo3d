/* Shared field layout, boundary condition, and I/O helpers used by every
 * solver variant (serial2d, serial3d, jacobi2d_mpi, jacobi3d_mpi). Keeping
 * these identical across variants is what makes bitwise-identical output
 * checkable at all -- see docs/DESIGN.md "why bitwise reproducibility is
 * achievable here". */
#ifndef HALO3D_COMMON_H
#define HALO3D_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Row-major, x fastest-varying. Local arrays carry a 1-cell ghost layer on
 * every side, so a padded array of logical size (nx x ny x nz) is allocated
 * as (nx+2) x (ny+2) x (nz+2) and indexed with 1-based interior coordinates. */
static inline size_t idx3(int i, int j, int k, int pnx, int pny) {
    return (size_t)k * (size_t)pny * (size_t)pnx + (size_t)j * (size_t)pnx + (size_t)i;
}

static inline size_t idx2(int i, int j, int pnx) {
    return (size_t)j * (size_t)pnx + (size_t)i;
}

/* Fixed synthetic Dirichlet boundary field. This is NOT a manufactured
 * solution to Laplace's equation -- it is just a deterministic function of
 * global coordinates, chosen so every rank/decomposition computes the exact
 * same boundary values without communication. Correctness here means "the
 * parallel decomposition reproduces the serial arithmetic exactly", not
 * "this converges to a known analytic field". See docs/DESIGN.md.
 *
 * Deliberately affine, not sin(pi*x)-style: a product-of-sines vanishes at
 * every point where any coordinate is 0 or 1, which is every boundary point
 * on a [0,1]^d cube -- that would silently zero the entire field and turn
 * "checksum matches" into a trivial pass. */
static inline double boundary_value(double x, double y, double z) {
    return 1.0 + x + 2.0 * y + 3.0 * z;
}

static inline double boundary_value_2d(double x, double y) {
    return 1.0 + x + 2.0 * y;
}

/* Little-endian raw double dump: used by every variant so a plain `cmp`
 * between two output files is a valid bitwise-identity check. */
static inline int write_field_binary(const char *path, const double *data, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(data, sizeof(double), n, f);
    fclose(f);
    return written == n ? 0 : -1;
}

static inline double checksum(const double *data, size_t n) {
    /* Fixed left-to-right accumulation order -- required for the sum itself
     * to be reproducible across runs; the field values it sums are already
     * guaranteed bitwise-identical by construction. */
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += fabs(data[i]);
    return s;
}

#endif
