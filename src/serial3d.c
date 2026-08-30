/* S0 -- serial reference for the 3D 7-point Jacobi stencil.
 *
 * This is the golden output that every MPI variant (jacobi3d_mpi at any
 * rank count / process grid) must reproduce byte-for-byte. See
 * docs/DESIGN.md for why exact reproducibility is a reasonable bar here.
 *
 * usage: serial3d NX NY NZ NITERS out.bin
 */
#include "common.h"

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s NX NY NZ NITERS out.bin\n", argv[0]);
        return 1;
    }
    int NX = atoi(argv[1]), NY = atoi(argv[2]), NZ = atoi(argv[3]);
    int niters = atoi(argv[4]);
    const char *outpath = argv[5];

    int pnx = NX + 2, pny = NY + 2, pnz = NZ + 2;
    size_t n = (size_t)pnx * pny * pnz;
    double *u_old = calloc(n, sizeof(double));
    double *u_new = calloc(n, sizeof(double));
    if (!u_old || !u_new) { fprintf(stderr, "alloc failed\n"); return 1; }

    double dx = 1.0 / (NX - 1), dy = 1.0 / (NY - 1), dz = 1.0 / (NZ - 1);

    /* Interior offset by 1 for the ghost/pad layer, even though a
     * single-rank serial run never touches ghost cells -- keeps indexing
     * identical to the MPI variant's local-array convention. */
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                double val = 0.0;
                if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
                    val = boundary_value(i * dx, j * dy, k * dz);
                }
                u_old[idx3(i + 1, j + 1, k + 1, pnx, pny)] = val;
                u_new[idx3(i + 1, j + 1, k + 1, pnx, pny)] = val;
            }
        }
    }

    for (int it = 0; it < niters; it++) {
        for (int k = 1; k <= NZ - 2; k++) {
            for (int j = 1; j <= NY - 2; j++) {
                for (int i = 1; i <= NX - 2; i++) {
                    size_t c = idx3(i + 1, j + 1, k + 1, pnx, pny);
                    double sum =
                        u_old[idx3(i, j + 1, k + 1, pnx, pny)] +
                        u_old[idx3(i + 2, j + 1, k + 1, pnx, pny)] +
                        u_old[idx3(i + 1, j, k + 1, pnx, pny)] +
                        u_old[idx3(i + 1, j + 2, k + 1, pnx, pny)] +
                        u_old[idx3(i + 1, j + 1, k, pnx, pny)] +
                        u_old[idx3(i + 1, j + 1, k + 2, pnx, pny)];
                    u_new[c] = sum / 6.0;
                }
            }
        }
        double *tmp = u_old; u_old = u_new; u_new = tmp;
    }

    /* Pack out the logical (non-padded) field, k-major/j/i, matching what
     * the MPI variant assembles via MPI_Gatherv. */
    double *packed = malloc((size_t)NX * NY * NZ * sizeof(double));
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
            for (int i = 0; i < NX; i++)
                packed[idx3(i, j, k, NX, NY)] = u_old[idx3(i + 1, j + 1, k + 1, pnx, pny)];

    if (write_field_binary(outpath, packed, (size_t)NX * NY * NZ) != 0) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    printf("serial3d: %dx%dx%d, %d iters, checksum=%.17g\n",
           NX, NY, NZ, niters, checksum(packed, (size_t)NX * NY * NZ));

    free(u_old); free(u_new); free(packed);
    return 0;
}
