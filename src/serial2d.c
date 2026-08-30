/* S1 stepping-stone -- serial reference for the 2D 5-point Jacobi stencil.
 * Golden output for jacobi2d_mpi. See serial3d.c for the same pattern in 3D
 * and docs/DESIGN.md for why S1 exists as a warm-up before the full 3D
 * decomposition in S2.
 *
 * usage: serial2d NX NY NITERS out.bin
 */
#include "common.h"

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s NX NY NITERS out.bin\n", argv[0]);
        return 1;
    }
    int NX = atoi(argv[1]), NY = atoi(argv[2]);
    int niters = atoi(argv[3]);
    const char *outpath = argv[4];

    int pnx = NX + 2, pny = NY + 2;
    size_t n = (size_t)pnx * pny;
    double *u_old = calloc(n, sizeof(double));
    double *u_new = calloc(n, sizeof(double));
    if (!u_old || !u_new) { fprintf(stderr, "alloc failed\n"); return 1; }

    double dx = 1.0 / (NX - 1), dy = 1.0 / (NY - 1);

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            double val = 0.0;
            if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
                val = boundary_value_2d(i * dx, j * dy);
            }
            u_old[idx2(i + 1, j + 1, pnx)] = val;
            u_new[idx2(i + 1, j + 1, pnx)] = val;
        }
    }

    for (int it = 0; it < niters; it++) {
        for (int j = 1; j <= NY - 2; j++) {
            for (int i = 1; i <= NX - 2; i++) {
                size_t c = idx2(i + 1, j + 1, pnx);
                double sum =
                    u_old[idx2(i, j + 1, pnx)] +
                    u_old[idx2(i + 2, j + 1, pnx)] +
                    u_old[idx2(i + 1, j, pnx)] +
                    u_old[idx2(i + 1, j + 2, pnx)];
                u_new[c] = sum / 4.0;
            }
        }
        double *tmp = u_old; u_old = u_new; u_new = tmp;
    }

    double *packed = malloc((size_t)NX * NY * sizeof(double));
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            packed[idx2(i, j, NX)] = u_old[idx2(i + 1, j + 1, pnx)];

    if (write_field_binary(outpath, packed, (size_t)NX * NY) != 0) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    printf("serial2d: %dx%d, %d iters, checksum=%.17g\n",
           NX, NY, niters, checksum(packed, (size_t)NX * NY));

    free(u_old); free(u_new); free(packed);
    return 0;
}
