/* S1 -- 2D 5-point Jacobi, 1D decomposition (rows split across ranks along
 * y), blocking MPI_Sendrecv halo exchange.
 *
 * Deliberately the simplest possible parallel version: with x fastest-
 * varying and the split along y, each rank's top/bottom ghost row is a
 * single contiguous run of NX doubles, so a plain MPI_DOUBLE count=NX
 * message is enough -- no derived datatype needed yet. That contrast is the
 * point: S2 (jacobi3d_mpi.c) splits in all three dimensions, so its halo
 * faces are *not* contiguous and derived datatypes stop being optional.
 * See docs/DESIGN.md.
 *
 * usage: mpirun -n P jacobi2d_mpi NX NY NITERS out.bin
 * Requires NY % P == 0 (uniform decomposition; see docs/DESIGN.md limitations).
 */
#include <mpi.h>
#include "common.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 5) {
        if (rank == 0) fprintf(stderr, "usage: %s NX NY NITERS out.bin\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int NX = atoi(argv[1]), NY = atoi(argv[2]);
    int niters = atoi(argv[3]);
    const char *outpath = argv[4];

    if (NY % nprocs != 0) {
        if (rank == 0)
            fprintf(stderr, "NY (%d) must be divisible by nprocs (%d)\n", NY, nprocs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int lny = NY / nprocs;
    int j0 = rank * lny; /* this rank's first global row */

    int pnx = NX + 2, plny = lny + 2;
    size_t n = (size_t)pnx * plny;
    double *u_old = calloc(n, sizeof(double));
    double *u_new = calloc(n, sizeof(double));

    double dx = 1.0 / (NX - 1), dy = 1.0 / (NY - 1);
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    for (int jl = 0; jl < lny; jl++) {
        int jg = j0 + jl;
        for (int i = 0; i < NX; i++) {
            double val = 0.0;
            if (i == 0 || i == NX - 1 || jg == 0 || jg == NY - 1) {
                val = boundary_value_2d(i * dx, jg * dy);
            }
            u_old[idx2(i + 1, jl + 1, pnx)] = val;
            u_new[idx2(i + 1, jl + 1, pnx)] = val;
        }
    }

    double t_halo_total = 0.0, t_step_total = 0.0;

    for (int it = 0; it < niters; it++) {
        double t_step0 = MPI_Wtime();

        double t_halo0 = MPI_Wtime();
        /* Ghost row above (from `up`'s last owned row) and below (from
         * `down`'s first owned row). Contiguous NX-double rows -- no
         * MPI_Type needed here, see file header. */
        MPI_Sendrecv(&u_old[idx2(1, 1, pnx)], NX, MPI_DOUBLE, up, 0,
                     &u_old[idx2(1, lny + 1, pnx)], NX, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_old[idx2(1, lny, pnx)], NX, MPI_DOUBLE, down, 1,
                     &u_old[idx2(1, 0, pnx)], NX, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t_halo_total += MPI_Wtime() - t_halo0;

        for (int jl = 1; jl <= lny; jl++) {
            int jg = j0 + (jl - 1);
            if (jg == 0 || jg == NY - 1) continue; /* boundary row, fixed */
            for (int i = 1; i <= NX - 2; i++) {
                size_t c = idx2(i + 1, jl, pnx);
                double sum =
                    u_old[idx2(i, jl, pnx)] +
                    u_old[idx2(i + 2, jl, pnx)] +
                    u_old[idx2(i + 1, jl - 1, pnx)] +
                    u_old[idx2(i + 1, jl + 1, pnx)];
                u_new[c] = sum / 4.0;
            }
            /* x-boundary columns of an owned interior row stay fixed */
            u_new[idx2(1, jl, pnx)] = u_old[idx2(1, jl, pnx)];
            u_new[idx2(NX, jl, pnx)] = u_old[idx2(NX, jl, pnx)];
        }
        double *tmp = u_old; u_old = u_new; u_new = tmp;
        t_step_total += MPI_Wtime() - t_step0;
    }

    /* Gather owned rows (no ghosts) onto rank 0 in global row order. */
    double *local_packed = malloc((size_t)NX * lny * sizeof(double));
    for (int jl = 0; jl < lny; jl++)
        for (int i = 0; i < NX; i++)
            local_packed[idx2(i, jl, NX)] = u_old[idx2(i + 1, jl + 1, pnx)];

    double *full = NULL;
    if (rank == 0) full = malloc((size_t)NX * NY * sizeof(double));
    MPI_Gather(local_packed, NX * lny, MPI_DOUBLE,
               full, NX * lny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (write_field_binary(outpath, full, (size_t)NX * NY) != 0) {
            fprintf(stderr, "write failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("jacobi2d_mpi: %dx%d, %d iters, %d ranks, checksum=%.17g\n",
               NX, NY, niters, nprocs, checksum(full, (size_t)NX * NY));
        free(full);
    }

    double halo_pct = 100.0 * t_halo_total / t_step_total;
    double max_halo_pct;
    MPI_Reduce(&halo_pct, &max_halo_pct, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        fprintf(stderr, "halo_time_pct_of_step(max_over_ranks)=%.2f\n", max_halo_pct);

    free(u_old); free(u_new); free(local_packed);
    MPI_Finalize();
    return 0;
}
