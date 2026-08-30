/* S2 -- 3D 7-point Jacobi, full 3D decomposition via MPI_Cart_create,
 * MPI_Type_create_subarray halo faces, non-blocking Irecv/Isend with
 * interior/boundary overlap.
 *
 * Design decisions (see docs/DESIGN.md for the full writeup and the
 * alternative considered for each):
 *
 * 1. Face-only halo, no edges/corners. A 7-point stencil only ever reads a
 *    neighbor that differs from the current cell in exactly one axis, so no
 *    owned cell ever needs a diagonal (edge/corner) ghost value. Six
 *    face exchanges are necessary and sufficient; a 26-neighbor exchange
 *    would move extra bytes for cells nothing reads.
 * 2. MPI_Type_create_subarray per face, built once at setup and reused every
 *    iteration, instead of packing each face into a contiguous send buffer
 *    by hand. The datatype describes strides directly against the padded
 *    local array; MPI does the gather/scatter at the wire.
 * 3. Overlap scheme: post all 6 Irecv + 6 Isend, compute the "core" cells
 *    (>=2 cells from every face, so no dependency on a ghost) while those
 *    are in flight, MPI_Waitall the receives, compute the boundary shell,
 *    then MPI_Waitall the sends before the double-buffer swap. This is a
 *    conservative version of overlap -- see docs/DESIGN.md "not taken" for
 *    what a persistent-request / pipelined-across-iterations version would
 *    add.
 *
 * usage: mpirun -n P jacobi3d_mpi NX NY NZ NITERS out.bin [PX PY PZ]
 * PX*PY*PZ must equal P if given; otherwise MPI_Dims_create picks a
 * balanced factorization. Requires NX%PX==0, NY%PY==0, NZ%PZ==0 (uniform
 * decomposition; see docs/DESIGN.md limitations).
 */
#include <mpi.h>
#include "common.h"

enum { XLO = 0, XHI, YLO, YHI, ZLO, ZHI, NFACES };

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 6 && argc != 9) {
        if (rank == 0)
            fprintf(stderr, "usage: %s NX NY NZ NITERS out.bin [PX PY PZ]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int NX = atoi(argv[1]), NY = atoi(argv[2]), NZ = atoi(argv[3]);
    int niters = atoi(argv[4]);
    const char *outpath = argv[5];

    int dims[3] = {0, 0, 0};
    if (argc == 9) {
        dims[0] = atoi(argv[6]); dims[1] = atoi(argv[7]); dims[2] = atoi(argv[8]);
        if (dims[0] * dims[1] * dims[2] != nprocs) {
            if (rank == 0) fprintf(stderr, "PX*PY*PZ must equal nprocs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        MPI_Dims_create(nprocs, 3, dims);
    }
    if (NX % dims[0] || NY % dims[1] || NZ % dims[2]) {
        if (rank == 0)
            fprintf(stderr, "grid %dx%dx%d not divisible by process grid %dx%dx%d\n",
                    NX, NY, NZ, dims[0], dims[1], dims[2]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    int xlo_r, xhi_r, ylo_r, yhi_r, zlo_r, zhi_r;
    MPI_Cart_shift(cart, 0, 1, &xlo_r, &xhi_r);
    MPI_Cart_shift(cart, 1, 1, &ylo_r, &yhi_r);
    MPI_Cart_shift(cart, 2, 1, &zlo_r, &zhi_r);
    int neighbor[NFACES] = {xlo_r, xhi_r, ylo_r, yhi_r, zlo_r, zhi_r};

    int lnx = NX / dims[0], lny = NY / dims[1], lnz = NZ / dims[2];
    int gx0 = coords[0] * lnx, gy0 = coords[1] * lny, gz0 = coords[2] * lnz;
    int pnx = lnx + 2, pny = lny + 2, pnz = lnz + 2;
    size_t n = (size_t)pnx * pny * pnz;
    double *u_old = calloc(n, sizeof(double));
    double *u_new = calloc(n, sizeof(double));
    if (!u_old || !u_new) { fprintf(stderr, "alloc failed\n"); MPI_Abort(cart, 1); }

    /* --- halo datatypes: one send + one recv subarray per face, built once --- */
    MPI_Datatype send_t[NFACES], recv_t[NFACES];
    int sizes[3] = {pnz, pny, pnx};
    struct { int subsizes[3]; int send_start[3]; int recv_start[3]; } spec[NFACES] = {
        /* XLO */ {{lnz, lny, 1}, {1, 1, 1},       {1, 1, 0}},
        /* XHI */ {{lnz, lny, 1}, {1, 1, lnx},     {1, 1, lnx + 1}},
        /* YLO */ {{lnz, 1, lnx}, {1, 1, 1},       {1, 0, 1}},
        /* YHI */ {{lnz, 1, lnx}, {1, lny, 1},     {1, lny + 1, 1}},
        /* ZLO */ {{1, lny, lnx}, {1, 1, 1},       {0, 1, 1}},
        /* ZHI */ {{1, lny, lnx}, {lnz, 1, 1},     {lnz + 1, 1, 1}},
    };
    for (int f = 0; f < NFACES; f++) {
        MPI_Type_create_subarray(3, sizes, spec[f].subsizes, spec[f].send_start,
                                  MPI_ORDER_C, MPI_DOUBLE, &send_t[f]);
        MPI_Type_create_subarray(3, sizes, spec[f].subsizes, spec[f].recv_start,
                                  MPI_ORDER_C, MPI_DOUBLE, &recv_t[f]);
        MPI_Type_commit(&send_t[f]);
        MPI_Type_commit(&recv_t[f]);
    }

    double dx = 1.0 / (NX - 1), dy = 1.0 / (NY - 1), dz = 1.0 / (NZ - 1);
    for (int lk = 0; lk < lnz; lk++) {
        int gk = gz0 + lk;
        for (int lj = 0; lj < lny; lj++) {
            int gj = gy0 + lj;
            for (int li = 0; li < lnx; li++) {
                int gi = gx0 + li;
                double val = 0.0;
                if (gi == 0 || gi == NX - 1 || gj == 0 || gj == NY - 1 || gk == 0 || gk == NZ - 1)
                    val = boundary_value(gi * dx, gj * dy, gk * dz);
                u_old[idx3(li + 1, lj + 1, lk + 1, pnx, pny)] = val;
                u_new[idx3(li + 1, lj + 1, lk + 1, pnx, pny)] = val;
            }
        }
    }

    int have_core = (lnx >= 3 && lny >= 3 && lnz >= 3);
    double t_recvwait_total = 0.0, t_sendwait_total = 0.0, t_step_total = 0.0;

    for (int it = 0; it < niters; it++) {
        double t_step0 = MPI_Wtime();

        MPI_Request recv_req[NFACES], send_req[NFACES];
        for (int f = 0; f < NFACES; f++)
            MPI_Irecv(u_old, 1, recv_t[f], neighbor[f], f, cart, &recv_req[f]);
        /* tag must match the *opposite* face on the sending side */
        int opp[NFACES] = {XHI, XLO, YHI, YLO, ZHI, ZLO};
        for (int f = 0; f < NFACES; f++)
            MPI_Isend(u_old, 1, send_t[f], neighbor[f], opp[f], cart, &send_req[f]);

        if (have_core) {
            for (int lk = 2; lk <= lnz - 1; lk++) {
                int gk = gz0 + lk - 1;
                for (int lj = 2; lj <= lny - 1; lj++) {
                    int gj = gy0 + lj - 1;
                    for (int li = 2; li <= lnx - 1; li++) {
                        int gi = gx0 + li - 1;
                        if (gi == 0 || gi == NX - 1 || gj == 0 || gj == NY - 1 ||
                            gk == 0 || gk == NZ - 1) {
                            u_new[idx3(li, lj, lk, pnx, pny)] = u_old[idx3(li, lj, lk, pnx, pny)];
                            continue;
                        }
                        double sum =
                            u_old[idx3(li - 1, lj, lk, pnx, pny)] +
                            u_old[idx3(li + 1, lj, lk, pnx, pny)] +
                            u_old[idx3(li, lj - 1, lk, pnx, pny)] +
                            u_old[idx3(li, lj + 1, lk, pnx, pny)] +
                            u_old[idx3(li, lj, lk - 1, pnx, pny)] +
                            u_old[idx3(li, lj, lk + 1, pnx, pny)];
                        u_new[idx3(li, lj, lk, pnx, pny)] = sum / 6.0;
                    }
                }
            }
        }

        double t_rw0 = MPI_Wtime();
        MPI_Waitall(NFACES, recv_req, MPI_STATUSES_IGNORE);
        t_recvwait_total += MPI_Wtime() - t_rw0;

        for (int lk = 1; lk <= lnz; lk++) {
            int gk = gz0 + lk - 1;
            int kcore = have_core && lk >= 2 && lk <= lnz - 1;
            for (int lj = 1; lj <= lny; lj++) {
                int gj = gy0 + lj - 1;
                int jcore = kcore && lj >= 2 && lj <= lny - 1;
                for (int li = 1; li <= lnx; li++) {
                    int gi = gx0 + li - 1;
                    int icore = jcore && li >= 2 && li <= lnx - 1;
                    if (icore) continue; /* already computed above */
                    size_t c = idx3(li, lj, lk, pnx, pny);
                    if (gi == 0 || gi == NX - 1 || gj == 0 || gj == NY - 1 ||
                        gk == 0 || gk == NZ - 1) {
                        u_new[c] = u_old[c];
                        continue;
                    }
                    double sum =
                        u_old[idx3(li - 1, lj, lk, pnx, pny)] +
                        u_old[idx3(li + 1, lj, lk, pnx, pny)] +
                        u_old[idx3(li, lj - 1, lk, pnx, pny)] +
                        u_old[idx3(li, lj + 1, lk, pnx, pny)] +
                        u_old[idx3(li, lj, lk - 1, pnx, pny)] +
                        u_old[idx3(li, lj, lk + 1, pnx, pny)];
                    u_new[c] = sum / 6.0;
                }
            }
        }

        double t_sw0 = MPI_Wtime();
        MPI_Waitall(NFACES, send_req, MPI_STATUSES_IGNORE);
        t_sendwait_total += MPI_Wtime() - t_sw0;

        double *tmp = u_old; u_old = u_new; u_new = tmp;
        t_step_total += MPI_Wtime() - t_step0;
    }

    double *local_packed = malloc((size_t)lnx * lny * lnz * sizeof(double));
    for (int lk = 0; lk < lnz; lk++)
        for (int lj = 0; lj < lny; lj++)
            for (int li = 0; li < lnx; li++)
                local_packed[idx3(li, lj, lk, lnx, lny)] =
                    u_old[idx3(li + 1, lj + 1, lk + 1, pnx, pny)];

    /* Assemble the global field on rank 0. MPI_Gatherv needs one recvtype
     * shared by every source, but each rank's block lands at a different
     * offset in `full` -- a single subarray type can't express that. So
     * this is point-to-point: rank 0 receives each other rank's block
     * straight into its final position in `full` via a per-rank subarray
     * type (no manual unpacking on the receive side either), and copies
     * its own block directly. */
    double *full = (rank == 0) ? malloc((size_t)NX * NY * NZ * sizeof(double)) : NULL;
    if (rank == 0) {
        int gsizes[3] = {NZ, NY, NX};
        for (int p = 0; p < nprocs; p++) {
            int pc[3];
            MPI_Cart_coords(cart, p, 3, pc);
            int p_gx0 = pc[0] * lnx, p_gy0 = pc[1] * lny, p_gz0 = pc[2] * lnz;
            if (p == 0) {
                for (int lk = 0; lk < lnz; lk++)
                    for (int lj = 0; lj < lny; lj++)
                        for (int li = 0; li < lnx; li++)
                            full[idx3(p_gx0 + li, p_gy0 + lj, p_gz0 + lk, NX, NY)] =
                                local_packed[idx3(li, lj, lk, lnx, lny)];
            } else {
                int starts[3] = {p_gz0, p_gy0, p_gx0};
                int subs[3] = {lnz, lny, lnx};
                MPI_Datatype blk;
                MPI_Type_create_subarray(3, gsizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &blk);
                MPI_Type_commit(&blk);
                MPI_Recv(full, 1, blk, p, 0, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&blk);
            }
        }
    } else {
        MPI_Send(local_packed, lnx * lny * lnz, MPI_DOUBLE, 0, 0, cart);
    }

    if (rank == 0) {
        if (write_field_binary(outpath, full, (size_t)NX * NY * NZ) != 0) {
            fprintf(stderr, "write failed\n");
            MPI_Abort(cart, 1);
        }
        printf("jacobi3d_mpi: %dx%dx%d, %d iters, %d ranks (%dx%dx%d), checksum=%.17g\n",
               NX, NY, NZ, niters, nprocs, dims[0], dims[1], dims[2],
               checksum(full, (size_t)NX * NY * NZ));
        free(full);
    }

    double step_ms = 1000.0 * t_step_total / niters;
    double recvwait_pct = 100.0 * t_recvwait_total / t_step_total;
    double sendwait_pct = 100.0 * t_sendwait_total / t_step_total;
    double max_recvwait_pct, max_sendwait_pct, max_step_ms;
    MPI_Reduce(&recvwait_pct, &max_recvwait_pct, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&sendwait_pct, &max_sendwait_pct, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&step_ms, &max_step_ms, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0)
        fprintf(stderr,
                "step_ms(max)=%.4f recvwait_pct_of_step(max)=%.2f sendwait_pct_of_step(max)=%.2f\n",
                max_step_ms, max_recvwait_pct, max_sendwait_pct);

    for (int f = 0; f < NFACES; f++) { MPI_Type_free(&send_t[f]); MPI_Type_free(&recv_t[f]); }
    free(u_old); free(u_new); free(local_packed);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
