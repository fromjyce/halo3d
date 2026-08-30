CC      ?= cc
MPICC   ?= mpicc
CFLAGS  := -O2 -Wall -Wextra -std=c11
BUILD   := .build

.PHONY: all test bench clean

all: $(BUILD)/serial2d $(BUILD)/serial3d $(BUILD)/jacobi2d_mpi $(BUILD)/jacobi3d_mpi

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/serial2d: src/serial2d.c src/common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/serial2d.c -lm

$(BUILD)/serial3d: src/serial3d.c src/common.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/serial3d.c -lm

$(BUILD)/jacobi2d_mpi: src/jacobi2d_mpi.c src/common.h | $(BUILD)
	$(MPICC) $(CFLAGS) -o $@ src/jacobi2d_mpi.c -lm

$(BUILD)/jacobi3d_mpi: src/jacobi3d_mpi.c src/common.h | $(BUILD)
	$(MPICC) $(CFLAGS) -o $@ src/jacobi3d_mpi.c -lm

test: all
	tests/run_correctness.sh

bench: all
	bench/run_halo_overhead.sh

clean:
	rm -rf $(BUILD)
