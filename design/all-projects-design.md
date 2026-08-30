# SYSTEMS_PROJECTS.md — build spec for Claude Code

Drop this at the repo root (or rename to `CLAUDE.md`). It is written as instructions to an
agent, not as prose to a reader.

**Scope filter applied.** Only software projects in CSE / systems / ML systems / data
infra / security are included. Everything requiring an FPGA board, a fab shuttle, or an
MCU has been cut — see `## Cut from scope` at the end for the list and the reason, so you
don't re-add them.

---

## 0. Global rules for the agent

Apply these to every project below. They override any default habit.

1. **Validation harness before features.** For each project, the first milestone that
   produces a number is a test/benchmark harness. If the harness does not run, stop and
   fix the harness; do not add functionality on top of an unverified base.
2. **Never invent a measurement.** Every number that ends up in a README, a commit
   message, or a resume line must come from a command in `bench/` that can be re-run.
   If a benchmark was not run, write `TODO: measure` — not a plausible-looking figure.
3. **Repo layout** (same for all projects):
   ```
   src/            implementation
   tests/          correctness tests
   bench/          benchmark scripts + raw output committed under bench/results/
   docs/DESIGN.md  what it does, the hard part, the design decision and its alternative
   docs/RESULTS.md tables/plots generated from bench/results, with the exact command
   README.md       one screen: what, how to run, headline measured numbers
   ```
4. **Commit discipline.** One logical change per commit. The commit message states the
   mechanism, not the intent ("add write-ahead log replay on mount", not "improve fs").
5. **Never run kernel or filesystem code on the host.** All kernel-module and block-device
   work happens inside QEMU/KVM or a disposable VM. Confirm the environment before any
   `insmod`, `mkfs`, or device-mapper command.
6. **Distinctiveness gate.** Each project has a "novel angle" field. If the work so far is
   only the publicly published course lab, it is not done. State the novel angle in one
   sentence in `README.md`; if that sentence cannot be written, the project is incomplete.
7. **Laptop constraints are real.** Where a project normally needs a cluster or a GPU, the
   local-substitute path is specified. Correctness is provable locally; scaling numbers
   are not. Do not report single-node numbers as scaling results.
8. **Toolchain first.** Before writing code, verify every tool in the project's
   `Toolchain` list is installed and runs, and record versions in `docs/DESIGN.md`.

---

## 1. `logfs` — transactional (journaling) filesystem

**Category:** CSE / systems; data-infra adjacent.

**Goal.** A filesystem that groups metadata (optionally data) mutations into transactions
that either fully survive a crash or fully disappear.

**The hard part** is not "a filesystem" — it is **crash consistency**: the on-disk state
after an arbitrary power loss must always be recoverable. That means reasoning about write
ordering, FLUSH/FUA barriers, and fsync semantics. Reference for how Linux does it: `fs/jbd2/`
(`jbd2_journal_start` / `_stop`, `jbd2_journal_get_write_access`,
`jbd2_journal_dirty_metadata`, commit, then `jbd2_log_do_checkpoint`), and ext4's three
mount-time modes: `data=ordered` (metadata journaled, data forced first — default),
`data=writeback` (fastest, weakest), `data=journal` (both, most durable).

**Reading before coding:** OSTEP persistence chapters (free, pages.cs.wisc.edu/~remzi/OSTEP/);
kernel Journalling API doc (docs.kernel.org/filesystems/journalling.html); ext4 journal
on-disk format (kernel.org/doc/html/latest/filesystems/ext4/journal.html). For contrast:
FSCQ (SOSP '15) for a verified FS, McKusick's FreeBSD book for soft updates as the
non-journaling alternative.

**Toolchain.** Kernel 6.6 LTS+; Kbuild for out-of-tree modules (`obj-m += logfs.o`);
QEMU/KVM with a scratch virtual disk; ftrace, kgdb (`qemu -s -S` + gdb), bpftrace for
tracing commits; CrashMonkey + ACE; dm-log-writes; xfstests; fio, filebench.

**Stages — do not skip ahead.**

| Stage | Deliverable | Acceptance criteria |
|---|---|---|
| S0 | QEMU dev VM + build harness | `make` builds a hello-world module inside the VM; kgdb attaches |
| S1 | xv6 log extended to multi-block atomic transactions | Kill QEMU mid-commit at ≥20 injected points; every recovery leaves a consistent tree |
| S2 | FUSE FS in userspace with explicit WAL + selectable journal mode | Mount, survive `kill -9` mid-write, replay log on remount; note in DESIGN.md that FUSE cannot be block-level crash-tested faithfully |
| S3 | In-kernel VFS module: `file_system_type`, `super_operations`, `inode_operations`, `address_space_operations`, jbd2-backed journal | Mounts in VM; passes a chosen xfstests subset; CrashMonkey/ACE run completes |

**Validation (this is half the value).**
- **CrashMonkey + ACE** (github.com/utsaslab/crashmonkey, OSDI '18) — black-box record/replay
  that constructs post-crash disk states and auto-checks consistency; ACE generates bounded
  workloads exhaustively (the "B3" method). Published result for calibration: the tools found
  24 of 26 crash-consistency bugs reported in the preceding 5 years, plus 10 new ones in
  mature Linux filesystems.
- **dm-log-writes** to log and replay block writes at arbitrary points.
- **xfstests** for regression; **fio**/**filebench** for throughput and latency.

**Novel angle — pick exactly one:**
(a) measured durability/throughput trade-off across `ordered` / `writeback` / `journal`;
(b) **io_uring** journal write path vs synchronous, with latency distribution;
(c) your own crash-consistency model checker (enumerate reorderings of the committed block
stream, assert the recovery invariant) that reproduces a known ext4/btrfs-class bug pattern;
(d) fast-commit-style delta journaling (store the minimal delta to recreate affected
metadata) in miniature.

**Budget.** 3–4 weeks part-time to S1; 8–14 weeks for S3.

---

## 2. `halo3d` — 3D stencil solver over MPI with derived datatypes

**Category:** CSE / HPC; direct skill transfer to distributed ML training.

**Goal.** A 7-point Laplacian/Jacobi stencil over a 3D grid, decomposed across ranks in a
Cartesian topology, exchanging halo (ghost) cells each step.

**The hard part** is not the stencil loop. It is (i) describing **non-contiguous halo faces
without manual packing**, using MPI derived datatypes, and (ii) **overlapping communication
with computation** and choosing a blocking scheme that cuts memory traffic.

**Reading:** MPI Standard (mpi-forum.org) for datatypes and neighborhood collectives; PSC
"Advanced MPI" notes; CS:APP (csapp.cs.cmu.edu) for the memory-hierarchy reasoning behind
blocking; "Using MPI" (Gropp/Lusk/Skjellum).

**Toolchain.** OpenMPI **and** MPICH (benchmark both — behaviour differs). Datatypes:
`MPI_Type_vector` (strided), `MPI_Type_create_subarray` (preferred for halos: state start/
size/subsize, let MPI compute strides), `MPI_Type_create_resized` (fix extent),
`MPI_Type_create_hindexed` (irregular). Collectives: `MPI_Neighbor_alltoallw`. Alternative
exchange: MPI-3 RMA (`MPI_Put`/`MPI_Get`, passive target). Profiling: Likwid, PAPI, Intel
Advisor (roofline).

**Local constraint.** A laptop gives correctness and single-node scaling only. Run with
`mpirun --oversubscribe -n 8..16` for correctness. **Do not report laptop numbers as a
scaling study.** For real strong/weak scaling use a multi-node allocation (in India: NSM /
PARAM systems via institutional access; elsewhere NSF ACCESS or a short cloud cluster).
Keep `bench/` parameterized by rank count so the same script runs on both.

**Stages.**

| Stage | Deliverable | Acceptance criteria |
|---|---|---|
| S0 | Serial reference solver | Deterministic output; checksum committed as golden |
| S1 | 2D 5-point Jacobi, 1D decomposition, blocking `MPI_Sendrecv` | Bitwise-identical to serial reference |
| S2 | 3D 7-point, `MPI_Cart_create`, `MPI_Type_create_subarray` halos incl. edges/corners; non-blocking `Isend/Irecv + Waitall` with interior/boundary overlap | Bitwise-identical; halo time measured as % of step time |
| S3 | Single `MPI_Neighbor_alltoallw` exchange + temporal blocking | Bitwise-identical; roofline plot; A/B vs S2 on both MPI implementations |

**Validation.** Bitwise-identical vs serial at every stage, and vs a manual-packing variant.
Strong scaling (fixed problem, more ranks) and weak scaling (fixed work per rank). Overlay a
roofline (arithmetic intensity vs achieved GFLOP/s) to show bandwidth- vs compute-bound.
**Note honestly:** whether derived datatypes beat manual packing is implementation-dependent
and actively contested in the literature — your measured A/B across OpenMPI and MPICH *is*
the result, so do not assume a win.

**Novel angle — pick one:** diamond tiling; cache-oblivious wavefront; multicore wavefront
diamond temporal blocking (Malas et al., arXiv:1410.3060); or GPU-aware MPI exchanging
directly from device buffers.

**Budget.** 1–2 weeks to S1; 6–10 weeks to S3.

---

## 3. `ptracer` + `jsh` — syscall tracer and job-control shell

**Category:** CSE / systems + **security** (this is the cheapest genuine cyber artifact on
the list); the eBPF provenance variant is also data-lineage work.

**Goal.** (1) A strace-like tracer that intercepts and decodes a process's syscalls.
(2) A shell with pipelines and POSIX job control.

**The hard part** is the **ptrace state machine** (entry/exit stops across `fork`/`clone`,
multi-process and multi-threaded targets) and, for the shell, **correct job control** —
process groups, controlling-terminal handoff, signal handling to spec.

**Reading:** `ptrace(2)`; seccomp BPF doc (docs.kernel.org/userspace-api/seccomp_filter.html);
the strace source (github.com/strace) as the reference implementation; CS:APP Shell Lab
handout (public; solutions are instructor-only); OSTEP on processes and signals.

**Toolchain.** `PTRACE_TRACEME` / `ATTACH` / `SEIZE` / `SYSCALL` / `GET_SYSCALL_INFO`;
`PTRACE_SETOPTIONS` with `PTRACE_O_TRACEFORK|CLONE|EXEC|TRACESECCOMP`; `/proc/<pid>/mem`
for argument decoding; seccomp-bpf with `SECCOMP_RET_TRACE` (the tracer gets
`PTRACE_EVENT_SECCOMP` and reads `SECCOMP_RET_DATA` via `PTRACE_GETEVENTMSG`) — this is how
strace's `--seccomp-bpf` mode avoids stopping on every syscall. eBPF path: libbpf CO-RE,
BCC, bpftrace.

**Stages.**

| Stage | Deliverable | Acceptance criteria |
|---|---|---|
| S1 | Single-process tracer (`PTRACE_TRACEME` + `PTRACE_SYSCALL`), number + arg decoding; shell with `fork`/`execvp`, redirection, one pipe | Output matches `strace` on 10 fixed workloads |
| S2 | Follow children; decode args from tracee memory; `PTRACE_GET_SYSCALL_INFO`; shell with N-stage pipelines via `dup2`, background jobs | Matches `strace -f` including fork/exec events |
| S3 | seccomp-bpf filtered fast path; full job control (`setpgid`, `tcsetpgrp`, SIGTTOU/TTIN/TSTP/CHLD, fg/bg/jobs) | Overhead measured vs strace with and without the filter; job-control battery passes |

**Validation.** Diff against `strace`/`ltrace` on identical workloads (same syscalls, same
decoded args, same follow-fork behaviour). For the shell: Ctrl-Z / fg / bg, terminal
ownership after a pipeline, orphaned process groups. Measure tracer overhead in both modes.

**Novel angle — pick one:** build the tracer on **eBPF** instead of ptrace and reconstruct
the full process tree with low overhead, adding **file provenance** (which process read/wrote
what) — an audit/fanotify-style provenance graph; or add a seccomp-bpf policy sandbox that
`jsh` applies to launched jobs.

**Budget.** 4–8 weeks part-time.

---

## 4. Catalog — remaining in-scope projects

Difficulty ★ reachable · ★★ substantial · ★★★ hard. Times are part-time weeks.
Tags: **SYS** systems · **SEC** security · **ML** ML systems · **DS** data infra ·
**PL** compilers/languages · **FM** formal methods. All are pure software and run on a
laptop unless noted.

### OS & kernels
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 1 | `scx-locality` | Custom CPU scheduler in BPF via `sched_ext` (mainline since 6.12; build against github.com/sched-ext/scx). Implement enqueue/dispatch callbacks and beat CFS on a target workload. | SYS | ★★ 4–8w |
| 2 | `nsbox` | Container runtime from scratch: UTS/PID/NET/MNT/user namespaces, cgroups v2, `pivot_root`, seccomp. Hard part: cgroup v2 delegation and veth/bridge setup. | SYS SEC | ★★ 3–6w |
| 3 | `kvmm` | Type-2 VMM over `/dev/kvm` ioctls: guest memory, vCPU run loop, virtio. Hard part: MMIO/PIO exit handling. | SYS SEC | ★★★ 6–10w |
| 5 | `wasm-rt` | WebAssembly runtime: parse, validate, execute; then a baseline JIT. Sandboxing untrusted bytecode is the point. | SYS SEC PL | ★★ 5–9w |

### Compilers & languages
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 6 | `jitlang` | JIT for a toy language: LLVM Kaleidoscope frontend then the ORC JIT chapters. Hard part: SSA construction. | PL | ★★ 4–8w |
| 7 | `gcx` | Garbage collector: mark-sweep → generational/copying → concurrent. Hard part: precise root scanning, write barriers. | SYS PL | ★★ 3–6w |
| 8 | `rederive` | Regex engine via Brzozowski derivatives or Thompson NFA→DFA. Linear-time guarantee = the ReDoS defense; frame as SEC if you add an IDS-style benchmark. | PL SEC | ★★ 3–5w |
| 9 | `dlog` | Datalog engine: semi-naïve evaluation, stratified negation. It is a query engine. | DS PL | ★★ 4–7w |
| 10 | `incr` | Incremental computation / self-adjusting computation (Adapton-style dependency graphs) — i.e. incremental view maintenance. | DS PL | ★★★ 6–10w |

### Databases & storage
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 11 | `lsmdb` | LSM-tree engine: MemTable, SSTables, compaction, bloom filters, WAL. Reference: RocksDB design docs. | DS SYS | ★★ 5–9w |
| 12 | `pagedb` | B+-tree + buffer pool + executor + concurrency control. Base on CMU 15-445 / BusTub (github.com/cmu-db/bustub, MIT-licensed), then extend past the course. Its README asks you not to fork publicly. | DS SYS | ★★ 5–9w |
| 13 | `vecexec` | Vectorized execution engine: columnar batches, SIMD operators, push-based pipelines. | DS | ★★★ 6–10w |
| 14 | `qopt` | Cost-based query optimizer: System-R / Cascades join ordering. Optional ML angle: learned cardinality estimation. | DS ML | ★★★ 8–12w |
| 37 | `colfmt` | Columnar format + Arrow-compatible compute: dictionary/RLE/bit-packing encodings, predicate pushdown. | DS | ★★ 5–8w |

### Distributed systems
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 15 | `raft-kv` | Raft (election, log replication, snapshots) + linearizable KV. Validate with Porcupine (github.com/anishathalye/porcupine) — the same checker used in etcd's robustness tests and by PingCAP for TiDB. Note: Raft tolerates *crash* faults, not Byzantine — this is not a security project. | SYS DS | ★★ 6–10w |
| 16 | `linkv` | Distributed KV tested under Jepsen: partition nemesis, checked with Elle/Porcupine. | SYS DS | ★★★ 8–12w |
| 17 | `mrlite` | MapReduce framework with fault-tolerant scheduling. | DS | ★ 3–5w |
| 18 | `mesi-sim` | Cache-coherence protocol simulator (MSI/MESI/MOESI) + memory-consistency model, verified against litmus tests. Software only. Add a Flush+Reload side-channel experiment to make it SEC. | SYS SEC | ★★ 4–7w |

### Networking
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 21 | `utcp` | Userspace TCP stack: handshake, retransmit, congestion control; test against packet captures. | SYS SEC | ★★★ 8–12w |
| 22 | `xdp-path` | Fast datapath on XDP/eBPF (or DPDK); measure Mpps. XDP is the production DDoS/firewall datapath. | SYS SEC DS | ★★★ 6–10w |
| 23 | `mintls` | TLS 1.3: handshake state machine, key schedule, AEAD; interop-test against OpenSSL. **Learning artifact only — never ship.** | SEC | ★★★ 8–12w |

### ML systems
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 24 | `mlir-pass` | MLIR custom dialect + optimization pass (Toy tutorial Ch.1–7, then your own pass). ML *systems*, not ML research — you write no model. | ML PL | ★★ 5–9w |
| 25 | `tunepass` | TVM/MLIR autotuner or scheduling pass: search over tiling/fusion. | ML | ★★★ 6–10w |
| 26 | `tiled-attn` | Fused, tiled, online-softmax attention kernel in CUDA or Triton; benchmark vs naïve and vs cuBLAS. **Needs a GPU** — use a rented instance or Colab; everything else here is laptop-local. | ML | ★★★ 5–9w |
| 27 | `ktune` | GPU kernel autotuner over tile sizes / launch configs with a roofline stopping criterion. GPU required. | ML | ★★ 5–8w |
| 36 | `libar` | Distributed training communication library: ring/tree all-reduce, overlapped with backprop. This is NCCL's problem. Multi-node or multi-GPU for real numbers. | ML SYS | ★★★ 8–12w |

### Performance
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 28 | `lfqueue` | Lock-free structure (Treiber stack / Michael-Scott queue) with explicit C++11 memory ordering; verify under TSan. Hard part: ABA, acquire/release correctness. | SYS | ★★★ 5–9w |
| 29 | `roofline-<kernel>` | Take one kernel to peak with Likwid/PAPI/Advisor; the method (measure → model → verify) is the deliverable. | SYS ML | ★★ 3–6w |

### Security
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 30 | `covfuzz` | Coverage-guided fuzzer, AFL-style: edge-coverage instrumentation, mutation, crash triage. Optional ML angle: learned seed scheduling. | SEC | ★★ 4–7w |
| 31 | `symex` | Symbolic execution engine over a small bytecode: path exploration + Z3. | SEC FM | ★★★ 8–12w |
| 32 | `jailbpf` | Syscall-policy sandbox: seccomp-bpf + Landlock; measure remaining escape surface. Pairs with `jsh`. | SEC | ★★ 3–5w |

### Formal methods
| # | Name | Project | Tags | Diff / time |
|---|---|---|---|---|
| 33 | `spec-raft` | TLA+/Alloy spec of a real protocol (e.g. your own Raft) checked for safety and liveness. | FM | ★★ 3–6w |
| 34 | `crashcheck` | Crash-consistency model checker: enumerate write reorderings, assert recovery invariants. Pairs with `logfs`. | FM DS | ★★★ 6–10w |
| 35 | `verified-avl` | Coq/Lean-verified balanced tree or small verified parser. | FM | ★★★ 8–12w |

**Highest-leverage picks given an ML-research + data-infra background:** `tiled-attn` (26)
and `libar` (36) convert ML knowledge into a systems artifact; `lsmdb` (11) or `colfmt` (37)
convert data-infra knowledge into a storage artifact; `ptracer`'s eBPF provenance variant is
the cheapest way to acquire a defensible security artifact, since it reads as endpoint
telemetry *and* as data lineage.

---

## 5. How the agent should write results up

For every README headline and every resume line: **mechanism + measured result + what was
hard.** Drop tutorial framing ("followed the MIT lab"), vague adjectives ("robust,
scalable"), and undifferentiated stack lists.

Template shape (the numbers below are **placeholders — replace with real measurements or
leave `TODO: measure`**):

- `logfs` — "Journaling filesystem as an in-kernel VFS module with selectable
  ordered/writeback modes; crash consistency validated with CrashMonkey/ACE and
  dm-log-writes across N generated workloads; measured X% write-throughput delta between
  modes under fio."
- `halo3d` — "3D 7-point stencil with MPI Cartesian decomposition and
  `MPI_Type_create_subarray` halo datatypes; replaced point-to-point with
  `MPI_Neighbor_alltoallw` and added wavefront-diamond temporal blocking; halo time cut
  Nx, weak-scaling efficiency Y% to R ranks; bitwise-identical to serial."
- `ptracer` — "ptrace-based syscall tracer with follow-fork/exec via
  `PTRACE_GET_SYSCALL_INFO` and a seccomp-bpf fast path matching strace's decoded output at
  Z% lower overhead; paired with a POSIX job-control shell (process groups, `tcsetpgrp`,
  SIGCHLD reaping, N-stage pipelines via `dup2`)."
- `raft-kv` — "Raft with snapshots plus a linearizable KV layer; verified with Porcupine
  under injected reordering, drops, and partition nemeses across H-event histories."

---

## Cut from scope

Removed because they need hardware outside a laptop (ECE/EEE territory), not because they
are weak projects. Do not re-add them to this repo.

- **RV32I CPU on an FPGA** (Basys 3 / Arty A7 / iCEBreaker; Vivado or Yosys+nextpnr;
  RISCOF compliance). Digital design, not CSE software.
- **Custom RISC-V accelerator extension** — same reason.
- **Tiny Tapeout / OpenMPW tapeout** — needs a fab shuttle; also in flux (Efabless shut
  down in March 2025; Tiny Tapeout moved to IHP 130nm with chips loaned, not owned).
- **RTOS for a microcontroller** — needs Cortex-M / RISC-V MCU hardware.
- **RISC-V hypervisor H-extension** — architecture-bound; borderline, but excluded.

If you later want the hardware track, the CPU project is the one to revive first: it is the
best-documented and has a fully open toolchain.

## Standing caveats

- Course-lab availability shifts by term. CS:APP self-study handouts are public but
  solutions are instructor-only; CMU autograders are gated. Check the current-year page
  before depending on a specific lab.
- Every published figure quoted above (CrashMonkey's bug counts, Porcupine's users) is from
  the cited paper or repo README, not from your runs. Keep the distinction visible in
  `docs/RESULTS.md`.
- Benchmarks on a laptop measure a laptop. Label them as such.