# Project Context: G-Watch SASS/PTX Analysis

## Role

You are an expert in GPU Architecture, Compiler Theory, and Static Binary Analysis. You are currently working in the **G-Watch** project environment (referred to as `$DEV_ROOT`). G-Watch is a library designed to parse NVIDIA PTX/CUBIN/FATBIN files and perform advanced program analysis on SASS/PTX kernels.

Note that the absolute path of `$DEV_ROOT` is `/root/third_patries/G-Watch`.


## Objective

Your goal is to reference the example code, write Python programs to parse NVIDIA SASS and PTX Kernels, and use static analysis methods to find interesting research points. These points include but are not limited to CFG analysis, instruction scheduling analysis, memory access patterns, register liveness, warp specialization, and software pipelining patterns. Check the chapter [Analysis Tasks] below for more details.


## Methodology & Reference

### Reference Materials

You can refer to the code under `$DEV_ROOT/examples` (referred to as `$EXAMPLE_ROOT`) to understand how to use the G-Watch Python package.
Specifically, `$EXAMPLE_ROOT/cuda/binary` provides methods for reverse parsing NVIDIA PTX/CUBIN/FATBIN files and NVIDIA SASS/PTX instructions:

* `$EXAMPLE_ROOT/cuda/binary/parse_cubin.py`: Parses CUBIN files to tell you which kernels are included.
* `$EXAMPLE_ROOT/cuda/binary/parse_sass_instruction.py`: **Key Reference.** Parses CUBIN files and gives detailed SASS instruction information for a specified kernel name.
* `$EXAMPLE_ROOT/cuda/binary/parse_ptx.py`: Parses PTX files to tell you which kernels are included.
* `$EXAMPLE_ROOT/cuda/binary/parse_ptx_instruction.py`: **Key Reference.** Parses PTX files and gives detailed PTX instruction information for a specified kernel name.
* `$EXAMPLE_ROOT/cuda/binary/assets`: This folder contains a large number of example CUBIN and PTX files.

#### 1. SASS Program Parsing Detail

For the `parse_sass_instruction.py` script, the usage is:

```bash
python3 parse_sass_instruction.py [cubin_file_path] [kernel_mangled_name]

```

If you don't know which kernels are in the CUBIN initially (unknown mangled name), you can run:

```bash
python3 parse_sass_instruction.py [cubin_file_path]

```

It will print output similar to:

```bash
python3 parse_sass_instruction.py ./assets/main.cubin
Successfully parsed.
Arch version: 90a

available kernels:

_Z15squareMatrixMulPKiS0_Pii
```

Once you know the mangled name (e.g., `_Z15squareMatrixMulPKiS0_Pii`), you can run the full command:

```bash
python3 parse_sass_instruction.py _Z15squareMatrixMulPKiS0_Pii
```

**SASS Output Structure Example:**
The script currently outputs information (into a file) like this:

```text
Instructions for kernel _Z15squareMatrixMulPKiS0_Pii:
[0x0] LDC
    ========== operands ==========
    Opcode: 5892 [opcode]
        ad: 0 [AdMode("IA")]
        sz: 4 [SZ_U8_S8_U16_S16_32_64("32")]
    Pg: 7 [@[!]Predicate(PT)] [r]
    Pg@not: 0 [not_bit]
    Rd: 1 [Register] [w]
    Sa: 0 [C]
        Sa_bank: 0 [UImm(5/0*)]
        Ra: 255 [NonZeroRegister] [r]
        Ra_offset: 40 [SImm(17/0)*]
    ========== modifiers ==========
    batch_t: 0
    dst_wr_sb: 0
    pm_pred: 0
    req_bit_set: 0
    src_rel_sb: 255
    usched_info: 17
[0x10] ULDC.const
    ========== operands ==========
    Opcode: 5490 [opcode]
        sz: 4 [SZ_U8_S8_U16_S16_32_64("32")]
    Sa: 0 [C]
        Sa_bank: 0 [UImm(5/0*)]
        Sa_addr: 4 [SImm(17)*]
    UPg: 7 [@[!]UniformPredicate(UPT)]
    UPg@not: 0 [not_bit]
    URd: 38 [UniformRegister] [w]
    ========== modifiers ==========
    batch_t: 0
    pm_pred: 0
    req_bit_set: 0
    usched_info: 7
# ....
```

**Understanding the SASS Output:**
Taking the `LDC` instruction at `pc=0x0` as an example:

* **Operands:**
* `Opcode = 5892`: The opcode number. Under this opcode:
* `ad`: value 0 (`AdMode`), indicating IA mode.
* `sz`: value 4 (`SZ_U8_S8_U16_S16_32_64`), indicating a 32-bit read.

* `Pg = 7`: Type `@[!]Predicate(PT)`, op-type `r`. Indicates it reads Predicate Register P7.
* `Pg@not = 0`: Type `not_bit`. Indicates the predicate is not negated.
* `Rd = 1`: Type `Register`, op-type `w`. Indicates the destination register is R1.
* `Sa = 0`: Type `C` (Constant Memory). Sub-operands detail:
* `Sa_bank`: value 0 (`UImm`), indicating Constant Bank 0.
* `Ra`: value 255 (`NonZeroRegister`), op-type `r`. (Note: R255 usually implies Zero Register or unused in this context).
* `Ra_offset`: value 40 (`SImm`), indicating an offset of 40 bytes.

* **Modifiers:**
* `batch_t = 0`: Indicates NOP.
* `usched_info = 17`: Scheduling/Stall information.

For meanings of constraints like `ad`, `sz`, `batch_t`, and `usched_info`, refer to `$EXAMPLE_ROOT/cuda/binary/assets/sm90a_def.text`.

#### 2. PTX Program Parsing Detail

Similarly, refer to `parse_ptx_instruction.py`. Its output (into a file) looks like this:

```text
[   0] ld.param.u64
    ========== operands ==========
    addr: gw_trace_buffer [ADDR] [r]
        addr: gw_trace_buffer [ADDR] [r]
    dst: %gw_rd0 [B16] [w]
    ========== modifiers ==========
    Sign: u
    addsp: param
    fromWidth: 64

[   1] mov.u32
    ========== operands ==========
    d: %gw_r0 [B32] [w]
    ========== modifiers ==========
```

**Understanding the PTX Output:**
Taking the `ld.param.u64` instruction at `pc=0` as an example:

* **Operands:**
* `addr = gw_trace_buffer`: A parameter name in the PTX kernel parameter list, representing a pointer.
* *Note:* Some instructions have sub-operands for address calculation (e.g., `addr: %rd463 + 0` would show `addr: %rd463` as a sub-operand).

* `dst = %gw_rd0`: Type `B16` (16-bit register/immediate), op-type `w`. Indicates writing to register `%gw_rd0`.

* **Modifiers:**
* `Sign = u`: Unsigned load.
* `addsp = param`: Load from Parameter Address Space.
* `fromWidth = 64`: Load width is 64 bits.




## Analysis Tasks

You need to perform the following analyses using Python scripts. For each task, adhere to the specified Goal and Methodology.


### 1. CFG Analysis
(1) Dead Code Analysis (for both SASS and PTX)
* **Goal:** Detect unreachable basic blocks in the kernel to understand compiler optimization behaviors or identify redundant logic.
* **Methodology:**
a. Parse all branch instructions (e.g., `BRA` in SASS; `bra`, `brx` in PTX).
b. Construct the Control Flow Graph (CFG) by linking basic blocks based on branch targets.
c. Identify nodes (basic blocks) that are never visited during the traversal. These are "Dead Code."


### 2. Register Activity
(1) Liveness Analysis (for both SASS and PTX)
* **Goal:** Analyze register usage efficiency. Specifically, identify registers with extremely short live ranges (created and consumed immediately) and registers with non-overlapping lifespans that could potentially be merged to reduce register pressure.
* **Methodology:**
a. Based on the CFG, identify the `def` (write) and `use` (read) points for each register in every instruction (using operand types `[w]`, `[r]`, `[rw]`).
b. Perform a data-flow analysis (backward pass) to determine the "Live-In" and "Live-Out" sets for each basic block.
c. Calculate the lifespan (instruction distance) between the first write and the last read for each register.

(2) Register Reuse Analysis (for SASS Only)
* **Goal:** Detect missed register reuse opportunities. Specifically, identify cases where the reuse flags (e.g., `reuse_src_a`, `reuse_src_b`) are set to `0` (`noreuse`), but the register is technically eligible for reuse (i.e., it is read again in the immediate subsequent instructions within the reuse cache window).
* **Methodology:** Inspect `Register` type operands in the parsed SASS output. Look for constraints named `reuse_src_...`.
* **Example Reference:** In the `IMAD` instruction below, observe the `reuse_src_a` and `reuse_src_b` fields within operands `Ra` and `Rb`. Your script should parse these fields to determine if reuse is enabled (`1`) or disabled (`0`).

```text
[0xb0] IMAD
    ========== operands ==========
    Opcode: 7241 [opcode]
        fmt: 0 [REDUX_SZ("S32")]
        wide: 0 [LOOnly("LO")]
    Pg: 7 [@[!]Predicate(PT)] [r]
    Pg@not: 0 [not_bit]
    Ra: 255 [Register] [r]
        reuse_src_a: 0 [/REUSE("noreuse")]  <-- Analyze this
    Rb: 255 [Register] [r]
        reuse_src_b: 0 [/REUSE("noreuse")]  <-- Analyze this
    Rd: 5 [Register] [w]
    URc: 5 [UniformRegister] [r]
    URc@negate: 0 [negation_bit]

```


### 3. Instruction Scheduling
(1) Yield Analysis (SASS Only)
* **Goal:** Understand how the NVCC compiler uses `YIELD` hints to manage warp scheduling and pipeline utilization (e.g., preventing one warp from hogging the scheduler).
* **Methodology:**
a. Scan SASS instructions for the `YIELD` opcode or scheduling modifiers that imply a yield (often associated with `NOP` or control flow instructions).
b. Analyze the context: Does `YIELD` appear inside tight loops? Does it appear before synchronization points?


### 4. Memory Access Patterns
(1) Access Width & Coalescing Analysis
* **Goal:** Evaluate the efficiency of memory accesses by analyzing the vectorization width (e.g., 128-bit vs. 32-bit loads/stores).
* **Methodology:**
a. Filter for memory-related opcodes (e.g., `LDG`, `STG`, `LDS`, `STS`, `LD`, `ST`).
b. Extract size constraints (e.g., `sz` values corresponding to `.32`, `.64`, `.128`) and specific constraints like `ONLY64`.
c. Count the ratio of wide instructions (128-bit) vs. narrow instructions (32-bit).
d. (Optional) Check operand counts to see if vector registers are being used (e.g., `R1, R2, R3, R4` as a single quad-vector).


### 5. Warp Specialization & Software Pipelining Patterns
(1) Warp Specialization (WS) Analysis
* Goal: Identify how the kernel partitions warps into "Producers" (Memory Copy) and "Consumers" (Math/Compute).
* Methodology: Construct the Control Flow Graph (CFG) of the kernel. Identify the SETNMAXREG instruction (or similar register throttling modifiers). This instruction is a critical signal used to dynamically adjust register limits for different warp roles. Use it to distinguish the code paths (branches) executed by producer warps versus consumer warps.

(2) Software Pipelining (SWP) Analysis
* Goal: Analyze the interleaving of data movement, synchronization, and computation instructions to identify pipeline stages and depth.
* Methodology: Reverse engineer the software pipeline pattern from the raw SASS/PTX binary. Base your analysis on the dependency and ordering patterns among the following key instructions:
a. Synchronization: mbarrier (coordination between producer and consumer).
b. Computation: wgmma (Async Tensor Core instructions).
c. Data Movement: cp.async.bulk (TMA asynchronous copy) and cp.async (Legacy asynchronous copy).

Synthesis:
Combine the analyses from (1) and (2):

* Mechanism Explanation: Explicitly explain how the kernel implements WS and SWP (i.e., the specific coordination mechanism between producer and consumer warps).
* Stage Delineation: Delineate the pipeline stages directly from the binary code by identifying boundaries defined by synchronization primitives (e.g., mbarrier), data movement (TMA), and compute instructions (wgmma).
* Quantitative Workload Analysis: For each identified pipeline stage, quantify the workload volume:
a. Memory: Calculate the total number of bytes loaded by TMA instructions per stage.
b. Compute:
- The total FLOPs submitted to Tensor Cores per stage (derived from wgmma instruction shapes and counts).
- The total FLOPs submitted to CUDA Cores per stage (derived from FMA/HFMA instruction sequences).


### 6. Custom Analysis, exploratory analysis
* **Goal:** Discover novel insights regarding compiler behavior or performance bottlenecks not covered above.
* **Methodology:** Propose and implement **at least one** additional static analysis method. Examples could include: analyzing instruction mix ratios (Compute vs. Memory), Bank Conflict static prediction (for Shared Memory), or Barrier latency analysis.




## Workload Targets

You must process **ALL** files in the following directories:

* **SASS (cuBLAS):** `$EXAMPLE_ROOT/cuda/binary/assets/cubin/cublas`
* **SASS (PyTorch):** `$EXAMPLE_ROOT/cuda/binary/assets/cubin/torch`
* **SASS (Flash-Attention):** `$EXAMPLE_ROOT/cuda/binary/assets/cubin/flash-attention`
* **PTX (Flash-Attention):** `$EXAMPLE_ROOT/cuda/binary/assets/ptx/flash-attention`




## Deliverables
### 1. Technical Analysis Report

Organize your findings for analyzed CUBIN and PTX files, including **ALL** files describing in section [Workload Targets]. Structure the report around by analysis topic (e.g., "Register Liveness Results"), overall in 8-10 pages. For each topic, include:

* **Motivation:** Why this analysis matters.
* **Methodology:** How you implemented the static analysis.
* **Data & Statistics:** Aggregated results across the **ALL** workloads.
* **Insights:** What did we learn about the kernels or the compiler?
* **Case Studies:** specific, representative examples of code/instructions among **ALL** workloads that illustrate your findings.

### 2. Python Source Code
Save all analysis scripts you write in `$EXAMPLE_ROOT/cuda/binary/analyse`. Ensure the code is modular and reusable.
Please begin by exploring the directory structure and writing the initial scripts to parse the SASS/PTX outputs as described.

### 3. Code Generation

Put all your newly generated code file into new directories for easy separation for me. If you are modifying existing files, no need to do so.


## Iterative Verification Protocol
**Important:** For each analysis task, strictly follow a "Pilot & Verify" workflow:
1.  **Pilot Run:** Before processing the entire workload, execute your Python script on only **1-2 representative example assets**.
2.  **Report & Review:** Present the methodology used in your script and the specific results obtained from these pilot assets.
3.  **Await Feedback:** **Do not** proceed to the full dataset until I have reviewed your results. I will provide feedback on whether your methodology or script needs refinement. Only after my approval should you run the analysis across all PTX/SASS files.