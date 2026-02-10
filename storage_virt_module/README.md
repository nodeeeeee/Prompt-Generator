# Virt-CoW: Robust Embedded Copy-on-Write Module

## Overview
This module implements a high-performance, robust Copy-on-Write (CoW) target for the Linux Device Mapper. It is designed for embedded systems where reliability and performance are critical.

## Robustness Features

### 1. Strict Error Handling
- Every function checks return values from kernel APIs (`dm_io`, `alloc_page`, `mempool_alloc`, etc.).
- Errors are logged using `DMERR` with context-specific information (e.g., chunk number).
- Rollback mechanisms are implemented (e.g., clearing the bitmap bit if metadata persistence fails).

### 2. Input & Boundary Validation
- All pointers passed to functions are validated against `NULL`.
- IO requests are checked against target boundaries in `virt_cow_map`.
- Metadata sector indices are validated before writing to ensure no out-of-bounds access to the CoW device.
- `argc` and device handles are strictly validated in the constructor.

### 3. High Performance
- **Mempools**: Uses `mempool_t` for `cow_io_job` allocations to ensure forward progress under memory pressure and reduce allocation latency.
- **RCU (Read-Copy-Update)**: Uses RCU for metadata reads in the hot mapping path (`virt_cow_map`), allowing lock-free reads.
- **Optimized Persistence**: Metadata is persisted at the sector level (512 bytes) rather than writing the entire bitmap, reducing IO overhead.
- **Branch Prediction**: Uses `likely()` and `unlikely()` macros to optimize the instruction pipeline for the common path.
- **Asynchronous IO**: Heavy data copying and metadata persistence are offloaded to a dedicated workqueue.

### 4. State Management
The CoW process is managed via a state machine within `cow_io_job`:
- `JOB_INITIALIZED`: Job created and queued.
- `JOB_COPYING_DATA`: Reading from origin and writing to CoW device.
- `JOB_UPDATING_METADATA`: Updating the in-memory bitmap.
- `JOB_PERSISTING_METADATA`: Writing the updated bitmap sector to disk.
- `JOB_COMPLETING`: Remapping and resubmitting the original BIO.
- `JOB_ERROR`: Error state for cleanup and reporting.

## Build and Test
### Build
```bash
make
```

### Test Strategy
1. **Pilot Run**: Test with a small loop device to verify basic remapping.
2. **Failure Injection**: Simulate memory pressure to verify mempool efficacy.
3. **Boundary Testing**: Issue IOs at the very beginning and end of the device.
4. **Concurrency Testing**: Multiple threads writing to the same chunk to verify race condition handling.
