// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Set up dispatch size for path count stage.

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Bitflags for each stage that can fail allocation.
const STAGE_BINNING: u32 = 0x1u;
const STAGE_TILE_ALLOC: u32 = 0x2u;
const STAGE_FLATTEN: u32 = 0x4u;
const STAGE_PATH_COUNT: u32 = 0x8u;
const STAGE_COARSE: u32 = 0x10u;

// This must be kept in sync with the struct in config.rs in the encoding crate.
struct BumpAllocators {
    // Bitmask of stages that have failed allocation.
    failed: atomic<u32>,
    binning: atomic<u32>,
    ptcl: atomic<u32>,
    tile: atomic<u32>,
    seg_counts: atomic<u32>,
    segments: atomic<u32>,
    blend: atomic<u32>,
    lines: atomic<u32>,
}

struct IndirectCount {
    count_x: u32,
    count_y: u32,
    count_z: u32,
}


@group(0) @binding(0)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(1)
var<storage, read_write> indirect: IndirectCount;

// Partition size for path count stage
const WG_SIZE = 256u;

@compute @workgroup_size(1)
fn main() {
    if atomicLoad(&bump.failed) != 0u {
        indirect.count_x = 0u;
    } else {
        let lines = atomicLoad(&bump.lines);
        indirect.count_x = (lines + (WG_SIZE - 1u)) / WG_SIZE;
    }
    indirect.count_y = 1u;
    indirect.count_z = 1u;
}
