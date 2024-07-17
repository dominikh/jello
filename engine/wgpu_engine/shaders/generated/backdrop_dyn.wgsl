// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Prefix sum for dynamically allocated backdrops

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

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This must be kept in sync with the struct in src/encoding/resolve.rs
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,

    target_width: u32,
    target_height: u32,

    // The initial color applied to the pixels in a tile during the fine stage.
    // This is only used in the full pipeline. The format is packed RGBA8 in LSB
    // order.
    base_color: u32,

    n_drawobj: u32,
    n_path: u32,
    n_clip: u32,

    // To reduce the number of bindings, info and bin data are combined
    // into one buffer.
    bin_data_start: u32,

    // offsets within scene buffer (in u32 units)
    pathtag_base: u32,
    pathdata_base: u32,

    drawtag_base: u32,
    drawdata_base: u32,

    transform_base: u32,
    style_base: u32,

    // Sizes of bump allocated buffers (in element size units)
    lines_size: u32,
    binning_size: u32,
    tiles_size: u32,
    seg_counts_size: u32,
    segments_size: u32,
    ptcl_size: u32,
}

// Geometry of tiles and bins

const TILE_WIDTH = 16u;
const TILE_HEIGHT = 16u;
// Number of tiles per bin
const N_TILE_X = 16u;
const N_TILE_Y = 16u;
//let N_TILE = N_TILE_X * N_TILE_Y;
const N_TILE = 256u;

// Not currently supporting non-square tiles
const TILE_SCALE = 0.0625;

const BLEND_STACK_SPLIT = 4u;

// The following are computed in draw_leaf from the generic gradient parameters
// encoded in the scene, and stored in the gradient's info struct, for
// consumption during fine rasterization.

// Radial gradient kinds
const RAD_GRAD_KIND_CIRCULAR = 1u;
const RAD_GRAD_KIND_STRIP = 2u;
const RAD_GRAD_KIND_FOCAL_ON_CIRCLE = 3u;
const RAD_GRAD_KIND_CONE = 4u;

// Radial gradient flags
const RAD_GRAD_SWAPPED = 1u;

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Common datatypes for path and tile intermediate info.

struct Path {
    // bounding box in tiles
    bbox: vec4<u32>,
    // offset (in u32's) to tile rectangle
    tiles: u32,
}

struct Tile {
    backdrop: i32,
    // This is used for the count of the number of segments in the
    // tile up to coarse rasterization, and the index afterwards.
    // In the latter variant, the bits are inverted so that tiling
    // can detect whether the tile was allocated; it's best to
    // consider this an enum packed into a u32.
    segment_count_or_ix: u32,
}


@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(2)
var<storage> paths: array<Path>;

@group(0) @binding(3)
var<storage, read_write> tiles: array<Tile>;

const WG_SIZE = 256u;

var<workgroup> sh_row_width: array<u32, WG_SIZE>;
var<workgroup> sh_row_count: array<u32, WG_SIZE>;
var<workgroup> sh_offset: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // Abort if any of the prior stages failed.
    if local_id.x == 0u {
        sh_row_count[0] = atomicLoad(&bump.failed);
    }
    let failed = workgroupUniformLoad(&sh_row_count[0]);
    if failed != 0u {
        return;
    }
    let drawobj_ix = global_id.x;
    var row_count = 0u;
    if drawobj_ix < config.n_drawobj {
        // TODO: when rectangles, path and draw obj are not the same
        let path = paths[drawobj_ix];
        sh_row_width[local_id.x] = path.bbox.z - path.bbox.x;
        row_count = path.bbox.w - path.bbox.y;
        sh_offset[local_id.x] = path.tiles;
    } else {
        // Explicitly zero the row width, just in case.
        sh_row_width[local_id.x] = 0u;
    }
    sh_row_count[local_id.x] = row_count;

    // Prefix sum of row counts
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            row_count += sh_row_count[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_row_count[local_id.x] = row_count;
    }
    workgroupBarrier();
    let total_rows = sh_row_count[WG_SIZE - 1u];
    for (var row = local_id.x; row < total_rows; row += WG_SIZE) {
        var el_ix = 0u;
        for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
            let probe = el_ix + ((WG_SIZE / 2u) >> i);
            if row >= sh_row_count[probe - 1u] {
                el_ix = probe;
            }
        }
        let width = sh_row_width[el_ix];
        if width > 0u {
            var seq_ix = row - select(0u, sh_row_count[el_ix - 1u], el_ix > 0u);
            var tile_ix = sh_offset[el_ix] + seq_ix * width;
            var sum = tiles[tile_ix].backdrop;
            for (var x = 1u; x < width; x += 1u) {
                tile_ix += 1u;
                sum += tiles[tile_ix].backdrop;
                tiles[tile_ix].backdrop = sum;
            }
        }
    }
}
