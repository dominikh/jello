// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This shader is the second stage of reduction for the pathtag
// monoid scan, needed when the number of tags is large.

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This must be kept in sync with `ConfigUniform` in `vello_encoding/src/config.rs`
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
    blend_size: u32,
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

// The "split" point between using local memory in fine for the blend stack and spilling to the blend_spill buffer.
// A higher value will increase vgpr ("register") pressure in fine, but decrease required dynamic memory allocation.
// If changing, also change in vello_shaders/src/cpu/coarse.rs.
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

struct TagMonoid {
    trans_ix: u32,
    // TODO: I don't think pathseg_ix is used.
    pathseg_ix: u32,
    pathseg_offset: u32,
    style_ix: u32,
    path_ix: u32,
}

const PATH_TAG_SEG_TYPE = 3u;
const PATH_TAG_LINETO = 1u;
const PATH_TAG_QUADTO = 2u;
const PATH_TAG_CUBICTO = 3u;
const PATH_TAG_F32 = 8u;
const PATH_TAG_TRANSFORM = 0x20u;
const PATH_TAG_PATH = 0x10u;
const PATH_TAG_STYLE = 0x40u;
const PATH_TAG_SUBPATH_END = 4u;

// Size of the `Style` data structure in words
const STYLE_SIZE_IN_WORDS: u32 = 2u;

const STYLE_FLAGS_STYLE: u32 = 0x80000000u;
const STYLE_FLAGS_FILL: u32 = 0x40000000u;
const STYLE_MITER_LIMIT_MASK: u32 = 0xFFFFu;

const STYLE_FLAGS_START_CAP_MASK: u32 = 0x0C000000u;
const STYLE_FLAGS_END_CAP_MASK: u32 = 0x03000000u;

const STYLE_FLAGS_CAP_BUTT: u32 = 0u;
const STYLE_FLAGS_CAP_SQUARE: u32 = 0x01000000u;
const STYLE_FLAGS_CAP_ROUND: u32 = 0x02000000u;

const STYLE_FLAGS_JOIN_MASK: u32 = 0x30000000u;
const STYLE_FLAGS_JOIN_BEVEL: u32 = 0u;
const STYLE_FLAGS_JOIN_MITER: u32 = 0x10000000u;
const STYLE_FLAGS_JOIN_ROUND: u32 = 0x20000000u;

// TODO: Declare the remaining STYLE flags here.

fn tag_monoid_identity() -> TagMonoid {
    return TagMonoid();
}

fn combine_tag_monoid(a: TagMonoid, b: TagMonoid) -> TagMonoid {
    var c: TagMonoid;
    c.trans_ix = a.trans_ix + b.trans_ix;
    c.pathseg_ix = a.pathseg_ix + b.pathseg_ix;
    c.pathseg_offset = a.pathseg_offset + b.pathseg_offset;
    c.style_ix = a.style_ix + b.style_ix;
    c.path_ix = a.path_ix + b.path_ix;
    return c;
}

fn reduce_tag(tag_word: u32) -> TagMonoid {
    var c: TagMonoid;
    let point_count = tag_word & 0x3030303u;
    c.pathseg_ix = countOneBits((point_count * 7u) & 0x4040404u);
    c.trans_ix = countOneBits(tag_word & (PATH_TAG_TRANSFORM * 0x1010101u));
    let n_points = point_count + ((tag_word >> 2u) & 0x1010101u);
    var a = n_points + (n_points & (((tag_word >> 3u) & 0x1010101u) * 15u));
    a += a >> 8u;
    a += a >> 16u;
    c.pathseg_offset = a & 0xffu;
    c.path_ix = countOneBits(tag_word & (PATH_TAG_PATH * 0x1010101u));
    c.style_ix = countOneBits(tag_word & (PATH_TAG_STYLE * 0x1010101u)) * STYLE_SIZE_IN_WORDS;
    return c;
}


@group(0) @binding(0)
var<storage> reduced_in: array<TagMonoid>;

@group(0) @binding(1)
var<storage, read_write> reduced: array<TagMonoid>;

const LG_WG_SIZE = 8u;
const WG_SIZE = 256u;

var<workgroup> sh_scratch: array<TagMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let ix = global_id.x;
    var agg = reduced_in[ix];
    sh_scratch[local_id.x] = agg;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_scratch[local_id.x + (1u << i)];
            agg = combine_tag_monoid(agg, other);
        }
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
    }
    if local_id.x == 0u {
        reduced[ix >> LG_WG_SIZE] = agg;
    }
}
