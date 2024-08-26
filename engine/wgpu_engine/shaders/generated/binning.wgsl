// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// The binning stage

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This must be kept in sync with `ConfigUniform` in `vello_encoding/src/config.rs`
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,

    target_width: u32,
    target_height: u32,

    // The initial color applied to the pixels in a tile during the fine stage.
    // This is only used in the full pipeline.
    base_color: vec4<f32>,

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

// The DrawMonoid is computed as a prefix sum to aid in decoding
// the variable-length encoding of draw objects.
struct DrawMonoid {
    // The number of paths preceding this draw object.
    path_ix: u32,
    // The number of clip operations preceding this draw object.
    clip_ix: u32,
    // The offset of the encoded draw object in the scene (u32s).
    scene_offset: u32,
    // The offset of the associated info.
    info_offset: u32,
}

// Each draw object has a 32-bit draw tag, which is a bit-packed
// version of the draw monoid.
const DRAWTAG_NOP = 0u;
const DRAWTAG_FILL_COLOR = 0x50u;
const DRAWTAG_FILL_LIN_GRADIENT = 0x114u;
const DRAWTAG_FILL_RAD_GRADIENT = 0x29cu;
const DRAWTAG_FILL_SWEEP_GRADIENT = 0x254u;
const DRAWTAG_FILL_IMAGE = 0x248u;
const DRAWTAG_BEGIN_CLIP = 0x9u;
const DRAWTAG_END_CLIP = 0x21u;

/// The first word of each draw info stream entry contains the flags. This is not a part of the
/// draw object stream but get used after the draw objects have been reduced on the GPU.
/// 0 represents a non-zero fill. 1 represents an even-odd fill.
const DRAW_INFO_FLAGS_FILL_RULE_BIT = 1u;

fn draw_monoid_identity() -> DrawMonoid {
    return DrawMonoid();
}

fn combine_draw_monoid(a: DrawMonoid, b: DrawMonoid) -> DrawMonoid {
    var c: DrawMonoid;
    c.path_ix = a.path_ix + b.path_ix;
    c.clip_ix = a.clip_ix + b.clip_ix;
    c.scene_offset = a.scene_offset + b.scene_offset;
    c.info_offset = a.info_offset + b.info_offset;
    return c;
}

fn map_draw_tag(tag_word: u32) -> DrawMonoid {
    var c: DrawMonoid;
    c.path_ix = u32(tag_word != DRAWTAG_NOP);
    c.clip_ix = tag_word & 1u;
    c.scene_offset = (tag_word >> 2u) & 0x07u;
    c.info_offset = (tag_word >> 6u) & 0x0fu;
    return c;
}

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// The annotated bounding box for a path. It has been transformed,
// but contains a link to the active transform, mostly for gradients.
// Coordinates are integer pixels (for the convenience of atomic update)
// but will probably become fixed-point fractions for rectangles.
//
// TODO: This also carries a `draw_flags` field that contains information that gets propagated to
// the draw info stream. This is currently only used for the fill rule. If the other bits remain
// unused we could possibly pack this into some other field, such as the the MSB of `trans_ix`.
struct PathBbox {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    draw_flags: u32,
    trans_ix: u32,
}

fn bbox_intersect(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4(max(a.xy, b.xy), min(a.zw, b.zw));
}

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
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> draw_monoids: array<DrawMonoid>;

@group(0) @binding(2)
var<storage> path_bbox_buf: array<PathBbox>;

@group(0) @binding(3)
var<storage> clip_bbox_buf: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> intersected_bbox: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(6)
var<storage, read_write> bin_data: array<u32>;

// TODO: put in common place
struct BinHeader {
    element_count: u32,
    chunk_offset: u32,
}

@group(0) @binding(7)
var<storage, read_write> bin_header: array<BinHeader>;

// conversion factors from coordinates to bin
const SX = 0.00390625;
const SY = 0.00390625;
//let SX = 1.0 / f32(N_TILE_X * TILE_WIDTH);
//let SY = 1.0 / f32(N_TILE_Y * TILE_HEIGHT);

const WG_SIZE = 256u;
const N_SLICE = 8u;
//let N_SLICE = WG_SIZE / 32u;
const N_SUBSLICE = 4u;

var<workgroup> sh_bitmaps: array<array<atomic<u32>, N_TILE>, N_SLICE>;
// store count values packed two u16's to a u32
var<workgroup> sh_count: array<array<u32, N_TILE>, N_SUBSLICE>;
var<workgroup> sh_chunk_offset: array<u32, N_TILE>;
var<workgroup> sh_previous_failed: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    for (var i = 0u; i < N_SLICE; i += 1u) {
        atomicStore(&sh_bitmaps[i][local_id.x], 0u);
    }
    if local_id.x == 0u {
        let failed = atomicLoad(&bump.lines) > config.lines_size;
        sh_previous_failed = u32(failed);
    }
    // also functions as barrier to protect zeroing of bitmaps
    let failed = workgroupUniformLoad(&sh_previous_failed);
    if failed != 0u {
        if global_id.x == 0u {
            atomicOr(&bump.failed, STAGE_FLATTEN);
        }
        return;
    }

    // Read inputs and determine coverage of bins
    let element_ix = global_id.x;
    var x0 = 0;
    var y0 = 0;
    var x1 = 0;
    var y1 = 0;
    if element_ix < config.n_drawobj {
        let draw_monoid = draw_monoids[element_ix];
        var clip_bbox = vec4(-1e9, -1e9, 1e9, 1e9);
        if draw_monoid.clip_ix > 0u {
            // TODO: `clip_ix` should always be valid as long as the monoids are correct. Leaving
            // the bounds check in here for correctness but we should assert this condition instead
            // once there is a debug-assertion mechanism.
            clip_bbox = clip_bbox_buf[min(draw_monoid.clip_ix - 1u, config.n_clip - 1u)];
        }
        // For clip elements, clip_box is the bbox of the clip path,
        // intersected with enclosing clips.
        // For other elements, it is the bbox of the enclosing clips.
        // TODO check this is true

        let path_bbox = path_bbox_buf[draw_monoid.path_ix];
        let pb = vec4<f32>(vec4(path_bbox.x0, path_bbox.y0, path_bbox.x1, path_bbox.y1));
        let bbox = bbox_intersect(clip_bbox, pb);

        intersected_bbox[element_ix] = bbox;

        // `bbox_intersect` can result in a zero or negative area intersection if the path bbox lies
        // outside the clip bbox. If that is the case, Don't round up the bottom-right corner of the
        // and leave the coordinates at 0. This way the path will get clipped out and won't get
        // assigned to a bin.
        if bbox.x < bbox.z && bbox.y < bbox.w {
            x0 = i32(floor(bbox.x * SX));
            y0 = i32(floor(bbox.y * SY));
            x1 = i32(ceil(bbox.z * SX));
            y1 = i32(ceil(bbox.w * SY));
        }
    }
    let width_in_bins = i32((config.width_in_tiles + N_TILE_X - 1u) / N_TILE_X);
    let height_in_bins = i32((config.height_in_tiles + N_TILE_Y - 1u) / N_TILE_Y);
    x0 = clamp(x0, 0, width_in_bins);
    y0 = clamp(y0, 0, height_in_bins);
    x1 = clamp(x1, 0, width_in_bins);
    y1 = clamp(y1, 0, height_in_bins);
    if x0 == x1 {
        y1 = y0;
    }
    var x = x0;
    var y = y0;
    let my_slice = local_id.x / 32u;
    let my_mask = 1u << (local_id.x & 31u);
    while y < y1 {
        atomicOr(&sh_bitmaps[my_slice][y * width_in_bins + x], my_mask);
        x += 1;
        if x == x1 {
            x = x0;
            y += 1;
        }
    }

    workgroupBarrier();
    // Allocate output segments
    var element_count = 0u;
    for (var i = 0u; i < N_SUBSLICE; i += 1u) {
        element_count += countOneBits(atomicLoad(&sh_bitmaps[i * 2u][local_id.x]));
        let element_count_lo = element_count;
        element_count += countOneBits(atomicLoad(&sh_bitmaps[i * 2u + 1u][local_id.x]));
        let element_count_hi = element_count;
        let element_count_packed = element_count_lo | (element_count_hi << 16u);
        sh_count[i][local_id.x] = element_count_packed;
    }
    // element_count is the number of draw objects covering this thread's bin
    var chunk_offset = atomicAdd(&bump.binning, element_count);
    if chunk_offset + element_count > config.binning_size {
        chunk_offset = 0u;
        atomicOr(&bump.failed, STAGE_BINNING);
    }    
    sh_chunk_offset[local_id.x] = chunk_offset;
    bin_header[global_id.x].element_count = element_count;
    bin_header[global_id.x].chunk_offset = chunk_offset;
    workgroupBarrier();

    // loop over bbox of bins touched by this draw object
    x = x0;
    y = y0;
    while y < y1 {
        let bin_ix = y * width_in_bins + x;
        let out_mask = atomicLoad(&sh_bitmaps[my_slice][bin_ix]);
        // I think this predicate will always be true...
        if (out_mask & my_mask) != 0u {
            var idx = countOneBits(out_mask & (my_mask - 1u));
            if my_slice > 0u {
                let count_ix = my_slice - 1u;
                let count_packed = sh_count[count_ix / 2u][bin_ix];
                idx += (count_packed >> (16u * (count_ix & 1u))) & 0xffffu;
            }
            let offset = config.bin_data_start + sh_chunk_offset[bin_ix];
            bin_data[offset + idx] = element_ix;
        }
        x += 1;
        if x == x1 {
            x = x0;
            y += 1;
        }
    }
}
