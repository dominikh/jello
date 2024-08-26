// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Tile allocation (and zeroing of tiles)

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
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> draw_bboxes: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(4)
var<storage, read_write> paths: array<Path>;

@group(0) @binding(5)
var<storage, read_write> tiles: array<Tile>;

const WG_SIZE = 256u;

var<workgroup> sh_tile_count: array<u32, WG_SIZE>;
var<workgroup> sh_tile_offset: u32;
var<workgroup> sh_previous_failed: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // Exit early if prior stages failed, as we can't run this stage.
    // We need to check only prior stages, as if this stage has failed in another workgroup, 
    // we still want to know this workgroup's memory requirement.
    if local_id.x == 0u {
        let failed = (atomicLoad(&bump.failed) & (STAGE_BINNING | STAGE_FLATTEN)) != 0u;
        sh_previous_failed = u32(failed);
    }
    let failed = workgroupUniformLoad(&sh_previous_failed);
    if failed != 0u {
        return;
    }    
    // scale factors useful for converting coordinates to tiles
    // TODO: make into constants
    let SX = 1.0 / f32(TILE_WIDTH);
    let SY = 1.0 / f32(TILE_HEIGHT);

    let drawobj_ix = global_id.x;
    var drawtag = DRAWTAG_NOP;
    if drawobj_ix < config.n_drawobj {
        drawtag = scene[config.drawtag_base + drawobj_ix];
    }
    var x0 = 0;
    var y0 = 0;
    var x1 = 0;
    var y1 = 0;
    if drawtag != DRAWTAG_NOP && drawtag != DRAWTAG_END_CLIP {
        let bbox = draw_bboxes[drawobj_ix];

        // Don't round up the bottom-right corner of the bbox if the area is zero and leave the
        // coordinates at 0. This will make `tile_count` zero as the shape is clipped out.
        if bbox.x < bbox.z && bbox.y < bbox.w {
            x0 = i32(floor(bbox.x * SX));
            y0 = i32(floor(bbox.y * SY));
            x1 = i32(ceil(bbox.z * SX));
            y1 = i32(ceil(bbox.w * SY));
        }
    }
    let ux0 = u32(clamp(x0, 0, i32(config.width_in_tiles)));
    let uy0 = u32(clamp(y0, 0, i32(config.height_in_tiles)));
    let ux1 = u32(clamp(x1, 0, i32(config.width_in_tiles)));
    let uy1 = u32(clamp(y1, 0, i32(config.height_in_tiles)));
    let tile_count = (ux1 - ux0) * (uy1 - uy0);
    var total_tile_count = tile_count;
    sh_tile_count[local_id.x] = tile_count;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            total_tile_count += sh_tile_count[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sh_tile_count[local_id.x] = total_tile_count;
    }
    if local_id.x == WG_SIZE - 1u {
        let count = sh_tile_count[WG_SIZE - 1u];
        var offset = atomicAdd(&bump.tile, count);
        if offset + count > config.tiles_size {
            offset = 0u;
            atomicOr(&bump.failed, STAGE_TILE_ALLOC);
        }
        paths[drawobj_ix].tiles = offset;
    }    
    // Using storage barriers is a workaround for what appears to be a miscompilation
    // when a normal workgroup-shared variable is used to broadcast the value.
    storageBarrier();
    let tile_offset = paths[drawobj_ix | (WG_SIZE - 1u)].tiles;
    storageBarrier();
    if drawobj_ix < config.n_drawobj {
        let tile_subix = select(0u, sh_tile_count[local_id.x - 1u], local_id.x > 0u);
        let bbox = vec4(ux0, uy0, ux1, uy1);
        let path = Path(bbox, tile_offset + tile_subix);
        paths[drawobj_ix] = path;
    }

    // zero allocated memory
    // Note: if the number of draw objects is small, utilization will be poor.
    // There are two things that can be done to improve that. One would be a
    // separate (indirect) dispatch. Another would be to have each workgroup
    // process fewer draw objects than the number of threads in the wg.
    let total_count = sh_tile_count[WG_SIZE - 1u];
    for (var i = local_id.x; i < total_count; i += WG_SIZE) {
        // Note: could format output buffer as u32 for even better load balancing.
        tiles[tile_offset + i] = Tile(0, 0u);
    }
}
