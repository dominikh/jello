// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// The coarse rasterization stage.

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
const DRAWTAG_FILL_COLOR = 0x44u;
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

// Layout of per-tile command list
// Initial allocation, in u32's.
const PTCL_INITIAL_ALLOC = 64u;
const PTCL_INCREMENT = 256u;

// Amount of space taken by jump
const PTCL_HEADROOM = 2u;

// Tags for PTCL commands
const CMD_END = 0u;
const CMD_FILL = 1u;
const CMD_STROKE = 2u;
const CMD_SOLID = 3u;
const CMD_COLOR = 5u;
const CMD_LIN_GRAD = 6u;
const CMD_RAD_GRAD = 7u;
const CMD_SWEEP_GRAD = 8u;
const CMD_IMAGE = 9u;
const CMD_BEGIN_CLIP = 10u;
const CMD_END_CLIP = 11u;
const CMD_JUMP = 12u;

// The individual PTCL structs are written here, but read/write is by
// hand in the relevant shaders

struct CmdFill {
    size_and_rule: u32,
    seg_data: u32,
    backdrop: i32,
}

struct CmdStroke {
    tile: u32,
    half_width: f32,
}

struct CmdJump {
    new_ix: u32,
}

struct CmdColor {
    rgba_color: u32,
}

struct CmdLinGrad {
    index: u32,
    extend_mode: u32,
    line_x: f32,
    line_y: f32,
    line_c: f32,
}

struct CmdRadGrad {
    index: u32,
    extend_mode: u32,
    matrx: vec4<f32>,
    xlat: vec2<f32>,
    focal_x: f32,
    radius: f32,
    kind: u32,
    flags: u32,
}

struct CmdSweepGrad {
    index: u32,
    extend_mode: u32,
    matrx: vec4<f32>,
    xlat: vec2<f32>,
    t0: f32,
    t1: f32,
}

struct CmdImage {
    matrx: vec4<f32>,
    xlat: vec2<f32>,
	index: u32,
    extents: vec2<f32>,
}

struct CmdEndClip {
    blend: u32,
    alpha: f32,
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

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Color mixing modes

const MIX_NORMAL = 0u;
const MIX_MULTIPLY = 1u;
const MIX_SCREEN = 2u;
const MIX_OVERLAY = 3u;
const MIX_DARKEN = 4u;
const MIX_LIGHTEN = 5u;
const MIX_COLOR_DODGE = 6u;
const MIX_COLOR_BURN = 7u;
const MIX_HARD_LIGHT = 8u;
const MIX_SOFT_LIGHT = 9u;
const MIX_DIFFERENCE = 10u;
const MIX_EXCLUSION = 11u;
const MIX_HUE = 12u;
const MIX_SATURATION = 13u;
const MIX_COLOR = 14u;
const MIX_LUMINOSITY = 15u;
const MIX_CLIP = 128u;

fn screen(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return cb + cs - (cb * cs);
}

fn color_dodge(cb: f32, cs: f32) -> f32 {
    if cb == 0.0 {
        return 0.0;
    } else if cs == 1.0 {
        return 1.0;
    } else {
        return min(1.0, cb / (1.0 - cs));
    }
}

fn color_burn(cb: f32, cs: f32) -> f32 {
    if cb == 1.0 {
        return 1.0;
    } else if cs == 0.0 {
        return 0.0;
    } else {
        return 1.0 - min(1.0, (1.0 - cb) / cs);
    }
}

fn hard_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    return select(
        screen(cb, 2.0 * cs - 1.0),
        cb * 2.0 * cs,
        cs <= vec3(0.5)
    );
}

fn soft_light(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    let d = select(
        sqrt(cb),
        ((16.0 * cb - 12.0) * cb + 4.0) * cb,
        cb <= vec3(0.25)
    );
    return select(
        cb + (2.0 * cs - 1.0) * (d - cb),
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
        cs <= vec3(0.5)
    );
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}

fn lum(c: vec3<f32>) -> f32 {
    let f = vec3(0.3, 0.59, 0.11);
    return dot(c, f);
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum(c);
    let n = min(c.x, min(c.y, c.z));
    let x = max(c.x, max(c.y, c.z));
    if n < 0.0 {
        c = l + (((c - l) * l) / (l - n));
    }
    if x > 1.0 {
        c = l + (((c - l) * (1.0 - l)) / (x - l));
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_color(c + (l - lum(c)));
}

fn set_sat_inner(
    cmin: ptr<function, f32>,
    cmid: ptr<function, f32>,
    cmax: ptr<function, f32>,
    s: f32
) {
    if *cmax > *cmin {
        *cmid = ((*cmid - *cmin) * s) / (*cmax - *cmin);
        *cmax = s;
    } else {
        *cmid = 0.0;
        *cmax = 0.0;
    }
    *cmin = 0.0;
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    var r = c.r;
    var g = c.g;
    var b = c.b;
    if r <= g {
        if g <= b {
            set_sat_inner(&r, &g, &b, s);
        } else {
            if r <= b {
                set_sat_inner(&r, &b, &g, s);
            } else {
                set_sat_inner(&b, &r, &g, s);
            }
        }
    } else {
        if r <= b {
            set_sat_inner(&g, &r, &b, s);
        } else {
            if g <= b {
                set_sat_inner(&g, &b, &r, s);
            } else {
                set_sat_inner(&b, &g, &r, s);
            }
        }
    }
    return vec3(r, g, b);
}

// Blends two RGB colors together. The colors are assumed to be in sRGB
// color space, and this function does not take alpha into account.
fn blend_mix(cb: vec3<f32>, cs: vec3<f32>, mode: u32) -> vec3<f32> {
    var b = vec3(0.0);
    switch mode {
        case MIX_MULTIPLY: {
            b = cb * cs;
        }
        case MIX_SCREEN: {
            b = screen(cb, cs);
        }
        case MIX_OVERLAY: {
            b = hard_light(cs, cb);
        }
        case MIX_DARKEN: {
            b = min(cb, cs);
        }
        case MIX_LIGHTEN: {
            b = max(cb, cs);
        }
        case MIX_COLOR_DODGE: {
            b = vec3(color_dodge(cb.x, cs.x), color_dodge(cb.y, cs.y), color_dodge(cb.z, cs.z));
        }
        case MIX_COLOR_BURN: {
            b = vec3(color_burn(cb.x, cs.x), color_burn(cb.y, cs.y), color_burn(cb.z, cs.z));
        }
        case MIX_HARD_LIGHT: {
            b = hard_light(cb, cs);
        }
        case MIX_SOFT_LIGHT: {
            b = soft_light(cb, cs);
        }
        case MIX_DIFFERENCE: {
            b = abs(cb - cs);
        }
        case MIX_EXCLUSION: {
            b = cb + cs - 2.0 * cb * cs;
        }
        case MIX_HUE: {
            b = set_lum(set_sat(cs, sat(cb)), lum(cb));
        }
        case MIX_SATURATION: {
            b = set_lum(set_sat(cb, sat(cs)), lum(cb));
        }
        case MIX_COLOR: {
            b = set_lum(cs, lum(cb));
        }
        case MIX_LUMINOSITY: {
            b = set_lum(cb, lum(cs));
        }
        default: {
            b = cs;
        }
    }
    return b;
}

// Composition modes

const COMPOSE_CLEAR = 3u;
const COMPOSE_COPY = 1u;
const COMPOSE_DEST = 2u;
const COMPOSE_SRC_OVER = 0u;
const COMPOSE_DEST_OVER = 4u;
const COMPOSE_SRC_IN = 5u;
const COMPOSE_DEST_IN = 6u;
const COMPOSE_SRC_OUT = 7u;
const COMPOSE_DEST_OUT = 8u;
const COMPOSE_SRC_ATOP = 9u;
const COMPOSE_DEST_ATOP = 10u;
const COMPOSE_XOR = 11u;
const COMPOSE_PLUS = 12u;
const COMPOSE_PLUS_LIGHTER = 13u;

// Apply general compositing operation.
// Inputs are separated colors and alpha, output is premultiplied.
fn blend_compose(
    cb: vec3<f32>,
    cs: vec3<f32>,
    ab: f32,
    as_: f32,
    mode: u32
) -> vec4<f32> {
    var fa = 0.0;
    var fb = 0.0;
    switch mode {
        case COMPOSE_COPY: {
            fa = 1.0;
            fb = 0.0;
        }
        case COMPOSE_DEST: {
            fa = 0.0;
            fb = 1.0;
        }
        case COMPOSE_SRC_OVER: {
            fa = 1.0;
            fb = 1.0 - as_;
        }
        case COMPOSE_DEST_OVER: {
            fa = 1.0 - ab;
            fb = 1.0;
        }
        case COMPOSE_SRC_IN: {
            fa = ab;
            fb = 0.0;
        }
        case COMPOSE_DEST_IN: {
            fa = 0.0;
            fb = as_;
        }
        case COMPOSE_SRC_OUT: {
            fa = 1.0 - ab;
            fb = 0.0;
        }
        case COMPOSE_DEST_OUT: {
            fa = 0.0;
            fb = 1.0 - as_;
        }
        case COMPOSE_SRC_ATOP: {
            fa = ab;
            fb = 1.0 - as_;
        }
        case COMPOSE_DEST_ATOP: {
            fa = 1.0 - ab;
            fb = as_;
        }
        case COMPOSE_XOR: {
            fa = 1.0 - ab;
            fb = 1.0 - as_;
        }
        case COMPOSE_PLUS: {
            fa = 1.0;
            fb = 1.0;
        }
        case COMPOSE_PLUS_LIGHTER: {
            return min(vec4(1.0), vec4(as_ * cs + ab * cb, as_ + ab));
        }
        default: {}
    }
    let as_fa = as_ * fa;
    let ab_fb = ab * fb;
    let co = as_fa * cs + ab_fb * cb;
    // Modes like COMPOSE_PLUS can generate alpha > 1.0, so clamp.
    return vec4(co, min(as_fa + ab_fb, 1.0));
}

// Apply color mixing and composition. Both input and output colors are
// premultiplied RGB.
fn blend_mix_compose(backdrop: vec4<f32>, src: vec4<f32>, mode: u32) -> vec4<f32> {
    let BLEND_DEFAULT = ((MIX_NORMAL << 8u) | COMPOSE_SRC_OVER);
    let EPSILON = 1e-15;
    if (mode & 0x7fffu) == BLEND_DEFAULT {
        // Both normal+src_over blend and clip case
        return backdrop * (1.0 - src.a) + src;
    }
    // Un-premultiply colors for blending. Max with a small epsilon to avoid NaNs.
    let inv_src_a = 1.0 / max(src.a, EPSILON);
    var cs = src.rgb * inv_src_a;
    let inv_backdrop_a = 1.0 / max(backdrop.a, EPSILON);
    let cb = backdrop.rgb * inv_backdrop_a;
    let mix_mode = mode >> 8u;
    let mixed = blend_mix(cb, cs, mix_mode);
    cs = mix(cs, mixed, backdrop.a);
    let compose_mode = mode & 0xffu;
    if compose_mode == COMPOSE_SRC_OVER {
        let co = mix(backdrop.rgb, cs, src.a);
        return vec4(co, src.a + backdrop.a * (1.0 - src.a));
    } else {
        return blend_compose(cb, cs, backdrop.a, src.a, compose_mode);
    }
}


@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> draw_monoids: array<DrawMonoid>;

// TODO: dedup
struct BinHeader {
    element_count: u32,
    chunk_offset: u32,
}

@group(0) @binding(3)
var<storage> bin_headers: array<BinHeader>;

@group(0) @binding(4)
var<storage> info_bin_data: array<u32>;

@group(0) @binding(5)
var<storage> paths: array<Path>;

@group(0) @binding(6)
var<storage, read_write> tiles: array<Tile>;

@group(0) @binding(7)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(8)
var<storage, read_write> ptcl: array<u32>;



// Much of this code assumes WG_SIZE == N_TILE. If these diverge, then
// a fair amount of fixup is needed.
const WG_SIZE = 256u;
//let N_SLICE = WG_SIZE / 32u;
const N_SLICE = 8u;

var<workgroup> sh_bitmaps: array<array<atomic<u32>, N_TILE>, N_SLICE>;
var<workgroup> sh_part_count: array<u32, WG_SIZE>;
var<workgroup> sh_part_offsets: array<u32, WG_SIZE>;
var<workgroup> sh_drawobj_ix: array<u32, WG_SIZE>;
var<workgroup> sh_tile_stride: array<u32, WG_SIZE>;
var<workgroup> sh_tile_width: array<u32, WG_SIZE>;
var<workgroup> sh_tile_x0y0: array<u32, WG_SIZE>;
var<workgroup> sh_tile_count: array<u32, WG_SIZE>;
var<workgroup> sh_tile_base: array<u32, WG_SIZE>;

// helper functions for writing ptcl

var<private> cmd_offset: u32;
var<private> cmd_limit: u32;

// Make sure there is space for a command of given size, plus a jump if needed
fn alloc_cmd(size: u32) {
    if cmd_offset + size >= cmd_limit {
        // We might be able to save a little bit of computation here
        // by setting the initial value of the bump allocator.
        let ptcl_dyn_start = config.width_in_tiles * config.height_in_tiles * PTCL_INITIAL_ALLOC;
        var new_cmd = ptcl_dyn_start + atomicAdd(&bump.ptcl, PTCL_INCREMENT);
        if new_cmd + PTCL_INCREMENT > config.ptcl_size {
            // This sets us up for technical UB, as lots of threads will be writing
            // to the same locations. But I think it's fine, and predicating the
            // writes would probably slow things down.
            new_cmd = 0u;
            atomicOr(&bump.failed, STAGE_COARSE);
        }
        ptcl[cmd_offset] = CMD_JUMP;
        ptcl[cmd_offset + 1u] = new_cmd;
        cmd_offset = new_cmd;
        cmd_limit = cmd_offset + (PTCL_INCREMENT - PTCL_HEADROOM);
    }
}

fn write_path(tile: Tile, tile_ix: u32, draw_flags: u32) {
    // We overload the "segments" field to store both count (written by
    // path_count stage) and segment allocation (used by path_tiling and
    // fine).
    let n_segs = tile.segment_count_or_ix;
    if n_segs != 0u {
        var seg_ix = atomicAdd(&bump.segments, n_segs);
        tiles[tile_ix].segment_count_or_ix = ~seg_ix;
        alloc_cmd(4u);
        ptcl[cmd_offset] = CMD_FILL;
        let even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0u;
        let size_and_rule = (n_segs << 1u) | u32(even_odd);
        let fill = CmdFill(size_and_rule, seg_ix, tile.backdrop);
        ptcl[cmd_offset + 1u] = fill.size_and_rule;
        ptcl[cmd_offset + 2u] = fill.seg_data;
        ptcl[cmd_offset + 3u] = u32(fill.backdrop);
        cmd_offset += 4u;
    } else {
        alloc_cmd(1u);
        ptcl[cmd_offset] = CMD_SOLID;
        cmd_offset += 1u;
    }
}

fn write_color(color: CmdColor) {
    alloc_cmd(2u);
    ptcl[cmd_offset] = CMD_COLOR;
    ptcl[cmd_offset + 1u] = color.rgba_color;
    cmd_offset += 2u;
}

fn write_grad(ty: u32, index: u32, info_offset: u32) {
    alloc_cmd(3u);
    ptcl[cmd_offset] = ty;
    ptcl[cmd_offset + 1u] = index;
    ptcl[cmd_offset + 2u] = info_offset;
    cmd_offset += 3u;
}

fn write_image(info_offset: u32) {
    alloc_cmd(2u);
    ptcl[cmd_offset] = CMD_IMAGE;
    ptcl[cmd_offset + 1u] = info_offset;
    cmd_offset += 2u;
}

fn write_begin_clip() {
    alloc_cmd(1u);
    ptcl[cmd_offset] = CMD_BEGIN_CLIP;
    cmd_offset += 1u;
}

fn write_end_clip(end_clip: CmdEndClip) {
    alloc_cmd(3u);
    ptcl[cmd_offset] = CMD_END_CLIP;
    ptcl[cmd_offset + 1u] = end_clip.blend;
    ptcl[cmd_offset + 2u] = bitcast<u32>(end_clip.alpha);
    cmd_offset += 3u;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Exit early if prior stages failed, as we can't run this stage.
    // We need to check only prior stages, as if this stage has failed in another workgroup, 
    // we still want to know this workgroup's memory requirement.   
    if local_id.x == 0u {
        var failed = atomicLoad(&bump.failed) & (STAGE_BINNING | STAGE_TILE_ALLOC | STAGE_FLATTEN);
        if atomicLoad(&bump.seg_counts) > config.seg_counts_size {
            failed |= STAGE_PATH_COUNT;
        }
        // Reuse sh_part_count to hold failed flag, shmem is tight
        sh_part_count[0] = u32(failed);
    }
    let failed = workgroupUniformLoad(&sh_part_count[0]);
    if failed != 0u {
        if wg_id.x == 0u && local_id.x == 0u {
            // propagate PATH_COUNT failure to path_tiling_setup so it doesn't need to bind config
            atomicOr(&bump.failed, failed);
        }
        return;
    }
    let width_in_bins = (config.width_in_tiles + N_TILE_X - 1u) / N_TILE_X;
    let bin_ix = width_in_bins * wg_id.y + wg_id.x;
    let n_partitions = (config.n_drawobj + N_TILE - 1u) / N_TILE;

    // Coordinates of the top left of this bin, in tiles.
    let bin_tile_x = N_TILE_X * wg_id.x;
    let bin_tile_y = N_TILE_Y * wg_id.y;

    let tile_x = local_id.x % N_TILE_X;
    let tile_y = local_id.x / N_TILE_X;
    let this_tile_ix = (bin_tile_y + tile_y) * config.width_in_tiles + bin_tile_x + tile_x;
    cmd_offset = this_tile_ix * PTCL_INITIAL_ALLOC;
    cmd_limit = cmd_offset + (PTCL_INITIAL_ALLOC - PTCL_HEADROOM);

    // clip state
    var clip_zero_depth = 0u;
    var clip_depth = 0u;

    var partition_ix = 0u;
    var rd_ix = 0u;
    var wr_ix = 0u;
    var part_start_ix = 0u;
    var ready_ix = 0u;

    // blend state
    var render_blend_depth = 0u;
    var max_blend_depth = 0u;

    let blend_offset = cmd_offset;
    cmd_offset += 1u;

    while true {
        for (var i = 0u; i < N_SLICE; i += 1u) {
            atomicStore(&sh_bitmaps[i][local_id.x], 0u);
        }

        while true {
            if ready_ix == wr_ix && partition_ix < n_partitions {
                part_start_ix = ready_ix;
                var count = 0u;
                if partition_ix + local_id.x < n_partitions {
                    let in_ix = (partition_ix + local_id.x) * N_TILE + bin_ix;
                    let bin_header = bin_headers[in_ix];
                    count = bin_header.element_count;
                    sh_part_offsets[local_id.x] = bin_header.chunk_offset;
                }
                // prefix sum the element counts
                for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
                    sh_part_count[local_id.x] = count;
                    workgroupBarrier();
                    if local_id.x >= (1u << i) {
                        count += sh_part_count[local_id.x - (1u << i)];
                    }
                    workgroupBarrier();
                }
                sh_part_count[local_id.x] = part_start_ix + count;
                ready_ix = workgroupUniformLoad(&sh_part_count[WG_SIZE - 1u]);
                partition_ix += WG_SIZE;
            }
            // use binary search to find draw object to read
            var ix = rd_ix + local_id.x;
            if ix >= wr_ix && ix < ready_ix {
                var part_ix = 0u;
                for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
                    let probe = part_ix + ((N_TILE / 2u) >> i);
                    if ix >= sh_part_count[probe - 1u] {
                        part_ix = probe;
                    }
                }
                ix -= select(part_start_ix, sh_part_count[part_ix - 1u], part_ix > 0u);
                let offset = config.bin_data_start + sh_part_offsets[part_ix];
                sh_drawobj_ix[local_id.x] = info_bin_data[offset + ix];
            }
            wr_ix = min(rd_ix + N_TILE, ready_ix);
            if wr_ix - rd_ix >= N_TILE || (wr_ix >= ready_ix && partition_ix >= n_partitions) {
                break;
            }
            workgroupBarrier();
        }
        // At this point, sh_drawobj_ix[0.. wr_ix - rd_ix] contains merged binning results.
        var tag = DRAWTAG_NOP;
        var drawobj_ix: u32;
        if local_id.x + rd_ix < wr_ix {
            drawobj_ix = sh_drawobj_ix[local_id.x];
            tag = scene[config.drawtag_base + drawobj_ix];
        }

        var tile_count = 0u;
        // I think this predicate is the same as the last, maybe they can be combined
        if tag != DRAWTAG_NOP {
            let path_ix = draw_monoids[drawobj_ix].path_ix;
            let path = paths[path_ix];
            let stride = path.bbox.z - path.bbox.x;
            sh_tile_stride[local_id.x] = stride;
            let dx = i32(path.bbox.x) - i32(bin_tile_x);
            let dy = i32(path.bbox.y) - i32(bin_tile_y);
            let x0 = clamp(dx, 0, i32(N_TILE_X));
            let y0 = clamp(dy, 0, i32(N_TILE_Y));
            let x1 = clamp(i32(path.bbox.z) - i32(bin_tile_x), 0, i32(N_TILE_X));
            let y1 = clamp(i32(path.bbox.w) - i32(bin_tile_y), 0, i32(N_TILE_Y));
            sh_tile_width[local_id.x] = u32(x1 - x0);
            sh_tile_x0y0[local_id.x] = u32(x0) | u32(y0 << 16u);
            tile_count = u32(x1 - x0) * u32(y1 - y0);
            // base relative to bin
            let base = path.tiles - u32(dy * i32(stride) + dx);
            sh_tile_base[local_id.x] = base;
            // TODO: there's a write_tile_alloc here in the source, not sure what it's supposed to do
        }

        // Prefix sum of tile counts
        sh_tile_count[local_id.x] = tile_count;
        for (var i = 0u; i < firstTrailingBit(N_TILE); i += 1u) {
            workgroupBarrier();
            if local_id.x >= (1u << i) {
                tile_count += sh_tile_count[local_id.x - (1u << i)];
            }
            workgroupBarrier();
            sh_tile_count[local_id.x] = tile_count;
        }
        workgroupBarrier();
        let total_tile_count = sh_tile_count[N_TILE - 1u];
        // Parallel iteration over all tiles
        for (var ix = local_id.x; ix < total_tile_count; ix += N_TILE) {
            // Binary search to find draw object which contains this tile
            var el_ix = 0u;
            for (var i = 0u; i < firstTrailingBit(N_TILE); i += 1u) {
                let probe = el_ix + ((N_TILE / 2u) >> i);
                if ix >= sh_tile_count[probe - 1u] {
                    el_ix = probe;
                }
            }
            drawobj_ix = sh_drawobj_ix[el_ix];
            tag = scene[config.drawtag_base + drawobj_ix];
            let seq_ix = ix - select(0u, sh_tile_count[el_ix - 1u], el_ix > 0u);
            let width = sh_tile_width[el_ix];
            let x0y0 = sh_tile_x0y0[el_ix];
            let x = (x0y0 & 0xffffu) + seq_ix % width;
            let y = (x0y0 >> 16u) + seq_ix / width;
            let tile_ix = sh_tile_base[el_ix] + sh_tile_stride[el_ix] * y + x;
            let tile = tiles[tile_ix];
            let is_clip = (tag & 1u) != 0u;
            var is_blend = false;
            if is_clip {
                let BLEND_CLIP = (MIX_CLIP << 8u) | COMPOSE_SRC_OVER;
                let scene_offset = draw_monoids[drawobj_ix].scene_offset;
                let dd = config.drawdata_base + scene_offset;
                let blend = scene[dd];
                is_blend = blend != BLEND_CLIP;
            }

            let di = draw_monoids[drawobj_ix].info_offset;
            let draw_flags = info_bin_data[di];
            let even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0u;
            let n_segs = tile.segment_count_or_ix;

            // If this draw object represents an even-odd fill and we know that no line segment
            // crosses this tile and then this draw object should not contribute to the tile if its
            // backdrop (i.e. the winding number of its top-left corner) is even.
            let backdrop_clear = select(tile.backdrop, abs(tile.backdrop) & 1, even_odd) == 0;
            let include_tile = n_segs != 0u || (backdrop_clear == is_clip) || is_blend;
            if include_tile {
                let el_slice = el_ix / 32u;
                let el_mask = 1u << (el_ix & 31u);
                atomicOr(&sh_bitmaps[el_slice][y * N_TILE_X + x], el_mask);
            }
        }
        workgroupBarrier();
        // At this point bit drawobj % 32 is set in sh_bitmaps[drawobj / 32][y * N_TILE_X + x]
        // if drawobj touches tile (x, y).

        // Write per-tile command list for this tile
        var slice_ix = 0u;
        var bitmap = atomicLoad(&sh_bitmaps[0u][local_id.x]);
        while true {
            if bitmap == 0u {
                slice_ix += 1u;
                // potential optimization: make iteration limit dynamic
                if slice_ix == N_SLICE {
                    break;
                }
                bitmap = atomicLoad(&sh_bitmaps[slice_ix][local_id.x]);
                if bitmap == 0u {
                    continue;
                }
            }

            let el_ix = slice_ix * 32u + firstTrailingBit(bitmap);
            drawobj_ix = sh_drawobj_ix[el_ix];
            // clear LSB of bitmap, using bit magic
            bitmap &= bitmap - 1u;
            let drawtag = scene[config.drawtag_base + drawobj_ix];
            let dm = draw_monoids[drawobj_ix];
            let dd = config.drawdata_base + dm.scene_offset;
            let di = dm.info_offset;
            let draw_flags = info_bin_data[di];
            if clip_zero_depth == 0u {
                let tile_ix = sh_tile_base[el_ix] + sh_tile_stride[el_ix] * tile_y + tile_x;
                let tile = tiles[tile_ix];
                switch drawtag {
                    case DRAWTAG_FILL_COLOR: {
                        write_path(tile, tile_ix, draw_flags);
                        let rgba_color = scene[dd];
                        write_color(CmdColor(rgba_color));
                    }
                    case DRAWTAG_FILL_LIN_GRADIENT: {
                        write_path(tile, tile_ix, draw_flags);
                        let index = scene[dd];
                        let info_offset = di + 1u;
                        write_grad(CMD_LIN_GRAD, index, info_offset);
                    }
                    case DRAWTAG_FILL_RAD_GRADIENT: {
                        write_path(tile, tile_ix, draw_flags);
                        let index = scene[dd];
                        let info_offset = di + 1u;
                        write_grad(CMD_RAD_GRAD, index, info_offset);
                    }
                    case DRAWTAG_FILL_SWEEP_GRADIENT: {
                        write_path(tile, tile_ix, draw_flags);
                        let index = scene[dd];
                        let info_offset = di + 1u;
                        write_grad(CMD_SWEEP_GRAD, index, info_offset);
                    }                    
                    case DRAWTAG_FILL_IMAGE: {
                        write_path(tile, tile_ix, draw_flags);
                        write_image(di + 1u);
                    }
                    case DRAWTAG_BEGIN_CLIP: {
                        if tile.segment_count_or_ix == 0u && tile.backdrop == 0 {
                            clip_zero_depth = clip_depth + 1u;
                        } else {
                            write_begin_clip();
                            render_blend_depth += 1u;
                            max_blend_depth = max(max_blend_depth, render_blend_depth);
                        }
                        clip_depth += 1u;
                    }
                    case DRAWTAG_END_CLIP: {
                        clip_depth -= 1u;
                        // A clip shape is always a non-zero fill (draw_flags=0).
                        write_path(tile, tile_ix, /*draw_flags=*/0u);
                        let blend = scene[dd];
                        let alpha = bitcast<f32>(scene[dd + 1u]);
                        write_end_clip(CmdEndClip(blend, alpha));
                        render_blend_depth -= 1u;
                    }
                    default: {}
                }
            } else {
                // In "clip zero" state, suppress all drawing
                switch drawtag {
                    case DRAWTAG_BEGIN_CLIP: {
                        clip_depth += 1u;
                    }
                    case DRAWTAG_END_CLIP: {
                        if clip_depth == clip_zero_depth {
                            clip_zero_depth = 0u;
                        }
                        clip_depth -= 1u;
                    }
                    default: {}
                }
            }
        }

        rd_ix += N_TILE;
        if rd_ix >= ready_ix && partition_ix >= n_partitions {
            break;
        }
        workgroupBarrier();
    }
    if bin_tile_x + tile_x < config.width_in_tiles && bin_tile_y + tile_y < config.height_in_tiles {
        ptcl[cmd_offset] = CMD_END;
        var blend_ix = 0u;
        if max_blend_depth > BLEND_STACK_SPLIT {
            let scratch_size = (max_blend_depth - BLEND_STACK_SPLIT) * TILE_WIDTH * TILE_HEIGHT;
            blend_ix = atomicAdd(&bump.blend, scratch_size);
            if blend_ix + scratch_size > config.blend_size {
                atomicOr(&bump.failed, STAGE_COARSE);
            }
        }
        ptcl[blend_offset] = blend_ix;
    }
}
