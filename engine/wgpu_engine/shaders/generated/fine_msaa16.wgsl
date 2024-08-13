// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Fine rasterizer. This can run in simple (just path rendering) and full
// modes, controllable by #define.
//
// To enable multisampled rendering, turn on both the msaa ifdef and one of msaa8
// or msaa16.


struct Tile {
    backdrop: i32,
    segments: u32,
}

// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Segments laid out for contiguous storage
struct Segment {
    // Points are relative to tile origin
    point0: vec2<f32>,
    point1: vec2<f32>,
    y_edge: f32,
}

// A line segment produced by flattening and ready for rasterization.
//
// The name is perhaps too playful, but reflects the fact that these
// lines are completely unordered. They will flow through coarse path
// rasterization, then the per-tile segments will be scatter-written into
// segment storage so that each (tile, path) tuple gets a contiguous
// slice of segments.
struct LineSoup {
    path_ix: u32,
    // Note: this creates an alignment gap. Don't worry about
    // this now, but maybe move to scalars.
    p0: vec2<f32>,
    p1: vec2<f32>,
}

// An intermediate data structure for sorting tile segments.
struct SegmentCount {
    // Reference to element of LineSoup array
    line_ix: u32,
    // Two count values packed into a single u32
    // Lower 16 bits: index of segment within line
    // Upper 16 bits: index of segment within segment slice
    counts: u32,
}

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


@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> segments: array<Segment>;

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


const GRADIENT_WIDTH = 512;

@group(0) @binding(2)
var<storage> ptcl: array<u32>;

@group(0) @binding(3)
var<storage> info: array<u32>;

@group(0) @binding(4)
var<storage, read_write> blend_spill: array<u32>;

@group(0) @binding(5)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(6)
var gradients: texture_2d<f32>;

@group(0) @binding(7)
var images: binding_array<texture_2d<f32>>;

// MSAA-only bindings and utilities

const MASK_LUT_INDEX: u32 = 8;


const MASK_WIDTH = 64u;
const MASK_HEIGHT = 64u;
const SH_SAMPLES_SIZE = 1024u;
const SAMPLE_WORDS_PER_PIXEL = 4u;
@group(0) @binding(MASK_LUT_INDEX)
var<storage> mask_lut: array<u32, 2048u>;

const WG_SIZE = 64u;
var<workgroup> sh_count: array<u32, WG_SIZE>;

// This array contains the winding number of the top left corner of each
// 16 pixel wide row of pixels, relative to the top left corner of the row
// immediately above.
//
// The values are stored packed, as 4 8-bit subwords in a 32 bit word.
// The values are biased signed integers, with 0x80 representing a winding
// number of 0, so that the range of -128 to 127 (inclusive) can be stored
// without carry.
//
// For the even-odd case, the same storage is repurposed, so that a single
// word contains 16 one-bit winding parity values packed to the word.
var<workgroup> sh_winding_y: array<atomic<u32>, 4u>;
// This array contains the winding number of the top left corner of each
// 16 pixel wide row of pixels, relative to the top left corner of the tile.
// It is expanded from sh_winding_y by inclusive prefix sum.
var<workgroup> sh_winding_y_prefix: array<atomic<u32>, 4u>;
// This array contains winding numbers of the top left corner of each
// pixel, relative to the top left corner of the enclosing 16 pixel
// wide row.
//
// During winding number accumulation, it stores a delta (winding number
// relative to the pixel immediately to the left), then expanded using
// prefix sum and reusing the same storage.
//
// The encoding and packing is the same as `sh_winding_y`. For the even-odd
// case, only the first 16 values are used, and each word stores packed
// parity values for one row of pixels.
var<workgroup> sh_winding: array<atomic<u32>, 64u>;
// This array contains winding numbers of multiple sample points within
// a pixel, relative to the winding number of the top left corner of the
// pixel. The encoding and packing is the same as `sh_winding_y`.
var<workgroup> sh_samples: array<atomic<u32>, SH_SAMPLES_SIZE>;

// number of integer cells spanned by interval defined by a, b
fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

const SEG_SIZE = 5u;

// See cpu_shaders/util.rs for explanation of these.
const ONE_MINUS_ULP: f32 = 0.99999994;
const ROBUST_EPSILON: f32 = 2e-7;

// Multisampled path rendering algorithm.
//
// FIXME: This could return an array when https://github.com/gfx-rs/naga/issues/1930 is fixed.
//
// Generally, this algorithm works in an accumulation phase followed by a
// resolving phase, with arrays in workgroup shared memory accumulating
// winding number deltas as the results of edge crossings detected in the
// path segments. Accumulation is in two stages: first a counting stage
// which computes the number of pixels touched by each line segment (with
// each thread processing one line segment), then a stage in which the
// deltas are bumped. Separating these two is a partition-wide prefix sum
// and a binary search to assign the work to threads in a load-balanced
// manner.
//
// The resolving phase is also two stages: prefix sums in both x and y
// directions, then counting nonzero winding numbers for all samples within
// all pixels in the tile.
//
// A great deal of SIMD within a register (SWAR) logic is used, as there
// are a great many winding numbers to be computed. The interested reader
// is invited to study the even-odd case first, as there only one bit is
// needed to represent a winding number parity, thus there is a lot less
// bit shifting, and less shuffling altogether.
fn fill_path_ms(fill: CmdFill, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let even_odd = (fill.size_and_rule & 1u) != 0u;
    // This isn't a divergent branch because the fill parameters are workgroup uniform,
    // provably so because the ptcl buffer is bound read-only.
    if even_odd {
        fill_path_ms_evenodd(fill, local_id, result);
        return;
    }
    let n_segs = fill.size_and_rule >> 1u;
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    // Initialize winding number arrays to a winding number of 0, which is 0x80 in an
    // 8 bit biased signed integer encoding.
    if th_ix < 64u {
        if th_ix < 4u {
            atomicStore(&sh_winding_y[th_ix], 0x80808080u);
        }
        atomicStore(&sh_winding[th_ix], 0x80808080u);
    }
    let sample_count = PIXELS_PER_THREAD * SAMPLE_WORDS_PER_PIXEL;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0x80808080u);
    }
    workgroupBarrier();
    let n_batch = (n_segs + (WG_SIZE - 1u)) / WG_SIZE;
    for (var batch = 0u; batch < n_batch; batch++) {
        let seg_ix = batch * WG_SIZE + th_ix;
        let seg_off = fill.seg_data + seg_ix;
        var count = 0u;
        let slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);
        // TODO: might save a register rewriting this in terms of limit
        if th_ix < slice_size {
            let segment = segments[seg_off];
            let xy0 = segment.point0;
            let xy1 = segment.point1;
            var y_edge_f = f32(TILE_HEIGHT);
            var delta = select(-1, 1, xy1.x <= xy0.x);
            if xy0.x == 0.0 {
                y_edge_f = xy0.y;
            } else if xy1.x == 0.0 {
                y_edge_f = xy1.y;
            }
            // discard horizontal lines aligned to pixel grid
            if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicAdd(&sh_winding_y[y_edge >> 2u], u32(delta) << ((y_edge & 3u) << 3u));
            }
        }
        // workgroup prefix sum of counts
        sh_count[th_ix] = count;
        let lg_n = firstLeadingBit(slice_size * 2u - 1u);
        for (var i = 0u; i < lg_n; i++) {
            workgroupBarrier();
            if th_ix >= 1u << i {
                count += sh_count[th_ix - (1u << i)];
            }
            workgroupBarrier();
            sh_count[th_ix] = count;
        }
        let total = workgroupUniformLoad(&sh_count[slice_size - 1u]);
        for (var i = th_ix; i < total; i += WG_SIZE) {
            // binary search to find pixel
            var lo = 0u;
            var hi = slice_size;
            let goal = i;
            while hi > lo + 1u {
                let mid = (lo + hi) >> 1u;
                if goal >= sh_count[mid - 1u] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let el_ix = lo;
            let last_pixel = i + 1u == sh_count[el_ix];
            let sub_ix = i - select(0u, sh_count[el_ix - 1u], el_ix > 0u);
            let seg_off = fill.seg_data + batch * WG_SIZE + el_ix;
            let segment = segments[seg_off];
            // Coordinates are relative to tile origin
            let xy0_in = segment.point0;
            let xy1_in = segment.point1;
            let is_down = xy1_in.y >= xy0_in.y;
            let xy0 = select(xy1_in, xy0_in, is_down);
            let xy1 = select(xy0_in, xy1_in, is_down);

            // Set up data for line rasterization
            // Note: this is duplicated work if total count exceeds a workgroup.
            // One alternative is to compute it in a separate dispatch.
            let dx = abs(xy1.x - xy0.x);
            let dy = xy1.y - xy0.y;
            let idxdy = 1.0 / (dx + dy);
            var a = dx * idxdy;
            // is_positive_slope is true for \ and | slopes, false for /. For
            // horizontal lines, it follows the original data.
            let is_positive_slope = xy1.x >= xy0.x;
            let x_sign = select(-1.0, 1.0, is_positive_slope);
            let xt0 = floor(xy0.x * x_sign);
            let c = xy0.x * x_sign - xt0;
            let y0i = floor(xy0.y);
            let ytop = y0i + 1.0;
            let b = min((dy * c + dx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
            let count_x = span(xy0.x, xy1.x) - 1u;
            let count = count_x + span(xy0.y, xy1.y);
            let robust_err = floor(a * (f32(count) - 1.0) + b) - f32(count_x);
            if robust_err != 0.0 {
                a -= ROBUST_EPSILON * sign(robust_err);
            }
            let x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));
            // Use line equation to plot pixel coordinates

            let zf = a * f32(sub_ix) + b;
            let z = floor(zf);
            let x = x0i + i32(x_sign * z);
            let y = i32(y0i) + i32(sub_ix) - i32(z);
            // is_delta captures whether the line crosses the top edge of this
            // pixel. If so, then a delta is added to `sh_winding`, followed by
            // a prefix sum, so that a winding number delta is applied to all
            // pixels to the right of this one.
            var is_delta: bool;
            // is_bump captures whether x0 crosses the left edge of this pixel.
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                // The first (top-most) pixel in the line. It is considered to be
                // a line crossing when it touches the top of the pixel.
                //
                // Note: horizontal lines aligned to the pixel grid have already
                // been discarded.
                is_delta = y0i == xy0.y;
                // The pixel is counted as a left edge crossing only at the left
                // edge of the tile (and when it is not the top left corner,
                // using logic analogous to tiling).
                is_bump = xy0.x == 0.0 && y0i != xy0.y;
            } else {
                // Pixels other than the first are a crossing at the top or on
                // the side, based on the conservative line rasterization. When
                // positive slope, the crossing is on the left.
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                let delta_pix = pix_ix + 1u;
                if is_delta {
                    let delta = select(u32(-1i), 1u, is_down) << ((delta_pix & 3u) << 3u);
                    atomicAdd(&sh_winding[delta_pix >> 2u], delta);
                }
            }
            // Apply sample mask
            let mask_block = u32(is_positive_slope) * (MASK_WIDTH * MASK_HEIGHT / 2u);
            let half_height = f32(MASK_HEIGHT / 2u);
            let mask_row = floor(min(a * half_height, half_height - 1.0)) * f32(MASK_WIDTH);
            let mask_col = floor((zf - z) * f32(MASK_WIDTH));
            let mask_ix = mask_block + u32(mask_row + mask_col);
            var mask = mask_lut[mask_ix / 2u] >> ((mask_ix % 2u) * 16u);
            mask &= 0xffffu;
            // Intersect with y half-plane masks
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(16.0 * (xy0.y - f32(y))));
                mask &= 0xffffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(16.0 * (xy1.y - f32(y))));
                mask &= ~(0xffffu << mask_shift);
            }
            // Similar logic as above, only a 16 bit mask is divided into
            // two 8 bit halves first, then each is expanded as above.
            // Mask is 0bABCD_EFGH_IJKL_MNOP. Expand to 4 32 bit words
            // mask0_exp will be 0b0000_000M_0000_000N_0000_000O_0000_000P
            // mask3_exp will be 0b0000_000A_0000_000B_0000_000C_0000_000D
            let mask0 = mask & 0xffu;
            // mask0_a = 0b0IJK_LMNO_*JKL_MNOP
            let mask0_a = mask0 ^ (mask0 << 7u);
            // mask0_b = 0b000I_JKLM_NO*J_KLMN_O*K_LMNO_*JKL_MNOP
            //                ^    ^    ^    ^   ^    ^    ^    ^
            let mask0_b = mask0_a ^ (mask0_a << 14u);
            let mask0_exp = mask0_b & 0x1010101u;
            var mask0_signed = select(mask0_exp, u32(-i32(mask0_exp)), is_down);
            let mask1_exp = (mask0_b >> 4u) & 0x1010101u;
            var mask1_signed = select(mask1_exp, u32(-i32(mask1_exp)), is_down);
            let mask1 = (mask >> 8u) & 0xffu;
            let mask1_a = mask1 ^ (mask1 << 7u);
            // mask1_a = 0b0ABC_DEFG_*BCD_EFGH
            let mask1_b = mask1_a ^ (mask1_a << 14u);
            // mask1_b = 0b000A_BCDE_FG*B_CDEF_G*C_DEFG_*BCD_EFGH
            let mask2_exp = mask1_b & 0x1010101u;
            var mask2_signed = select(mask2_exp, u32(-i32(mask2_exp)), is_down);
            let mask3_exp = (mask1_b >> 4u) & 0x1010101u;
            var mask3_signed = select(mask3_exp, u32(-i32(mask3_exp)), is_down);
            if is_bump {
                let bump_delta = select(u32(-0x1010101i), 0x1010101u, is_down);
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
                mask2_signed += bump_delta;
                mask3_signed += bump_delta;
            }
            atomicAdd(&sh_samples[pix_ix * 4u], mask0_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 1u], mask1_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 2u], mask2_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 3u], mask3_signed);
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    let major = (th_ix * PIXELS_PER_THREAD) >> 2u;
    var packed_w = atomicLoad(&sh_winding[major]);
    // Compute prefix sums of both `sh_winding` and `sh_winding_y`. Both
    // use the same technique. First, a per-word prefix sum is computed
    // of the 4 subwords within each word. The last subword is the sum
    // (reduction) of that group of 4 values, and is stored to shared
    // memory for broadcast to other threads. Then each thread computes
    // the prefix by adding the preceding reduced values.
    //
    // Addition of 2 biased signed values is accomplished by adding the
    // values, then subtracting the bias.
    packed_w += (packed_w - 0x808080u) << 8u;
    packed_w += (packed_w - 0x8080u) << 16u;
    var packed_y = atomicLoad(&sh_winding_y[local_id.y >> 2u]);
    packed_y += (packed_y - 0x808080u) << 8u;
    packed_y += (packed_y - 0x8080u) << 16u;
    var wind_y = (packed_y >> ((local_id.y & 3u) << 3u)) - 0x80u;
    if (local_id.y & 3u) == 3u && local_id.x == 0u {
        let prefix_y = wind_y;
        atomicStore(&sh_winding_y_prefix[local_id.y >> 2u], prefix_y);
    }
    let prefix_x = ((packed_w >> 24u) - 0x80u) * 0x1010101u;
    // reuse sh_winding to store prefix as well
    atomicStore(&sh_winding[major], prefix_x);
    workgroupBarrier();
    for (var i = (major & ~3u); i < major; i++) {
        packed_w += atomicLoad(&sh_winding[i]);
    }
    // packed_w now contains the winding numbers for a slice of 4 pixels,
    // each relative to the top left of the row.
    for (var i = 0u; i < (local_id.y >> 2u); i++) {
        wind_y += atomicLoad(&sh_winding_y_prefix[i]);
    }
    // wind_y now contains the winding number of the top left of the row of
    // pixels relative to the top left of the tile. Note that this is actually
    // a signed quantity stored without bias.

    // The winding number of a sample point is the sum of four levels of
    // hierarchy:
    // * The winding number of the top left of the tile (backdrop)
    // * The winding number of the pixel row relative to tile (wind_y)
    // * The winding number of the pixel relative to row (packed_w)
    // * The winding number of the sample relative to pixel (sh_samples)
    //
    // Conceptually, we want to compute each of these total winding numbers
    // for each sample within a pixel, then count the number that are non-zero.
    // However, we apply a shortcut, partly to make the computation more
    // efficient, and partly to avoid overflow of intermediate results.
    //
    // Here's the technique that's used. The `expected_zero` value contains
    // the *negation* of the sum of the first three levels of the hierarchy.
    // Thus, `sample - expected` is zero when the sum of all levels in the
    // hierarchy is zero, and this is true when `sample = expected`. We
    // compute this using SWAR techniques as follows: we compute the xor of
    // all bits of `expected` (repeated to all subwords) against the packed
    // samples, then the or-reduction of the bits within each subword. This
    // value is 1 when the values are unequal, thus the sum is nonzero, and
    // 0 when the sum is zero. These bits are then masked and counted.

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let minor = i; // assumes PIXELS_PER_THREAD == 4
        let expected_zero = (((packed_w >> (minor * 8u)) + wind_y) & 0xffu) - u32(fill.backdrop);
        // When the expected_zero value exceeds the range of what can be stored
        // in a (biased) signed integer, then there is no sample value that can
        // be equal to the expected value, thus all resulting bits are 1.
        if expected_zero >= 256u {
            area[i] = 1.0;
        } else {
            let samples0 = atomicLoad(&sh_samples[pix_ix * 4u]);
            let samples1 = atomicLoad(&sh_samples[pix_ix * 4u + 1u]);
            let samples2 = atomicLoad(&sh_samples[pix_ix * 4u + 2u]);
            let samples3 = atomicLoad(&sh_samples[pix_ix * 4u + 3u]);
            let xored0 = (expected_zero * 0x1010101u) ^ samples0;
            let xored0_2 = xored0 | (xored0 * 2u);
            let xored1 = (expected_zero * 0x1010101u) ^ samples1;
            let xored1_2 = xored1 | (xored1 >> 1u);
            // xored01 contains 2-reductions from words 0 and 1, interleaved
            let xored01 = (xored0_2 & 0xAAAAAAAAu) | (xored1_2 & 0x55555555u);
            // bits 4 * k + 2 and 4 * k + 3 contain 4-reductions
            let xored01_4 = xored01 | (xored01 * 4u);
            let xored2 = (expected_zero * 0x1010101u) ^ samples2;
            let xored2_2 = xored2 | (xored2 * 2u);
            let xored3 = (expected_zero * 0x1010101u) ^ samples3;
            let xored3_2 = xored3 | (xored3 >> 1u);
            // xored23 contains 2-reductions from words 2 and 3, interleaved
            let xored23 = (xored2_2 & 0xAAAAAAAAu) | (xored3_2 & 0x55555555u);
            // bits 4 * k and 4 * k + 1 contain 4-reductions
            let xored23_4 = xored23 | (xored23 >> 2u);
            // each bit is a 4-reduction, with values from all 4 words
            let xored4 = (xored01_4 & 0xCCCCCCCCu) | (xored23_4 & 0x33333333u);
            // bits 8 * k + {4, 5, 6, 7} contain 8-reductions
            let xored8 = xored4 | (xored4 * 16u);
            area[i] = f32(countOneBits(xored8 & 0xF0F0F0F0u)) * 0.0625;
        }
    }
    *result = area;
}

// Path rendering specialized to the even-odd rule.
//
// This proceeds very much the same as `fill_path_ms`, but is simpler because
// all winding numbers can be represented in one bit. Formally, addition is
// modulo 2, or, equivalently, winding numbers are elements of GF(2). One
// simplification is that we don't need to track the direction of crossings,
// as both have the same effect on winding number.
//
// TODO: factor some logic out to reduce code duplication.
fn fill_path_ms_evenodd(fill: CmdFill, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let n_segs = fill.size_and_rule >> 1u;
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    if th_ix < TILE_HEIGHT {
        if th_ix == 0u {
            atomicStore(&sh_winding_y[th_ix], 0u);
        }
        atomicStore(&sh_winding[th_ix], 0u);
    }
    let sample_count = PIXELS_PER_THREAD;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0u);
    }
    workgroupBarrier();
    let n_batch = (n_segs + (WG_SIZE - 1u)) / WG_SIZE;
    for (var batch = 0u; batch < n_batch; batch++) {
        let seg_ix = batch * WG_SIZE + th_ix;
        let seg_off = fill.seg_data + seg_ix;
        var count = 0u;
        let slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);
        // TODO: might save a register rewriting this in terms of limit
        if th_ix < slice_size {
            let segment = segments[seg_off];
            // Coordinates are relative to tile origin
            let xy0 = segment.point0;
            let xy1 = segment.point1;
            var y_edge_f = f32(TILE_HEIGHT);
            if xy0.x == 0.0 {
                y_edge_f = xy0.y;
            } else if xy1.x == 0.0 {
                y_edge_f = xy1.y;
            }
            // discard horizontal lines aligned to pixel grid
            if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicXor(&sh_winding_y[0], 1u << y_edge);
            }
        }
        // workgroup prefix sum of counts
        sh_count[th_ix] = count;
        let lg_n = firstLeadingBit(slice_size * 2u - 1u);
        for (var i = 0u; i < lg_n; i++) {
            workgroupBarrier();
            if th_ix >= 1u << i {
                count += sh_count[th_ix - (1u << i)];
            }
            workgroupBarrier();
            sh_count[th_ix] = count;
        }
        let total = workgroupUniformLoad(&sh_count[slice_size - 1u]);
        for (var i = th_ix; i < total; i += WG_SIZE) {
            // binary search to find pixel
            var lo = 0u;
            var hi = slice_size;
            let goal = i;
            while hi > lo + 1u {
                let mid = (lo + hi) >> 1u;
                if goal >= sh_count[mid - 1u] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let el_ix = lo;
            let last_pixel = i + 1u == sh_count[el_ix];
            let sub_ix = i - select(0u, sh_count[el_ix - 1u], el_ix > 0u);
            let seg_off = fill.seg_data + batch * WG_SIZE + el_ix;
            let segment = segments[seg_off];
            let xy0_in = segment.point0;
            let xy1_in = segment.point1;
            let is_down = xy1_in.y >= xy0_in.y;
            let xy0 = select(xy1_in, xy0_in, is_down);
            let xy1 = select(xy0_in, xy1_in, is_down);

            // Set up data for line rasterization
            // Note: this is duplicated work if total count exceeds a workgroup.
            // One alternative is to compute it in a separate dispatch.
            let dx = abs(xy1.x - xy0.x);
            let dy = xy1.y - xy0.y;
            let idxdy = 1.0 / (dx + dy);
            var a = dx * idxdy;
            let is_positive_slope = xy1.x >= xy0.x;
            let x_sign = select(-1.0, 1.0, is_positive_slope);
            let xt0 = floor(xy0.x * x_sign);
            let c = xy0.x * x_sign - xt0;
            let y0i = floor(xy0.y);
            let ytop = y0i + 1.0;
            let b = min((dy * c + dx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
            let count_x = span(xy0.x, xy1.x) - 1u;
            let count = count_x + span(xy0.y, xy1.y);
            let robust_err = floor(a * (f32(count) - 1.0) + b) - f32(count_x);
            if robust_err != 0.0 {
                a -= ROBUST_EPSILON * sign(robust_err);
            }
            let x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));
            // Use line equation to plot pixel coordinates

            let zf = a * f32(sub_ix) + b;
            let z = floor(zf);
            let x = x0i + i32(x_sign * z);
            let y = i32(y0i) + i32(sub_ix) - i32(z);
            var is_delta: bool;
            // See comments in nonzero case.
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                is_delta = y0i == xy0.y;
                is_bump = xy0.x == 0.0;
            } else {
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                if is_delta {
                    atomicXor(&sh_winding[y], 2u << u32(x));
                }
            }
            // Apply sample mask
            let mask_block = u32(is_positive_slope) * (MASK_WIDTH * MASK_HEIGHT / 2u);
            let half_height = f32(MASK_HEIGHT / 2u);
            let mask_row = floor(min(a * half_height, half_height - 1.0)) * f32(MASK_WIDTH);
            let mask_col = floor((zf - z) * f32(MASK_WIDTH));
            let mask_ix = mask_block + u32(mask_row + mask_col);
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
            var mask = mask_lut[mask_ix / 2u] >> ((mask_ix % 2u) * 16u);
            mask &= 0xffffu;
            // Intersect with y half-plane masks
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(16.0 * (xy0.y - f32(y))));
                mask &= 0xffffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(16.0 * (xy1.y - f32(y))));
                mask &= ~(0xffffu << mask_shift);
            }
            if is_bump {
                mask ^= 0xffffu;
            }
            atomicXor(&sh_samples[pix_ix], mask);
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    var scan_x = atomicLoad(&sh_winding[local_id.y]);
    // prefix sum over GF(2) is equivalent to carry-less multiplication
    // by 0xFFFF
    scan_x ^= scan_x << 1u;
    scan_x ^= scan_x << 2u;
    scan_x ^= scan_x << 4u;
    scan_x ^= scan_x << 8u;
    // scan_x contains the winding number parity for all pixels in the row
    var scan_y = atomicLoad(&sh_winding_y[0]);
    scan_y ^= scan_y << 1u;
    scan_y ^= scan_y << 2u;
    scan_y ^= scan_y << 4u;
    scan_y ^= scan_y << 8u;
    // winding number parity for the row of pixels is in the LSB of row_parity
    let row_parity = (scan_y >> local_id.y) ^ u32(fill.backdrop);

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let samples = atomicLoad(&sh_samples[pix_ix]);
        let pix_parity = row_parity ^ (scan_x >> (pix_ix % TILE_WIDTH));
        // The LSB of pix_parity contains the sum of the first three levels
        // of the hierarchy, thus the absolute winding number of the top left
        // of the pixel.
        let pix_mask = u32(-i32(pix_parity & 1u));
        // pix_mask is pix_party broadcast to all bits in the word.
        area[i] = f32(countOneBits((samples ^ pix_mask) & 0xffffu)) * 0.0625;
    }
    *result = area;
}

fn read_fill(cmd_ix: u32) -> CmdFill {
    let size_and_rule = ptcl[cmd_ix + 1u];
    let seg_data = ptcl[cmd_ix + 2u];
    let backdrop = i32(ptcl[cmd_ix + 3u]);
    return CmdFill(size_and_rule, seg_data, backdrop);
}

fn read_color(cmd_ix: u32) -> CmdColor {
    let rgba_color = ptcl[cmd_ix + 1u];
    return CmdColor(rgba_color);
}

fn read_lin_grad(cmd_ix: u32) -> CmdLinGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let line_x = bitcast<f32>(info[info_offset]);
    let line_y = bitcast<f32>(info[info_offset + 1u]);
    let line_c = bitcast<f32>(info[info_offset + 2u]);
    return CmdLinGrad(index, extend_mode, line_x, line_y, line_c);
}

fn read_rad_grad(cmd_ix: u32) -> CmdRadGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let focal_x = bitcast<f32>(info[info_offset + 6u]);
    let radius = bitcast<f32>(info[info_offset + 7u]);
    let flags_kind = info[info_offset + 8u];
    let flags = flags_kind >> 3u;
    let kind = flags_kind & 0x7u;
    return CmdRadGrad(index, extend_mode, matrx, xlat, focal_x, radius, kind, flags);
}

fn read_sweep_grad(cmd_ix: u32) -> CmdSweepGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let t0 = bitcast<f32>(info[info_offset + 6u]);
    let t1 = bitcast<f32>(info[info_offset + 7u]);
    return CmdSweepGrad(index, extend_mode, matrx, xlat, t0, t1);
}

fn read_image(cmd_ix: u32) -> CmdImage {
    let info_offset = ptcl[cmd_ix + 1u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));

    let index = info[info_offset + 6u];
    let width_height = info[info_offset + 7u];
    let width = f32(width_height >> 16u);
    let height = f32(width_height & 0xffffu);
    return CmdImage(matrx, xlat, index, vec2(width, height));
}

fn read_end_clip(cmd_ix: u32) -> CmdEndClip {
    let blend = ptcl[cmd_ix + 1u];
    let alpha = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdEndClip(blend, alpha);
}

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
fn extend_mode(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, 1.0);
        }
        case EXTEND_REPEAT: {
            return fract(t);
        }
        case EXTEND_REFLECT, default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

const PIXELS_PER_THREAD = 4u;


// The X size should be 16 / PIXELS_PER_THREAD
@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    if ptcl[0] == ~0u {
        // An earlier stage has failed, don't try to render.
        // We use ptcl[0] for this so we don't use up a binding for bump.
        return;
    }
    let tile_ix = wg_id.y * config.width_in_tiles + wg_id.x;
    let xy = vec2(f32(global_id.x * PIXELS_PER_THREAD), f32(global_id.y));
    let local_xy = vec2(f32(local_id.x * PIXELS_PER_THREAD), f32(local_id.y));
    var rgba: array<vec4<f32>, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        // Note: this differs from Vello, which uses .wzyx.
        // We changed it so that base color, ramps, and commands all
        // use the same format.
        rgba[i] = unpack4x8unorm(config.base_color);
    }
    var blend_stack: array<array<u32, PIXELS_PER_THREAD>, BLEND_STACK_SPLIT>;
    var clip_depth = 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    var cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    let blend_offset = ptcl[cmd_ix];
    cmd_ix += 1u;
    // main interpretation loop
    while true {
        let tag = ptcl[cmd_ix];
        if tag == CMD_END {
            break;
        }
        switch tag {
            case CMD_FILL: {
                let fill = read_fill(cmd_ix);
                fill_path_ms(fill, local_id.xy, &area);
                cmd_ix += 4u;
            }
            case CMD_SOLID: {
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    area[i] = 1.0;
                }
                cmd_ix += 1u;
            }
            case CMD_COLOR: {
                let color = read_color(cmd_ix);
                // Note: this differs from Vello, which uses .wzyx.
                // We changed it so that base color, ramps, and commands all
                // use the same format.
                let fg = unpack4x8unorm(color.rgba_color);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let fg_i = fg * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 2u;
            }
            case CMD_BEGIN_CLIP: {
                if clip_depth < BLEND_STACK_SPLIT {
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                } else {
                    let blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                    let local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                    let local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_spill[local_blend_start + i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                }
                clip_depth += 1u;
                cmd_ix += 1u;
            }
            case CMD_END_CLIP: {
                let end_clip = read_end_clip(cmd_ix);
                clip_depth -= 1u;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    var bg_rgba: u32;
                    if clip_depth < BLEND_STACK_SPLIT {
                        bg_rgba = blend_stack[clip_depth][i];
                    } else {
                        let blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                        let local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                        let local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                        bg_rgba = blend_spill[local_blend_start + i];
                    }
                    let bg = unpack4x8unorm(bg_rgba);
                    let fg = rgba[i] * area[i] * end_clip.alpha;
                    rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                }
                cmd_ix += 3u;
            }
            case CMD_JUMP: {
                cmd_ix = ptcl[cmd_ix + 1u];
            }
            case CMD_LIN_GRAD: {
                let lin = read_lin_grad(cmd_ix);
                let d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_d = d + lin.line_x * f32(i);
                    let x = i32(round(extend_mode(my_d, lin.extend_mode) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(lin.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            case CMD_RAD_GRAD: {
                let rad = read_rad_grad(cmd_ix);
                let focal_x = rad.focal_x;
                let radius = rad.radius;
                let is_strip = rad.kind == RAD_GRAD_KIND_STRIP;
                let is_circular = rad.kind == RAD_GRAD_KIND_CIRCULAR;
                let is_focal_on_circle = rad.kind == RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                let is_swapped = (rad.flags & RAD_GRAD_SWAPPED) != 0u;
                let r1_recip = select(1.0 / radius, 0.0, is_circular);
                let less_scale = select(1.0, -1.0, is_swapped || (1.0 - focal_x) < 0.0);
                let t_sign = sign(1.0 - focal_x);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = rad.matrx.xy * my_xy.x + rad.matrx.zw * my_xy.y + rad.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    let xx = x * x;
                    let yy = y * y;
                    var t = 0.0;
                    var is_valid = true;
                    if is_strip {
                        let a = radius - yy;
                        t = sqrt(a) + x;
                        is_valid = a >= 0.0;
                    } else if is_focal_on_circle {
                        t = (xx + yy) / x;
                        is_valid = t >= 0.0 && x != 0.0;
                    } else if radius > 1.0 {
                        t = sqrt(xx + yy) - x * r1_recip;
                    } else { // radius < 1.0
                        let a = xx - yy;
                        t = less_scale * sqrt(a) - x * r1_recip;
                        is_valid = a >= 0.0 && t >= 0.0;
                    }
                    if is_valid {
                        t = extend_mode(focal_x + t_sign * t, rad.extend_mode);
                        t = select(t, 1.0 - t, is_swapped);
                        let x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                        let fg_rgba = textureLoad(gradients, vec2(x, i32(rad.index)), 0);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 3u;
            }
            case CMD_SWEEP_GRAD: {
                let sweep = read_sweep_grad(cmd_ix);
                let scale = 1.0 / (sweep.t1 - sweep.t0);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = sweep.matrx.xy * my_xy.x + sweep.matrx.zw * my_xy.y + sweep.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    // xy_to_unit_angle from Skia:
                    // See <https://github.com/google/skia/blob/30bba741989865c157c7a997a0caebe94921276b/src/opts/SkRasterPipeline_opts.h#L5859>
                    let xabs = abs(x);
                    let yabs = abs(y);
                    let slope = min(xabs, yabs) / max(xabs, yabs);
                    let s = slope * slope;
                    // again, from Skia:
                    // Use a 7th degree polynomial to approximate atan.
                    // This was generated using sollya.gforge.inria.fr.
                    // A float optimized polynomial was generated using the following command.
                    // P1 = fpminimax((1/(2*Pi))*atan(x),[|1,3,5,7|],[|24...|],[2^(-40),1],relative);
                    var phi = slope * (0.15912117063999176025390625f + s * (-5.185396969318389892578125e-2f + s * (2.476101927459239959716796875e-2f + s * (-7.0547382347285747528076171875e-3f))));
                    phi = select(phi, 1.0 / 4.0 - phi, xabs < yabs);
                    phi = select(phi, 1.0 / 2.0 - phi, x < 0.0);
                    phi = select(phi, 1.0 - phi, y < 0.0);
                    phi = select(phi, 0.0, phi != phi); // check for NaN
                    phi = (phi - sweep.t0) * scale;
                    let t = extend_mode(phi, sweep.extend_mode);
                    let ramp_x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(ramp_x, i32(sweep.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            case CMD_IMAGE: {
                let image = read_image(cmd_ix);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
                    // This currently clips to the image bounds. TODO: extend modes
                    if all(uv < image.extents) && area[i] != 0.0 {
                       let uv_quad = vec4(floor(uv), ceil(uv));
                       let uv_frac = fract(uv);
                       let a = premul_alpha(textureLoad(images[image.index], vec2<i32>(uv_quad.xy), 0));
                       let b = premul_alpha(textureLoad(images[image.index], vec2<i32>(uv_quad.xw), 0));
                       let c = premul_alpha(textureLoad(images[image.index], vec2<i32>(uv_quad.zy), 0));
                       let d = premul_alpha(textureLoad(images[image.index], vec2<i32>(uv_quad.zw), 0));
                       let fg_rgba = mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
                       let fg_i = fg_rgba * area[i];
                       rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 2u;
            }
            default: {}
        }
    }
    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            let fg = rgba[i];
            // Max with a small epsilon to avoid NaNs
            let a_inv = 1.0 / max(fg.a, 1e-6);
            let rgba_sep = vec4(fg.rgb * a_inv, fg.a);
            textureStore(output, vec2<i32>(coords), rgba_sep);
        }
    } 
}

fn premul_alpha(rgba: vec4<f32>) -> vec4<f32> {
    return vec4(rgba.rgb * rgba.a, rgba.a);
}
