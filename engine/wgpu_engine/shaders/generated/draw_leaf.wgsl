// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Finish prefix sum of drawtags, decode draw objects.

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

struct Bic {
    a: u32,
    b: u32,
}

fn bic_combine(x: Bic, y: Bic) -> Bic {
    let m = min(x.b, y.a);
    return Bic(x.a + y.a - m, x.b + y.b - m);
}

struct ClipInp {
    // Index of the draw object.
    ix: u32,
    // This is a packed encoding of an enum with the sign bit as the tag. If positive,
    // this entry is a BeginClip and contains the associated path index. If negative,
    // it is an EndClip and contains the bitwise-not of the EndClip draw object index.
    path_ix: i32,
}

struct ClipEl {
    parent_ix: u32,
    bbox: vec4<f32>,
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

// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Helpers for working with transforms.

struct Transform {
    matrx: vec4<f32>,
    translate: vec2<f32>,
}

fn transform_apply(transform: Transform, p: vec2<f32>) -> vec2<f32> {
    return transform.matrx.xy * p.x + transform.matrx.zw * p.y + transform.translate;
}

fn transform_inverse(transform: Transform) -> Transform {
    let inv_det = 1.0 / (transform.matrx.x * transform.matrx.w - transform.matrx.y * transform.matrx.z);
    let inv_mat = inv_det * vec4(transform.matrx.w, -transform.matrx.y, -transform.matrx.z, transform.matrx.x);
    let inv_tr = mat2x2(inv_mat.xy, inv_mat.zw) * -transform.translate;
    return Transform(inv_mat, inv_tr);
}

fn transform_mul(a: Transform, b: Transform) -> Transform {
    return Transform(
        a.matrx.xyxy * b.matrx.xxzz + a.matrx.zwzw * b.matrx.yyww,
        a.matrx.xy * b.translate.x + a.matrx.zw * b.translate.y + a.translate
    );
}


@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage> reduced: array<DrawMonoid>;

@group(0) @binding(3)
var<storage> path_bbox: array<PathBbox>;

@group(0) @binding(4)
var<storage, read_write> draw_monoid: array<DrawMonoid>;

@group(0) @binding(5)
var<storage, read_write> info: array<u32>;

@group(0) @binding(6)
var<storage, read_write> clip_inp: array<ClipInp>;

// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This file defines utility functions that interact with host-shareable buffer objects. It should
// be imported once following the resource binding declarations in the shader module that access
// them.

// Reads a draw tag from the scene buffer, defaulting to DRAWTAG_NOP if the given `ix` is beyond the
// range of valid draw objects (e.g this can happen if `ix` is derived from an invocation ID in a
// workgroup that partially spans valid range).
//
// This function depends on the following global declarations:
//    * `scene`: array<u32>
//    * `config`: Config (see config.wgsl)
fn read_draw_tag_from_scene(ix: u32) -> u32 {
    var tag_word: u32;
    if ix < config.n_drawobj {
        let tag_ix = config.drawtag_base + ix;
        tag_word = scene[tag_ix];
    } else {
        tag_word = DRAWTAG_NOP;
    }
    return tag_word;
}


const WG_SIZE = 256u;

fn read_transform(transform_base: u32, ix: u32) -> Transform {
    let base = transform_base + ix * 6u;
    let c0 = bitcast<f32>(scene[base]);
    let c1 = bitcast<f32>(scene[base + 1u]);
    let c2 = bitcast<f32>(scene[base + 2u]);
    let c3 = bitcast<f32>(scene[base + 3u]);
    let c4 = bitcast<f32>(scene[base + 4u]);
    let c5 = bitcast<f32>(scene[base + 5u]);
    let matrx = vec4(c0, c1, c2, c3);
    let translate = vec2(c4, c5);
    return Transform(matrx, translate);
}

var<workgroup> sh_scratch: array<DrawMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Reduce prefix of workgroups up to this one
    var agg = draw_monoid_identity();
    if local_id.x < wg_id.x {
        agg = reduced[local_id.x];
    }
    sh_scratch[local_id.x] = agg;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_scratch[local_id.x + (1u << i)];
            agg = combine_draw_monoid(agg, other);
        }
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
    }
    // Two barriers can be eliminated if we use separate shared arrays
    // for prefix and intra-workgroup prefix sum.
    workgroupBarrier();
    var prefix = sh_scratch[0];

    // This is the same division of work as draw_reduce.
    let num_blocks_total = (config.n_drawobj + WG_SIZE - 1u) / WG_SIZE;
    let n_blocks_base = num_blocks_total / WG_SIZE;
    let remainder = num_blocks_total % WG_SIZE;
    let first_block = n_blocks_base * wg_id.x + min(wg_id.x, remainder);
    let n_blocks = n_blocks_base + u32(wg_id.x < remainder);
    var block_start = first_block * WG_SIZE;
    let blocks_end = block_start + n_blocks * WG_SIZE;
    while block_start != blocks_end {
        let ix = block_start + local_id.x;
        let tag_word = read_draw_tag_from_scene(ix);
        agg = map_draw_tag(tag_word);
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
        for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
            workgroupBarrier();
            if local_id.x >= 1u << i {
                let other = sh_scratch[local_id.x - (1u << i)];
                agg = combine_draw_monoid(agg, other);
            }
            workgroupBarrier();
            sh_scratch[local_id.x] = agg;
        }
        var m = prefix;
        workgroupBarrier();
        if local_id.x > 0u {
            m = combine_draw_monoid(m, sh_scratch[local_id.x - 1u]);
        }
        // m now contains exclusive prefix sum of draw monoid
        if ix < config.n_drawobj {
            draw_monoid[ix] = m;
        }
        let dd = config.drawdata_base + m.scene_offset;
        let di = m.info_offset;
        if tag_word == DRAWTAG_FILL_COLOR || tag_word == DRAWTAG_FILL_LIN_GRADIENT ||
            tag_word == DRAWTAG_FILL_RAD_GRADIENT || tag_word == DRAWTAG_FILL_SWEEP_GRADIENT ||
            tag_word == DRAWTAG_FILL_IMAGE || tag_word == DRAWTAG_BEGIN_CLIP
        {
            let bbox = path_bbox[m.path_ix];
            // TODO: bbox is mostly yagni here, sort that out. Maybe clips?
            // let x0 = f32(bbox.x0);
            // let y0 = f32(bbox.y0);
            // let x1 = f32(bbox.x1);
            // let y1 = f32(bbox.y1);
            // let bbox_f = vec4(x0, y0, x1, y1);
            var transform = Transform();
            let draw_flags = bbox.draw_flags;
            if tag_word == DRAWTAG_FILL_LIN_GRADIENT || tag_word == DRAWTAG_FILL_RAD_GRADIENT ||
                tag_word == DRAWTAG_FILL_SWEEP_GRADIENT || tag_word == DRAWTAG_FILL_IMAGE
            {
                transform = read_transform(config.transform_base, bbox.trans_ix);
            }
            switch tag_word {
                case DRAWTAG_FILL_COLOR: {
                    info[di] = draw_flags;
                }
                case DRAWTAG_FILL_LIN_GRADIENT: {
                    info[di] = draw_flags;
                    var p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                    var p1 = bitcast<vec2<f32>>(vec2(scene[dd + 3u], scene[dd + 4u]));
                    p0 = transform_apply(transform, p0);
                    p1 = transform_apply(transform, p1);
                    let dxy = p1 - p0;
                    let scale = 1.0 / dot(dxy, dxy);
                    let line_xy = dxy * scale;
                    let line_c = -dot(p0, line_xy);
                    info[di + 1u] = bitcast<u32>(line_xy.x);
                    info[di + 2u] = bitcast<u32>(line_xy.y);
                    info[di + 3u] = bitcast<u32>(line_c);
                }
                case DRAWTAG_FILL_RAD_GRADIENT: {
                    // Two-point conical gradient implementation based
                    // on the algorithm at <https://skia.org/docs/dev/design/conical/>
                    // This epsilon matches what Skia uses
                    let GRADIENT_EPSILON = 1.0 / f32(1u << 12u);
                    info[di] = draw_flags;
                    var p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                    var p1 = bitcast<vec2<f32>>(vec2(scene[dd + 3u], scene[dd + 4u]));
                    var r0 = bitcast<f32>(scene[dd + 5u]);
                    var r1 = bitcast<f32>(scene[dd + 6u]);
                    let user_to_gradient = transform_inverse(transform);
                    // Output variables
                    var xform = Transform();
                    var focal_x = 0.0;
                    var radius = 0.0;
                    var kind = 0u;
                    var flags = 0u;
                    if abs(r0 - r1) <= GRADIENT_EPSILON {
                        // When the radii are the same, emit a strip gradient
                        kind = RAD_GRAD_KIND_STRIP;
                        let scaled = r0 / distance(p0, p1);
                        xform = transform_mul(
                            two_point_to_unit_line(p0, p1),
                            user_to_gradient
                        );
                        radius = scaled * scaled;
                    } else {
                        // Assume a two point conical gradient unless the centers
                        // are equal.
                        kind = RAD_GRAD_KIND_CONE;
                        if all(p0 == p1) {
                            kind = RAD_GRAD_KIND_CIRCULAR;
                            // Nudge p0 a bit to avoid denormals.
                            p0 += GRADIENT_EPSILON;
                        }
                        if r1 == 0.0 {
                            // If r1 == 0.0, swap the points and radii
                            flags |= RAD_GRAD_SWAPPED;
                            let tmp_p = p0;
                            p0 = p1;
                            p1 = tmp_p;
                            let tmp_r = r0;
                            r0 = r1;
                            r1 = tmp_r;
                        }
                        focal_x = r0 / (r0 - r1);
                        let cf = (1.0 - focal_x) * p0 + focal_x * p1;
                        radius = r1 / (distance(cf, p1));
                        let user_to_unit_line = transform_mul(
                            two_point_to_unit_line(cf, p1),
                            user_to_gradient
                        );
                        var user_to_scaled = user_to_unit_line;
                        // When r == 1.0, focal point is on circle
                        if abs(radius - 1.0) <= GRADIENT_EPSILON {
                            kind = RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                            let scale = 0.5 * abs(1.0 - focal_x);
                            user_to_scaled = transform_mul(
                                Transform(vec4(scale, 0.0, 0.0, scale), vec2(0.0)),
                                user_to_unit_line
                            );
                        } else {
                            let a = radius * radius - 1.0;
                            let scale_ratio = abs(1.0 - focal_x) / a;
                            let scale_x = radius * scale_ratio;
                            let scale_y = sqrt(abs(a)) * scale_ratio;
                            user_to_scaled = transform_mul(
                                Transform(vec4(scale_x, 0.0, 0.0, scale_y), vec2(0.0)),
                                user_to_unit_line
                            );
                        }
                        xform = user_to_scaled;
                    }
                    info[di + 1u] = bitcast<u32>(xform.matrx.x);
                    info[di + 2u] = bitcast<u32>(xform.matrx.y);
                    info[di + 3u] = bitcast<u32>(xform.matrx.z);
                    info[di + 4u] = bitcast<u32>(xform.matrx.w);
                    info[di + 5u] = bitcast<u32>(xform.translate.x);
                    info[di + 6u] = bitcast<u32>(xform.translate.y);
                    info[di + 7u] = bitcast<u32>(focal_x);
                    info[di + 8u] = bitcast<u32>(radius);
                    info[di + 9u] = bitcast<u32>((flags << 3u) | kind);
                }
                case DRAWTAG_FILL_SWEEP_GRADIENT: {
                    info[di] = draw_flags;
                    let p0 = bitcast<vec2<f32>>(vec2(scene[dd + 1u], scene[dd + 2u]));
                    let xform = transform_mul(transform, Transform(vec4(1.0, 0.0, 0.0, 1.0), p0));
                    let inv = transform_inverse(xform);
                    info[di + 1u] = bitcast<u32>(inv.matrx.x);
                    info[di + 2u] = bitcast<u32>(inv.matrx.y);
                    info[di + 3u] = bitcast<u32>(inv.matrx.z);
                    info[di + 4u] = bitcast<u32>(inv.matrx.w);
                    info[di + 5u] = bitcast<u32>(inv.translate.x);
                    info[di + 6u] = bitcast<u32>(inv.translate.y);
                    info[di + 7u] = scene[dd + 3u];
                    info[di + 8u] = scene[dd + 4u];
                }
                case DRAWTAG_FILL_IMAGE: {
                    info[di] = draw_flags;
                    let inv = transform_inverse(transform);
                    info[di + 1u] = bitcast<u32>(inv.matrx.x);
                    info[di + 2u] = bitcast<u32>(inv.matrx.y);
                    info[di + 3u] = bitcast<u32>(inv.matrx.z);
                    info[di + 4u] = bitcast<u32>(inv.matrx.w);
                    info[di + 5u] = bitcast<u32>(inv.translate.x);
                    info[di + 6u] = bitcast<u32>(inv.translate.y);
                    info[di + 7u] = scene[dd];
                    info[di + 8u] = scene[dd + 1u];
                }
                default: {}
            }
        }
        if tag_word == DRAWTAG_BEGIN_CLIP || tag_word == DRAWTAG_END_CLIP {
            var path_ix = ~ix;
            if tag_word == DRAWTAG_BEGIN_CLIP {
                path_ix = m.path_ix;
            }
            clip_inp[m.clip_ix] = ClipInp(ix, i32(path_ix));
        }
        block_start += WG_SIZE;
        // break here on end to save monoid aggregation?
        prefix = combine_draw_monoid(prefix, sh_scratch[WG_SIZE - 1u]);
    }
}

fn two_point_to_unit_line(p0: vec2<f32>, p1: vec2<f32>) -> Transform {
    let tmp1 = from_poly2(p0, p1);
    let inv = transform_inverse(tmp1);
    let tmp2 = from_poly2(vec2(0.0), vec2(1.0, 0.0));
    return transform_mul(tmp2, inv);
}

fn from_poly2(p0: vec2<f32>, p1: vec2<f32>) -> Transform {
    return Transform(
        vec4(p1.y - p0.y, p0.x - p1.x, p1.x - p0.x, p1.y - p0.y),
        vec2(p0.x, p0.y)
    );
}
