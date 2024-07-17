// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

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


@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> clip_inp: array<ClipInp>;

@group(0) @binding(2)
var<storage> path_bboxes: array<PathBbox>;

@group(0) @binding(3)
var<storage> reduced: array<Bic>;

@group(0) @binding(4)
var<storage> clip_els: array<ClipEl>;

@group(0) @binding(5)
var<storage, read_write> draw_monoids: array<DrawMonoid>;

@group(0) @binding(6)
var<storage, read_write> clip_bboxes: array<vec4<f32>>;

const WG_SIZE = 256u;
var<workgroup> sh_bic: array<Bic, 510 >;
var<workgroup> sh_stack: array<u32, WG_SIZE>;
var<workgroup> sh_stack_bbox: array<vec4<f32>, WG_SIZE>;
var<workgroup> sh_bbox: array<vec4<f32>, WG_SIZE>;
var<workgroup> sh_link: array<i32, WG_SIZE>;

fn search_link(bic: ptr<function, Bic>, ix_in: u32) -> i32 {
    var ix = ix_in;
    var j = 0u;
    while j < firstTrailingBit(WG_SIZE) {
        let base = 2u * WG_SIZE - (2u << (firstTrailingBit(WG_SIZE) - j));
        if ((ix >> j) & 1u) != 0u {
            let test = bic_combine(sh_bic[base + (ix >> j) - 1u], *bic);
            if test.b > 0u {
                break;
            }
            *bic = test;
            ix -= 1u << j;
        }
        j += 1u;
    }
    if ix > 0u {
        while j > 0u {
            j -= 1u;
            let base = 2u * WG_SIZE - (2u << (firstTrailingBit(WG_SIZE) - j));
            let test = bic_combine(sh_bic[base + (ix >> j) - 1u], *bic);
            if test.b == 0u {
                *bic = test;
                ix -= 1u << j;
            }
        }
    }
    if ix > 0u {
        return i32(ix) - 1;
    } else {
        return i32(~0u - (*bic).a);
    }
}

fn load_clip_path(ix: u32) -> i32 {
    if ix < config.n_clip {
        return clip_inp[ix].path_ix;
    } else {
        return -2147483648;
        // literal too large?
        // return 0x80000000;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    var bic: Bic;
    if local_id.x < wg_id.x {
        bic = reduced[local_id.x];
    }
    sh_bic[local_id.x] = bic;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_bic[local_id.x + (1u << i)];
            bic = bic_combine(bic, other);
        }
        workgroupBarrier();
        sh_bic[local_id.x] = bic;
    }
    workgroupBarrier();
    let stack_size = sh_bic[0].b;
    // TODO: if stack depth > WG_SIZE desired, scan here

    // binary search in stack
    let sp = WG_SIZE - 1u - local_id.x;
    var ix = 0u;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        let probe = ix + ((WG_SIZE / 2u) >> i);
        if sp < sh_bic[probe].b {
            ix = probe;
        }
    }
    let b = sh_bic[ix].b;
    var bbox = vec4(-1e9, -1e9, 1e9, 1e9);
    if sp < b {
        let el = clip_els[ix * WG_SIZE + b - sp - 1u];
        sh_stack[local_id.x] = el.parent_ix;
        bbox = el.bbox;
    }
    // forward scan of bbox values of prefix stack
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        sh_stack_bbox[local_id.x] = bbox;
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            bbox = bbox_intersect(sh_stack_bbox[local_id.x - (1u << i)], bbox);
        }
        workgroupBarrier();
    }
    sh_stack_bbox[local_id.x] = bbox;

    // Read input and compute Bic binary tree
    let inp = load_clip_path(global_id.x);
    let is_push = inp >= 0;
    bic = Bic(1u - u32(is_push), u32(is_push));
    sh_bic[local_id.x] = bic;
    if is_push {
        let path_bbox = path_bboxes[inp];
        bbox = vec4(f32(path_bbox.x0), f32(path_bbox.y0), f32(path_bbox.x1), f32(path_bbox.y1));
    } else {
        bbox = vec4(-1e9, -1e9, 1e9, 1e9);
    }
    var inbase = 0u;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE) - 1u; i += 1u) {
        let outbase = 2u * WG_SIZE - (1u << (firstTrailingBit(WG_SIZE) - i));
        workgroupBarrier();
        if local_id.x < 1u << (firstTrailingBit(WG_SIZE) - 1u - i) {
            let in_off = inbase + local_id.x * 2u;
            sh_bic[outbase + local_id.x] = bic_combine(sh_bic[in_off], sh_bic[in_off + 1u]);
        }
        inbase = outbase;
    }
    workgroupBarrier();
    // search for predecessor node
    bic = Bic();
    var link = search_link(&bic, local_id.x);
    sh_link[local_id.x] = link;
    workgroupBarrier();
    let grandparent = select(link - 1, sh_link[link], link >= 0);
    var parent: i32;
    if link >= 0 {
        parent = i32(wg_id.x * WG_SIZE) + link;
    } else if link + i32(stack_size) >= 0 {
        parent = i32(sh_stack[i32(WG_SIZE) + link]);
    } else {
        parent = -1;
    }
    // bbox scan (intersect) across parent links
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        if i != 0u {
            sh_link[local_id.x] = link;
        }
        sh_bbox[local_id.x] = bbox;
        workgroupBarrier();
        if link >= 0 {
            bbox = bbox_intersect(sh_bbox[link], bbox);
            link = sh_link[link];
        }
        workgroupBarrier();
    }
    if link + i32(stack_size) >= 0 {
        bbox = bbox_intersect(sh_stack_bbox[i32(WG_SIZE) + link], bbox);
    }
    // At this point, bbox is the intersection of bboxes on the path to the root
    sh_bbox[local_id.x] = bbox;
    workgroupBarrier();

    if !is_push && global_id.x < config.n_clip {
        // Fix up drawmonoid so path_ix of EndClip matches BeginClip
        let parent_clip = clip_inp[parent];
        let path_ix = parent_clip.path_ix;
        let parent_ix = parent_clip.ix;
        let ix = ~inp;
        draw_monoids[ix].path_ix = u32(path_ix);
        // Make EndClip point to the same draw data as BeginClip
        draw_monoids[ix].scene_offset = draw_monoids[parent_ix].scene_offset;
        if grandparent >= 0 {
            bbox = sh_bbox[grandparent];
        } else if grandparent + i32(stack_size) >= 0 {
            bbox = sh_stack_bbox[i32(WG_SIZE) + grandparent];
        } else {
            bbox = vec4(-1e9, -1e9, 1e9, 1e9);
        }
    }
    if global_id.x < config.n_clip {
        clip_bboxes[global_id.x] = bbox;
    }
}
