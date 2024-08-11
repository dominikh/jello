// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

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


@group(0) @binding(0)
var<storage> clip_inp: array<ClipInp>;

@group(0) @binding(1)
var<storage> path_bboxes: array<PathBbox>;

@group(0) @binding(2)
var<storage, read_write> reduced: array<Bic>;

@group(0) @binding(3)
var<storage, read_write> clip_out: array<ClipEl>;

const WG_SIZE = 256u;
var<workgroup> sh_bic: array<Bic, WG_SIZE>;
var<workgroup> sh_parent: array<u32, WG_SIZE>;
var<workgroup> sh_path_ix: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let inp = clip_inp[global_id.x].path_ix;
    let is_push = inp >= 0;
    var bic = Bic(1u - u32(is_push), u32(is_push));
    // reverse scan of bicyclic semigroup
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
    if local_id.x == 0u {
        reduced[wg_id.x] = bic;
    }
    workgroupBarrier();
    let size = sh_bic[0].b;
    bic = Bic();
	if local_id.x + 1u < WG_SIZE {
	   bic = sh_bic[local_id.x + 1u];
	}
    if is_push && bic.a == 0u {
        let local_ix = size - bic.b - 1u;
        sh_parent[local_ix] = local_id.x;
        sh_path_ix[local_ix] = u32(inp);
    }
    workgroupBarrier();
    // TODO: possibly do forward scan here if depth can exceed wg size
    if local_id.x < size {
        let path_ix = sh_path_ix[local_id.x];
        let path_bbox = path_bboxes[path_ix];
        let parent_ix = sh_parent[local_id.x] + wg_id.x * WG_SIZE;
        let bbox = vec4(f32(path_bbox.x0), f32(path_bbox.y0), f32(path_bbox.x1), f32(path_bbox.y1));
        clip_out[global_id.x] = ClipEl(parent_ix, bbox);
    }
}
