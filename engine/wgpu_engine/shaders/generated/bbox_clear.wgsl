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


@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read_write> path_bboxes: array<PathBbox>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let ix = global_id.x;
    if ix < config.n_path {
        path_bboxes[ix].x0 = 0x7fffffff;
        path_bboxes[ix].y0 = 0x7fffffff;
        path_bboxes[ix].x1 = -0x80000000;
        path_bboxes[ix].y1 = -0x80000000;
    }
}
