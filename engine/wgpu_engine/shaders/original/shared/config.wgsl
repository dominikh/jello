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

let TILE_WIDTH = 16u;
let TILE_HEIGHT = 16u;
// Number of tiles per bin
let N_TILE_X = 16u;
let N_TILE_Y = 16u;
//let N_TILE = N_TILE_X * N_TILE_Y;
let N_TILE = 256u;

// Not currently supporting non-square tiles
let TILE_SCALE = 0.0625;

// The "split" point between using local memory in fine for the blend stack and spilling to the blend_spill buffer.
// A higher value will increase vgpr ("register") pressure in fine, but decrease required dynamic memory allocation.
// If changing, also change in vello_shaders/src/cpu/coarse.rs.
let BLEND_STACK_SPLIT = 4u;

// The following are computed in draw_leaf from the generic gradient parameters
// encoded in the scene, and stored in the gradient's info struct, for
// consumption during fine rasterization.

// Radial gradient kinds
let RAD_GRAD_KIND_CIRCULAR = 1u;
let RAD_GRAD_KIND_STRIP = 2u;
let RAD_GRAD_KIND_FOCAL_ON_CIRCLE = 3u;
let RAD_GRAD_KIND_CONE = 4u;

// Radial gradient flags
let RAD_GRAD_SWAPPED = 1u;
