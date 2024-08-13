// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Stage to compute counts of number of segments in each tile

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


// TODO: this is cut'n'pasted from path_coarse.
struct AtomicTile {
    backdrop: atomic<i32>,
    segment_count_or_ix: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(2)
var<storage> lines: array<LineSoup>;

@group(0) @binding(3)
var<storage> paths: array<Path>;

@group(0) @binding(4)
var<storage, read_write> tile: array<AtomicTile>;

@group(0) @binding(5)
var<storage, read_write> seg_counts: array<SegmentCount>;

// number of integer cells spanned by interval defined by a, b
fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

// See cpu_shaders/util.rs for explanation of these.
const ONE_MINUS_ULP: f32 = 0.99999994;
const ROBUST_EPSILON: f32 = 2e-7;

// Note regarding clipping to bounding box:
//
// We have to do the backdrop bumps for all tiles to the left of the bbox.
// This should probably be a separate loop. This also causes a numerical
// robustness issue.

// This shader is dispatched with one thread for each line.
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let n_lines = atomicLoad(&bump.lines);
    var count = 0u;
    if global_id.x < n_lines {
        let line = lines[global_id.x];
        // coarse rasterization logic to count number of tiles touched by line
        let is_down = line.p1.y >= line.p0.y;
        let xy0 = select(line.p1, line.p0, is_down);
        let xy1 = select(line.p0, line.p1, is_down);
        let s0 = xy0 * TILE_SCALE;
        let s1 = xy1 * TILE_SCALE;
        let count_x = span(s0.x, s1.x) - 1u;
        count = count_x + span(s0.y, s1.y);
        let line_ix = global_id.x;

        let dx = abs(s1.x - s0.x);
        let dy = s1.y - s0.y;
        if dx + dy == 0.0 {
            // Zero-length segment, drop it. Note, this could be culled in the
            // flattening stage, but eliding the test here would be fragile, as
            // it would be pretty bad to let it slip through.
            return;
        }
        if dy == 0.0 && floor(s0.y) == s0.y {
            return;
        }
        let idxdy = 1.0 / (dx + dy);
        var a = dx * idxdy;
        let is_positive_slope = s1.x >= s0.x;
        let x_sign = select(-1.0, 1.0, is_positive_slope);
        let xt0 = floor(s0.x * x_sign);
        let c = s0.x * x_sign - xt0;
        let y0 = floor(s0.y);
        let ytop = select(y0 + 1.0, ceil(s0.y), s0.y == s1.y);
        let b = min((dy * c + dx * (ytop - s0.y)) * idxdy, ONE_MINUS_ULP);
        let robust_err = floor(a * (f32(count) - 1.0) + b) - f32(count_x);
        if robust_err != 0.0 {
            a -= ROBUST_EPSILON * sign(robust_err);
        }
        let x0 = xt0 * x_sign + select(-1.0, 0.0, is_positive_slope);

        let path = paths[line.path_ix];
        let bbox = vec4<i32>(path.bbox);
        let xmin = min(s0.x, s1.x);
        // If line is to left of bbox, we may still need to do backdrop
        let stride = bbox.z - bbox.x;
        if s0.y >= f32(bbox.w) || s1.y <= f32(bbox.y) || xmin >= f32(bbox.z) || stride == 0 {
            return;
        }
        // Clip line to bounding box. Clipping is done in "i" space.
        var imin = 0u;
        if s0.y < f32(bbox.y) {
            var iminf = round((f32(bbox.y) - y0 + b - a) / (1.0 - a)) - 1.0;
            // Numerical robustness: goal is to find the first i value for which
            // the following predicate is false. Above formula is designed to
            // undershoot by 0.5.
            if y0 + iminf - floor(a * iminf + b) < f32(bbox.y) {
                iminf += 1.0;
            }
            imin = u32(iminf);
        }
        var imax = count;
        if s1.y > f32(bbox.w) {
            var imaxf = round((f32(bbox.w) - y0 + b - a) / (1.0 - a)) - 1.0;
            if y0 + imaxf - floor(a * imaxf + b) < f32(bbox.w) {
                imaxf += 1.0;
            }
            imax = u32(imaxf);
        }
        let delta = select(1, -1, is_down);
        var ymin = 0;
        var ymax = 0;
        if max(s0.x, s1.x) <= f32(bbox.x) {
            ymin = i32(ceil(s0.y));
            ymax = i32(ceil(s1.y));
            imax = imin;
        } else {
            let fudge = select(1.0, 0.0, is_positive_slope);
            if xmin < f32(bbox.x) {
                var f = round((x_sign * (f32(bbox.x) - x0) - b + fudge) / a);
                if (x0 + x_sign * floor(a * f + b) < f32(bbox.x)) == is_positive_slope {
                    f += 1.0;
                }
                let ynext = i32(y0 + f - floor(a * f + b) + 1.0);
                if is_positive_slope {
                    if u32(f) > imin {
                        ymin = i32(y0 + select(1.0, 0.0, y0 == s0.y));
                        ymax = ynext;
                        imin = u32(f);
                    }
                } else {
                    if u32(f) < imax {
                        ymin = ynext;
                        ymax = i32(ceil(s1.y));
                        imax = u32(f);
                    }
                }
            }
            if max(s0.x, s1.x) > f32(bbox.z) {
                var f = round((x_sign * (f32(bbox.z) - x0) - b + fudge) / a);
                if (x0 + x_sign * floor(a * f + b) < f32(bbox.z)) == is_positive_slope {
                    f += 1.0;
                }
                if is_positive_slope {
                    imax = min(imax, u32(f));
                } else {
                    imin = max(imin, u32(f));
                }
            }
        }
        imax = max(imin, imax);
        // Apply backdrop for part of line left of bbox
        ymin = max(ymin, bbox.y);
        ymax = min(ymax, bbox.w);
        for (var y = ymin; y < ymax; y++) {
            let base = i32(path.tiles) + (y - bbox.y) * stride;
            atomicAdd(&tile[base].backdrop, delta);
        }
        var last_z = floor(a * (f32(imin) - 1.0) + b);
        let seg_base = atomicAdd(&bump.seg_counts, imax - imin);
        for (var i = imin; i < imax; i++) {
            let subix = i;
            // coarse rasterization logic
            // Note: we hope fast-math doesn't strength reduce this.
            let zf = a * f32(subix) + b;
            let z = floor(zf);
            // x, y are tile coordinates relative to render target
            let y = i32(y0 + f32(subix) - z);
            let x = i32(x0 + x_sign * z);
            let base = i32(path.tiles) + (y - bbox.y) * stride - bbox.x;
            let top_edge = select(last_z == z, y0 == s0.y, subix == 0u);
            if top_edge && x + 1 < bbox.z {
                let x_bump = max(x + 1, bbox.x);
                atomicAdd(&tile[base + x_bump].backdrop, delta);
            }
            let seg_within_slice = atomicAdd(&tile[base + x].segment_count_or_ix, 1u);
            // Pack two count values into a single u32
            let counts = (seg_within_slice << 16u) | subix;
            let seg_count = SegmentCount(line_ix, counts);
            let seg_ix = seg_base + i - imin;
            if seg_ix < config.seg_counts_size {
                seg_counts[seg_ix] = seg_count;
            }
            // Note: since we're iterating, we have a reliable value for
            // last_z.
            last_z = z;
        }
    }
}
