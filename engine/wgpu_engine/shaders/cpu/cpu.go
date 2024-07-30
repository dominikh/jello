// Copyright 2023 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Package cpu provides CPU implementations of the compute shaders.
//
// These shaders intentionally replicate the compute shaders in CPU instead of
// using more CPU-friendly alternatives. They're a debug tool, not a viable
// fallback.
package cpu

import (
	"fmt"
	"math"
	"unsafe"

	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/jello/renderer"
	"honnef.co/go/safeish"
)

const WG_SIZE = 256

// XXX rename consts
const PTCL_INITIAL_ALLOC = 64

// Radial gradient kinds
const RAD_GRAD_KIND_CIRCULAR = 1
const RAD_GRAD_KIND_STRIP = 2
const RAD_GRAD_KIND_FOCAL_ON_CIRCLE = 3
const RAD_GRAD_KIND_CONE = 4

// Radial gradient flags
const RAD_GRAD_SWAPPED = 1

const TILE_WIDTH = 16
const TILE_HEIGHT = 16
const TILE_SCALE = 1.0 / 16.0
const N_TILE_X = 16
const N_TILE_Y = 16
const N_TILE = N_TILE_X * N_TILE_Y

const PTCL_INCREMENT = 256
const PTCL_HEADROOM = 2

// Tags for PTCL commands
const CMD_END = 0
const CMD_FILL = 1

// const CMD_STROKE = 2;
const CMD_SOLID = 3
const CMD_COLOR = 5
const CMD_LIN_GRAD = 6
const CMD_RAD_GRAD = 7
const CMD_SWEEP_GRAD = 8
const CMD_IMAGE = 9
const CMD_BEGIN_CLIP = 10
const CMD_END_CLIP = 11
const CMD_JUMP = 12

const PATH_TAG_SEG_TYPE = 3
const PATH_TAG_PATH = 0x10
const PATH_TAG_LINETO = 1
const PATH_TAG_QUADTO = 2
const PATH_TAG_CUBICTO = 3
const PATH_TAG_F32 = 8

const DRAW_INFO_FLAGS_FILL_RULE_BIT = 1

// / The largest floating point value strictly less than 1.
// /
// / This value is used to limit the value of b so that its floor is strictly less
// / than 1. That guarantees that floor(a * i + b) == 0 for i == 0, which lands on
// / the correct first tile.
const ONE_MINUS_ULP = 0.99999994

// / An epsilon to be applied in path numerical robustness.
// /
// / When floor(a * (n - 1) + b) does not match the expected value (the width in
// / grid cells minus one), this delta is applied to a to push it in the correct
// / direction. The theory is that a is not off by more than a few ulp, and it's
// / always in the range of 0..1.
const ROBUST_EPSILON = 2e-7

func assert(b bool) {
	if !b {
		panic("failed assert")
	}
}

func span(a, b float32) uint32 {
	return uint32(max(jmath.Ceil32(max(a, b))-jmath.Floor32(min(a, b)), 1))
}

type Transform [6]float32

var identity = Transform{1, 0, 0, 1, 0, 0}

func (t Transform) apply(p Vec2) Vec2 {
	z := t
	x := z[0]*p.x + z[2]*p.y + z[4]
	y := z[1]*p.x + z[3]*p.y + z[5]
	return Vec2{x, y}
}

func (t Transform) inverse() Transform {
	z := t
	inv_det := 1.0 / (z[0]*z[3] - z[1]*z[2])
	inv_mat := [4]float32{
		z[3] * inv_det,
		-z[1] * inv_det,
		-z[2] * inv_det,
		z[0] * inv_det,
	}
	return Transform{
		inv_mat[0],
		inv_mat[1],
		inv_mat[2],
		inv_mat[3],
		-(inv_mat[0]*z[4] + inv_mat[2]*z[5]),
		-(inv_mat[1]*z[4] + inv_mat[3]*z[5]),
	}
}

func (t Transform) Mul(other Transform) Transform {
	return Transform{
		t[0]*other[0] + t[2]*other[1],
		t[1]*other[0] + t[3]*other[1],
		t[0]*other[2] + t[2]*other[3],
		t[1]*other[2] + t[3]*other[3],
		t[0]*other[4] + t[2]*other[5] + t[4],
		t[1]*other[4] + t[3]*other[5] + t[5],
	}
}

type Vec2 struct {
	x, y float32
}

func (v Vec2) to_array() [2]float32 {
	return [2]float32{v.x, v.y}
}

func Vec2FromArray(arr [2]float32) Vec2 {
	return Vec2{arr[0], arr[1]}
}

func (v Vec2) Add(o Vec2) Vec2 {
	return Vec2{
		v.x + o.x,
		v.y + o.y,
	}
}

func (v Vec2) Sub(o Vec2) Vec2 {
	return Vec2{
		v.x - o.x,
		v.y - o.y,
	}
}

func (v Vec2) Mul(f float32) Vec2 {
	return Vec2{
		v.x * f,
		v.y * f,
	}
}

func (v Vec2) dot(other Vec2) float32 {
	return v.x*other.x + v.y*other.y
}

func (v Vec2) distance(other Vec2) float32 {
	return v.Sub(other).length()
}

func (v Vec2) length() float32 {
	return jmath.Hypot32(v.x, v.y)
}

func (v Vec2) lengthSquared() float32 {
	return v.dot(v)
}

func (v Vec2) NaN() bool {
	return math.IsNaN(float64(v.x)) || math.IsNaN(float64(v.y))
}

func (self Vec2) mix(other Vec2, t float32) Vec2 {
	x := self.x + (other.x-self.x)*t
	y := self.y + (other.y-self.y)*t
	return Vec2{x, y}
}

func (self Vec2) normalize() Vec2 {
	return Vec2{
		self.x / self.length(),
		self.y / self.length(),
	}
}

func (self Vec2) atan2() float32 {
	return jmath.Atan232(self.y, self.x)
}

type CPUBinding interface {
	// One of CPUBuffer, CPUTexture
}

type CPUBuffer []byte

type CPUTexture struct {
	Width  int
	Height int
	Pixels []uint32
}

// XXX move this into safeish
func fromBytes[E any, T *E](b []byte) T {
	if uintptr(len(b)) < unsafe.Sizeof(*new(E)) {
		panic(fmt.Sprintf(
			"buffer of size %d cannot represent object of size %d", len(b), unsafe.Sizeof(*new(E))))
	}

	return safeish.Cast[T](&b[0])
}

func Backdrop(_ uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[2].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[3].(CPUBuffer))

	for drawobj_ix := range config.Layout.NumDrawObjects {
		path := paths[drawobj_ix]
		width := path.Bbox[2] - path.Bbox[0]
		height := path.Bbox[3] - path.Bbox[1]
		base := path.Tiles
		for y := range height {
			var sum int32
			for x := range width {
				tile := &tiles[(base + y*width + x)]
				sum += tile.Backdrop
				tile.Backdrop = sum
			}
		}
	}
}

func BboxClear(_ uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	pathBboxes := safeish.SliceCast[[]renderer.PathBbox](resources[1].(CPUBuffer))
	for i := range config.Layout.NumPaths {
		pathBboxes[i].X0 = 0x7fff_ffff
		pathBboxes[i].Y0 = 0x7fff_ffff
		pathBboxes[i].X1 = -0x8000_0000
		pathBboxes[i].Y1 = -0x8000_0000
	}
}

func PathTagReduce(numWgs uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.PathMonoid](resources[2].(CPUBuffer))

	pathtag_base := config.Layout.PathTagBase
	for i := range int(numWgs) {
		m := renderer.PathMonoid{}
		for j := range WG_SIZE {
			tag := scene[(int(pathtag_base)+i*WG_SIZE)+j]
			m = m.Combine(renderer.NewPathMonoid(tag))
		}
		reduced[i] = m
	}
}

func TileAlloc(_ uint32, resources []CPUBinding) {
	const SX = 1.0 / (TILE_WIDTH)
	const SY = 1.0 / (TILE_HEIGHT)

	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	draw_bboxes := safeish.SliceCast[[][4]float32](resources[2].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[3].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[4].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[5].(CPUBuffer))

	drawtag_base := config.Layout.DrawTagBase
	width_in_tiles := int32(config.WidthInTiles)
	height_in_tiles := int32(config.HeightInTiles)
	for drawobj_ix := range config.Layout.NumDrawObjects {
		drawtag := encoding.DrawTag(scene[(drawtag_base + drawobj_ix)])
		var x0 int32
		var y0 int32
		var x1 int32
		var y1 int32
		if drawtag != encoding.DrawTagNop && drawtag != encoding.DrawTagEndClip {
			bbox := draw_bboxes[drawobj_ix]
			if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
				x0 = int32(jmath.Floor32((bbox[0] * SX)))
				y0 = int32(jmath.Floor32((bbox[1] * SY)))
				x1 = int32(jmath.Ceil32((bbox[2] * SX)))
				y1 = int32(jmath.Ceil32((bbox[3] * SY)))
			}
		}
		ux0 := uint32(jmath.Clamp(width_in_tiles, 0, x0))
		uy0 := uint32(jmath.Clamp(height_in_tiles, 0, y0))
		ux1 := uint32(jmath.Clamp(width_in_tiles, 0, x1))
		uy1 := uint32(jmath.Clamp(height_in_tiles, 0, y1))
		tile_count := (ux1 - ux0) * (uy1 - uy0)
		offset := bump.Tile
		bump.Tile += tile_count
		// We construct it this way because padding is private.
		var path renderer.Path
		path.Bbox = [4]uint32{ux0, uy0, ux1, uy1}
		path.Tiles = offset
		paths[drawobj_ix] = path
		for i := range tile_count {
			tiles[(offset + i)] = renderer.Tile{}
		}
	}
}

func Binning(numWgs uint32, resources []CPUBinding) {
	const SX = 1.0 / (N_TILE_X * TILE_WIDTH)
	const SY = 1.0 / (N_TILE_Y * TILE_HEIGHT)

	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	draw_monoids := safeish.SliceCast[[]renderer.DrawMonoid](resources[1].(CPUBuffer))
	path_bbox_buf := safeish.SliceCast[[]renderer.PathBbox](resources[2].(CPUBuffer))
	clip_bbox_buf := safeish.SliceCast[[][4]float32](resources[3].(CPUBuffer))
	intersected_bbox := safeish.SliceCast[[][4]float32](resources[4].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[5].(CPUBuffer))
	bin_data := safeish.SliceCast[[]uint32](resources[6].(CPUBuffer))
	bin_header := safeish.SliceCast[[]renderer.BinHeader](resources[7].(CPUBuffer))

	for wg := range numWgs {
		// OPT(dh): use arena
		counts := make([]uint32, WG_SIZE)
		bboxes := make([][4]int32, WG_SIZE)
		width_in_bins := int32((config.WidthInTiles + N_TILE_X - 1) / N_TILE_X)
		height_in_bins := int32((config.HeightInTiles + N_TILE_Y - 1) / N_TILE_Y)
		for local_ix := range uint32(WG_SIZE) {
			element_ix := wg*WG_SIZE + local_ix
			var x0 int32
			var y0 int32
			var x1 int32
			var y1 int32
			if element_ix < config.Layout.NumDrawObjects {
				draw_monoid := draw_monoids[element_ix]
				clip_bbox := [4]float32{-1e9, -1e9, 1e9, 1e9}
				if draw_monoid.ClipIdx > 0 {
					if draw_monoid.ClipIdx-1 >= config.Layout.NumClips {
						panic("unreachable")
					}
					clip_bbox = clip_bbox_buf[draw_monoid.ClipIdx-1]
				}
				path_bbox := path_bbox_buf[draw_monoid.PathIdx]
				pb := [4]float32{
					float32(path_bbox.X0),
					float32(path_bbox.Y0),
					float32(path_bbox.X1),
					float32(path_bbox.Y1),
				}
				bbox := bbox_intersect(clip_bbox, pb)
				intersected_bbox[element_ix] = bbox
				if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
					x0 = int32(math.Floor(float64(bbox[0] * SX)))
					y0 = int32(math.Floor(float64(bbox[1] * SY)))
					x1 = int32(math.Ceil(float64(bbox[2] * SX)))
					y1 = int32(math.Ceil(float64(bbox[3] * SY)))
				}
			}
			x0 = jmath.Clamp(x0, 0, width_in_bins)
			y0 = jmath.Clamp(y0, 0, height_in_bins)
			x1 = jmath.Clamp(x1, 0, width_in_bins)
			y1 = jmath.Clamp(y1, 0, height_in_bins)
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					counts[y*width_in_bins+x]++
				}
			}
			bboxes[local_ix] = [4]int32{x0, y0, x1, y1}
		}
		// OPT(dh): use arena
		chunk_offset := make([]uint32, WG_SIZE)
		for local_ix := range uint32(WG_SIZE) {
			global_ix := wg*WG_SIZE + local_ix
			chunk_offset[local_ix] = bump.Binning
			bump.Binning += counts[local_ix]
			bin_header[global_ix] = renderer.BinHeader{
				ElementCount: counts[local_ix],
				ChunkOffset:  chunk_offset[local_ix],
			}
		}
		for local_ix := range uint32(WG_SIZE) {
			element_ix := wg*WG_SIZE + local_ix
			bbox := bboxes[local_ix]
			for y := bbox[1]; y < bbox[3]; y++ {
				for x := bbox[0]; x < bbox[2]; x++ {
					bin_ix := (y*width_in_bins + x)
					ix := config.Layout.BinDataStart + chunk_offset[bin_ix]
					bin_data[ix] = element_ix
					chunk_offset[bin_ix]++
				}
			}
		}
	}
}

func bbox_intersect(a, b [4]float32) [4]float32 {
	return [4]float32{
		max(a[0], b[0]),
		max(a[1], b[1]),
		min(a[2], b[2]),
		min(a[3], b[3]),
	}
}

func ClipLeaf(_ uint32, resources []CPUBinding) {
	type clipStackElement struct {
		// index of draw object
		parent_ix uint32
		path_ix   uint32
		bbox      [4]float32
	}

	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	clip_inp := safeish.SliceCast[[]renderer.Clip](resources[1].(CPUBuffer))
	path_bboxes := safeish.SliceCast[[]renderer.PathBbox](resources[2].(CPUBuffer))
	draw_monoids := safeish.SliceCast[[]renderer.DrawMonoid](resources[5].(CPUBuffer))
	clip_bboxes := safeish.SliceCast[[][4]float32](resources[6].(CPUBuffer))

	// OPT(dh): use arena?
	var stack []clipStackElement
	for global_ix := range config.Layout.NumClips {
		clip_el := clip_inp[global_ix]
		if clip_el.PathIdx >= 0 {
			// begin clip
			path_ix := uint32(clip_el.PathIdx)
			path_bbox := path_bboxes[path_ix]
			p_bbox := [4]float32{
				float32(path_bbox.X0),
				float32(path_bbox.Y0),
				float32(path_bbox.X1),
				float32(path_bbox.Y1),
			}
			var bbox [4]float32
			if len(stack) > 0 {
				last := stack[len(stack)-1]
				bbox = [4]float32{
					max(p_bbox[0], last.bbox[0]),
					max(p_bbox[1], last.bbox[1]),
					max(p_bbox[2], last.bbox[2]),
					max(p_bbox[3], last.bbox[3]),
				}
			} else {
				bbox = p_bbox
			}
			clip_bboxes[global_ix] = bbox
			parent_ix := clip_el.Idx
			stack = append(stack, clipStackElement{
				parent_ix,
				path_ix,
				bbox,
			})
		} else {
			// end clip
			tos := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			var bbox [4]float32
			if len(stack) > 0 {
				bbox = stack[len(stack)-1].bbox
			} else {
				bbox = [4]float32{-1e9, -1e9, 1e9, 1e9}
			}
			clip_bboxes[global_ix] = bbox
			draw_monoids[clip_el.Idx].PathIdx = tos.path_ix
			draw_monoids[clip_el.Idx].SceneOffset =
				draw_monoids[tos.parent_ix].SceneOffset
		}
	}
}

func ClipReduce(numWgs uint32, resources []CPUBinding) {
	clip_inp := safeish.SliceCast[[]renderer.Clip](resources[0].(CPUBuffer))
	path_bboxes := safeish.SliceCast[[]renderer.PathBbox](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.ClipBic](resources[2].(CPUBuffer))
	clip_out := safeish.SliceCast[[]renderer.ClipElement](resources[3].(CPUBuffer))

	// OPT(dh): use arena?
	scratch := make([]uint32, 0, WG_SIZE)
	for wg_ix := range numWgs {
		scratch = scratch[:0]
		var bic_reduced renderer.ClipBic
		// reverse scan
		for local_ix := WG_SIZE - 1; local_ix >= 0; local_ix-- {
			global_ix := wg_ix*WG_SIZE + uint32(local_ix)
			inp := clip_inp[global_ix].PathIdx
			var is_push uint32
			if inp >= 0 {
				is_push = 1
			}
			bic := renderer.ClipBic{A: 1 - is_push, B: is_push}
			bic_reduced = bic.Combine(bic_reduced)
			if is_push != 0 && bic_reduced.A == 0 {
				scratch = append(scratch, global_ix)
			}
		}
		reduced[wg_ix] = bic_reduced
		for i := len(scratch) - 1; i >= 0; i-- {
			parent_ix := scratch[i]
			var clip_el renderer.ClipElement
			clip_el.ParentIdx = parent_ix
			path_ix := clip_inp[parent_ix].PathIdx
			path_bbox := path_bboxes[path_ix]
			clip_el.Bbox = [4]float32{
				float32(path_bbox.X0),
				float32(path_bbox.Y0),
				float32(path_bbox.X1),
				float32(path_bbox.Y1),
			}
			global_ix := wg_ix*WG_SIZE + uint32(i)
			clip_out[global_ix] = clip_el
		}
	}
}

func DrawReduce(n_wg uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.DrawMonoid](resources[2].(CPUBuffer))

	num_blocks_total := (config.Layout.NumDrawObjects + (WG_SIZE - 1)) / WG_SIZE
	n_blocks_base := num_blocks_total / WG_SIZE
	remainder := num_blocks_total % WG_SIZE
	for i := range n_wg {
		first_block := n_blocks_base*i + min(i, remainder)
		var b uint32
		if i < remainder {
			b = 1
		}
		n_blocks := n_blocks_base + b
		var m renderer.DrawMonoid
		for j := range WG_SIZE * n_blocks {
			ix := (first_block * WG_SIZE) + j
			tag := read_draw_tag_from_scene(config, scene, ix)
			m = m.Combine(renderer.NewDrawMonoid(encoding.DrawTag(tag)))
		}
		reduced[i] = m
	}
}

const DRAWTAG_NOP = 0

// Read draw tag, guarded by number of draw objects.
//
// The `ix` argument is allowed to exceed the number of draw objects,
// in which case a NOP is returned.
func read_draw_tag_from_scene(config *renderer.ConfigUniform, scene []uint32, ix uint32) uint32 {
	if ix < config.Layout.NumDrawObjects {
		tag_ix := config.Layout.DrawTagBase + ix
		return scene[tag_ix]
	} else {
		return DRAWTAG_NOP
	}
}

func PathCountSetup(_ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[0].(CPUBuffer))
	indirect := fromBytes[renderer.IndirectCount](resources[1].(CPUBuffer))

	lines := bump.Lines
	indirect.X = (lines + (WG_SIZE - 1)) / WG_SIZE
	indirect.Y = 1
	indirect.Z = 1
}

func PathTagScan(n_wg uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.PathMonoid](resources[2].(CPUBuffer))
	tag_monoids := safeish.SliceCast[[]renderer.PathMonoid](resources[3].(CPUBuffer))

	pathtag_base := config.Layout.PathTagBase
	var prefix renderer.PathMonoid
	for i := range uint32(n_wg) {
		m := prefix
		for j := range uint32(WG_SIZE) {
			ix := (i * WG_SIZE) + j
			tag_monoids[ix] = m
			tag := scene[pathtag_base+ix]
			m = m.Combine(renderer.NewPathMonoid(tag))
		}
		prefix = prefix.Combine(reduced[i])
	}
}

func PathTiling(_ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[0].(CPUBuffer))
	seg_counts := safeish.SliceCast[[]renderer.SegmentCount](resources[1].(CPUBuffer))
	lines := safeish.SliceCast[[]renderer.LineSoup](resources[2].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[3].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[4].(CPUBuffer))
	segments := safeish.SliceCast[[]renderer.PathSegment](resources[5].(CPUBuffer))

	for seg_ix := range bump.SegCounts {
		seg_count := seg_counts[seg_ix]
		line := lines[seg_count.LineIdx]
		counts := seg_count.Counts
		seg_within_slice := counts >> 16
		seg_within_line := counts & 0xffff

		// coarse rasterization logic
		p0 := Vec2FromArray(line.P0)
		p1 := Vec2FromArray(line.P1)
		is_down := p1.y >= p0.y
		var xy0, xy1 Vec2
		if is_down {
			xy0, xy1 = p0, p1
		} else {
			xy0, xy1 = p1, p0
		}
		s0 := xy0.Mul(TILE_SCALE)
		s1 := xy1.Mul(TILE_SCALE)
		count_x := span(s0.x, s1.x) - 1
		count := count_x + span(s0.y, s1.y)

		dx := jmath.Abs32(s1.x - s0.x)
		dy := s1.y - s0.y
		idxdy := 1.0 / (dx + dy)
		a := dx * idxdy
		is_positive_slope := s1.x >= s0.x
		var sign float32
		if is_positive_slope {
			sign = 1.0
		} else {
			sign = -1.0
		}
		xt0 := jmath.Floor32(s0.x * sign)
		c := s0.x*sign - xt0
		y0 := jmath.Floor32(s0.y)
		var ytop float32
		if s0.y == s1.y {
			ytop = jmath.Ceil32(s0.y)
		} else {
			ytop = y0 + 1.0
		}
		b := min(((dy*c + dx*(ytop-s0.y)) * idxdy), ONE_MINUS_ULP)
		robust_err := jmath.Floor32(a*(float32(count)-1.0)+b) - float32(count_x)
		if robust_err != 0.0 {
			a -= jmath.Copysign32(ROBUST_EPSILON, robust_err)
		}
		x0 := xt0 * sign
		if is_positive_slope {
			x0 += 0.0
		} else {
			x0 += -1.0
		}
		z := jmath.Floor32(a*float32(seg_within_line) + b)
		x := int32(x0) + int32(sign*z)
		y := int32(y0 + float32(seg_within_line) - z)

		path := paths[line.PathIdx]
		bboxu := path.Bbox
		bbox := [4]int32{
			int32(bboxu[0]),
			int32(bboxu[1]),
			int32(bboxu[2]),
			int32(bboxu[3]),
		}
		stride := bbox[2] - bbox[0]
		tile_ix := int32(path.Tiles) + (y-bbox[1])*stride + x - bbox[0]
		tile := tiles[tile_ix]
		seg_start := ^tile.SegmentCountOrIdx
		if (int32(seg_start)) < 0 {
			continue
		}
		tile_xy := Vec2{float32(x) * TILE_WIDTH, float32(y) * TILE_HEIGHT}
		tile_xy1 := tile_xy.Add(Vec2{TILE_WIDTH, TILE_HEIGHT})

		if seg_within_line > 0 {
			z_prev := jmath.Floor32(a*(float32(seg_within_line)-1.0) + b)
			if z == z_prev {
				// Top edge is clipped
				xt := xy0.x + (xy1.x-xy0.x)*(tile_xy.y-xy0.y)/(xy1.y-xy0.y)
				xt = jmath.Clamp(xt, tile_xy.x+1e-3, tile_xy1.x)
				xy0 = Vec2{xt, tile_xy.y}
			} else {
				// If is_positive_slope, left edge is clipped, otherwise right
				var x_clip float32
				if is_positive_slope {
					x_clip = tile_xy.x
				} else {
					x_clip = tile_xy1.x
				}
				yt := xy0.y + (xy1.y-xy0.y)*(x_clip-xy0.x)/(xy1.x-xy0.x)
				yt = jmath.Clamp(yt, tile_xy.y+1e-3, tile_xy1.y)
				xy0 = Vec2{x_clip, yt}
			}
		}
		if seg_within_line < count-1 {
			z_next := jmath.Floor32(a*(float32(seg_within_line)+1.0) + b)
			if z == z_next {
				// Bottom edge is clipped
				xt := xy0.x + (xy1.x-xy0.x)*(tile_xy1.y-xy0.y)/(xy1.y-xy0.y)
				xt = jmath.Clamp(xt, tile_xy.x+1e-3, tile_xy1.x)
				xy1 = Vec2{xt, tile_xy1.y}
			} else {
				// If is_positive_slope, right edge is clipped, otherwise left
				var x_clip float32
				if is_positive_slope {
					x_clip = tile_xy1.x
				} else {
					x_clip = tile_xy.x
				}
				yt := xy0.y + (xy1.y-xy0.y)*(x_clip-xy0.x)/(xy1.x-xy0.x)
				yt = jmath.Clamp(yt, tile_xy.y+1e-3, tile_xy1.y)
				xy1 = Vec2{x_clip, yt}
			}
		}
		y_edge := float32(1e9)
		// Apply numerical robustness logic
		p0 = xy0.Sub(tile_xy)
		p1 = xy1.Sub(tile_xy)
		const EPSILON = 1e-6
		if p0.x == 0.0 {
			if p1.x == 0.0 {
				p0.x = EPSILON
				if p0.y == 0.0 {
					// Entire tile
					p1.x = EPSILON
					p1.y = TILE_HEIGHT
				} else {
					// Make segment disappear
					p1.x = 2.0 * EPSILON
					p1.y = p0.y
				}
			} else if p0.y == 0.0 {
				p0.x = EPSILON
			} else {
				y_edge = p0.y
			}
		} else if p1.x == 0.0 {
			if p1.y == 0.0 {
				p1.x = EPSILON
			} else {
				y_edge = p1.y
			}
		}
		if p0.x == jmath.Floor32(p0.x) && p0.x != 0.0 {
			p0.x -= EPSILON
		}
		if p1.x == jmath.Floor32(p1.x) && p1.x != 0.0 {
			p1.x -= EPSILON
		}
		if !is_down {
			p0, p1 = p1, p0
		}
		segment := renderer.PathSegment{
			Point0: p0.to_array(),
			Point1: p1.to_array(),
			YEdge:  y_edge,
		}
		assert(p0.x >= 0.0 && p0.x <= TILE_WIDTH)
		assert(p0.y >= 0.0 && p0.y <= TILE_HEIGHT)
		assert(p1.x >= 0.0 && p1.x <= TILE_WIDTH)
		assert(p1.y >= 0.0 && p1.y <= TILE_HEIGHT)
		segments[(seg_start + seg_within_slice)] = segment
	}
}

func PathCount(_ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[1].(CPUBuffer))
	lines := safeish.SliceCast[[]renderer.LineSoup](resources[2].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[3].(CPUBuffer))
	tile := safeish.SliceCast[[]renderer.Tile](resources[4].(CPUBuffer))
	seg_counts := safeish.SliceCast[[]renderer.SegmentCount](resources[5].(CPUBuffer))

	for line_ix := range bump.Lines {
		line := lines[line_ix]
		p0 := Vec2FromArray(line.P0)
		p1 := Vec2FromArray(line.P1)
		is_down := p1.y >= p0.y
		var xy0, xy1 Vec2
		if is_down {
			xy0, xy1 = p0, p1
		} else {
			xy0, xy1 = p1, p0
		}
		s0 := xy0.Mul(TILE_SCALE)
		s1 := xy1.Mul(TILE_SCALE)
		count_x := span(s0.x, s1.x) - 1
		count := count_x + span(s0.y, s1.y)

		dx := jmath.Abs32(s1.x - s0.x)
		dy := s1.y - s0.y
		if dx+dy == 0.0 {
			continue
		}
		if dy == 0.0 && jmath.Floor32(s0.y) == s0.y {
			continue
		}
		idxdy := 1.0 / (dx + dy)
		a := dx * idxdy
		is_positive_slope := s1.x >= s0.x
		var sign float32
		if is_positive_slope {
			sign = 1
		} else {
			sign = -1
		}
		xt0 := jmath.Floor32(s0.x * sign)
		c := s0.x*sign - xt0
		y0 := jmath.Floor32(s0.y)
		var ytop float32
		if s0.y == s1.y {
			ytop = jmath.Ceil32(s0.y)
		} else {
			ytop = y0 + 1.0
		}
		b := min(((dy*c + dx*(ytop-s0.y)) * idxdy), ONE_MINUS_ULP)
		robust_err := jmath.Floor32(a*(float32(count)-1.0)+b) - float32(count_x)
		if robust_err != 0.0 {
			a -= jmath.Copysign32(ROBUST_EPSILON, robust_err)
		}
		x0 := xt0 * sign
		if is_positive_slope {
			x0 += 0.0
		} else {
			x0 += -1.0
		}

		path := paths[line.PathIdx]
		bboxu := path.Bbox
		bbox := [4]int32{
			int32(bboxu[0]),
			int32(bboxu[1]),
			int32(bboxu[2]),
			int32(bboxu[3]),
		}
		xmin := min(s0.x, s1.x)
		stride := bbox[2] - bbox[0]
		if s0.y >= float32(bbox[3]) || s1.y < float32(bbox[1]) || xmin >= float32(bbox[2]) || stride == 0 {
			continue
		}
		// Clip line to bounding box. Clipping is done in "i" space.
		imin := uint32(0)
		if s0.y < float32(bbox[1]) {
			iminf := jmath.Round32((float32(bbox[1])-y0+b-a)/(1.0-a)) - 1.0
			if y0+iminf-jmath.Floor32(a*iminf+b) < float32(bbox[1]) {
				iminf += 1.0
			}
			imin = uint32(iminf)
		}
		imax := count
		if s1.y > float32(bbox[3]) {
			imaxf := jmath.Round32((float32(bbox[3])-y0+b-a)/(1.0-a)) - 1.0
			if y0+imaxf-jmath.Floor32(a*imaxf+b) < float32(bbox[3]) {
				imaxf += 1.0
			}
			imax = uint32(imaxf)
		}
		var delta int32
		if is_down {
			delta = -1
		} else {
			delta = 1
		}
		var ymin, ymax int32
		if max(s0.x, s1.x) < float32(bbox[0]) {
			ymin = int32(jmath.Ceil32(s0.y))
			ymax = int32(jmath.Ceil32(s1.y))
			imax = imin
		} else {
			var fudge float32
			if is_positive_slope {
				fudge = 0.0
			} else {
				fudge = 1.0
			}
			if xmin < float32(bbox[0]) {
				f := jmath.Round32((sign*(float32(bbox[0])-x0) - b + fudge) / a)
				if (x0+sign*jmath.Floor32(a*f+b) < float32(bbox[0])) == is_positive_slope {
					f += 1.0
				}
				ynext := int32(y0 + f - jmath.Floor32(a*f+b) + 1.0)
				if is_positive_slope {
					if uint32(f) > imin {
						ymin = int32(y0)
						if y0 == s0.y {
							ymin += 0
						} else {
							ymin += 1
						}
						ymax = ynext
						imin = uint32(f)
					}
				} else if (uint32(f)) < imax {
					ymin = ynext
					ymax = int32(jmath.Ceil32(s1.y))
					imax = uint32(f)
				}
			}
			if max(s0.x, s1.x) > float32(bbox[2]) {
				f := jmath.Round32((sign*(float32(bbox[2])-x0) - b + fudge) / a)
				if (x0+sign*jmath.Floor32(a*f+b) < float32(bbox[2])) == is_positive_slope {
					f += 1.0
				}
				if is_positive_slope {
					imax = min(imax, uint32(f))
				} else {
					imin = max(imin, uint32(f))
				}
			}
		}
		imax = max(imin, imax)
		ymin = max(ymin, bbox[1])
		ymax = min(ymax, bbox[3])
		for y := ymin; y < ymax; y++ {
			base := int32(path.Tiles) + (y-bbox[1])*stride
			tile[base].Backdrop += delta
		}
		last_z := jmath.Floor32(a*(float32(imin)-1.0) + b)
		seg_base := bump.SegCounts
		bump.SegCounts += imax - imin
		for i := imin; i < imax; i++ {
			zf := a*float32(i) + b
			z := jmath.Floor32(zf)
			y := int32(y0 + float32(i) - z)
			x := int32(x0 + sign*z)
			base := int32(path.Tiles) + (y-bbox[1])*stride - bbox[0]
			var top_edge bool
			if i == 0 {
				top_edge = y0 == s0.y
			} else {
				top_edge = last_z == z
			}
			if top_edge && x+1 < bbox[2] {
				x_bump := max((x + 1), bbox[0])
				tile[(base + x_bump)].Backdrop += delta
			}
			// .segments is another name for the .count field; it's overloaded
			seg_within_slice := tile[(base + x)].SegmentCountOrIdx
			tile[(base + x)].SegmentCountOrIdx += 1
			counts := (seg_within_slice << 16) | i
			seg_count := renderer.SegmentCount{LineIdx: line_ix, Counts: counts}
			seg_counts[(seg_base + i - imin)] = seg_count
			last_z = z
		}
	}
}

type TileState struct {
	cmd_offset uint32
	cmd_limit  uint32
}

func NewTileState(tile_ix uint32) TileState {
	cmd_offset := tile_ix * PTCL_INITIAL_ALLOC
	cmd_limit := cmd_offset + (PTCL_INITIAL_ALLOC - PTCL_HEADROOM)
	return TileState{
		cmd_offset,
		cmd_limit,
	}
}

func (self *TileState) alloc_cmd(
	size uint32,
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
) {
	if self.cmd_offset+size >= self.cmd_limit {
		ptcl_dyn_start := config.WidthInTiles * config.HeightInTiles * PTCL_INITIAL_ALLOC
		chunk_size := max(PTCL_INCREMENT, size+PTCL_HEADROOM)
		new_cmd := ptcl_dyn_start + bump.Ptcl
		bump.Ptcl += chunk_size
		ptcl[self.cmd_offset] = CMD_JUMP
		ptcl[self.cmd_offset+1] = new_cmd
		self.cmd_offset = new_cmd
		self.cmd_limit = new_cmd + (PTCL_INCREMENT - PTCL_HEADROOM)
	}
}

func (self *TileState) write(ptcl []uint32, offset uint32, value uint32) {
	ptcl[self.cmd_offset+offset] = value
}

func (self *TileState) write_path(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	tile *renderer.Tile,
	draw_flags uint32,
) {
	n_segs := tile.SegmentCountOrIdx
	if n_segs != 0 {
		seg_ix := bump.Segments
		tile.SegmentCountOrIdx = ^seg_ix
		bump.Segments += n_segs
		self.alloc_cmd(4, config, bump, ptcl)
		self.write(ptcl, 0, CMD_FILL)
		var even_odd uint32
		if (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0 {
			even_odd = 1
		}
		size_and_rule := (n_segs << 1) | even_odd
		self.write(ptcl, 1, size_and_rule)
		self.write(ptcl, 2, seg_ix)
		self.write(ptcl, 3, uint32(tile.Backdrop))
		self.cmd_offset += 4
	} else {
		self.alloc_cmd(1, config, bump, ptcl)
		self.write(ptcl, 0, CMD_SOLID)
		self.cmd_offset += 1
	}
}

func (self *TileState) write_color(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	rgba_color uint32,
) {
	self.alloc_cmd(2, config, bump, ptcl)
	self.write(ptcl, 0, CMD_COLOR)
	self.write(ptcl, 1, rgba_color)
	self.cmd_offset += 2
}

func (self *TileState) write_image(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	info_offset uint32,
) {
	self.alloc_cmd(2, config, bump, ptcl)
	self.write(ptcl, 0, CMD_IMAGE)
	self.write(ptcl, 1, info_offset)
	self.cmd_offset += 2
}

func (self *TileState) write_grad(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	ty uint32,
	index uint32,
	info_offset uint32,
) {
	self.alloc_cmd(3, config, bump, ptcl)
	self.write(ptcl, 0, ty)
	self.write(ptcl, 1, index)
	self.write(ptcl, 2, info_offset)
	self.cmd_offset += 3
}

func (self *TileState) write_begin_clip(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
) {
	self.alloc_cmd(1, config, bump, ptcl)
	self.write(ptcl, 0, CMD_BEGIN_CLIP)
	self.cmd_offset += 1
}

func (self *TileState) write_end_clip(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	blend uint32,
	alpha float32,
) {
	self.alloc_cmd(3, config, bump, ptcl)
	self.write(ptcl, 0, CMD_END_CLIP)
	self.write(ptcl, 1, blend)
	self.write(ptcl, 2, math.Float32bits(alpha))
	self.cmd_offset += 3
}

func Coarse(_ uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	draw_monoids := safeish.SliceCast[[]renderer.DrawMonoid](resources[2].(CPUBuffer))
	bin_headers := safeish.SliceCast[[]renderer.BinHeader](resources[3].(CPUBuffer))
	info_bin_data := safeish.SliceCast[[]uint32](resources[4].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[5].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[6].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[7].(CPUBuffer))
	ptcl := safeish.SliceCast[[]uint32](resources[8].(CPUBuffer))

	width_in_tiles := config.WidthInTiles
	height_in_tiles := config.HeightInTiles
	width_in_bins := (width_in_tiles + N_TILE_X - 1) / N_TILE_X
	height_in_bins := (height_in_tiles + N_TILE_Y - 1) / N_TILE_Y
	n_bins := width_in_bins * height_in_bins
	bin_data_start := config.Layout.BinDataStart
	drawtag_base := config.Layout.DrawTagBase
	compacted := make([][]uint32, N_TILE)
	n_partitions := (config.Layout.NumDrawObjects + N_TILE - 1) / N_TILE
	for bin := range n_bins {
		for i := range compacted {
			compacted[i] = compacted[i][:0]
		}
		bin_x := bin % width_in_bins
		bin_y := bin / width_in_bins
		bin_tile_x := N_TILE_X * bin_x
		bin_tile_y := N_TILE_Y * bin_y
		for part := range n_partitions {
			in_ix := part*N_TILE + bin
			bin_header := bin_headers[in_ix]
			start := bin_data_start + bin_header.ChunkOffset
			for i := range bin_header.ElementCount {
				drawobj_ix := info_bin_data[start+i]
				tag := scene[drawtag_base+drawobj_ix]
				if encoding.DrawTag(tag) != encoding.DrawTagNop {
					draw_monoid := draw_monoids[drawobj_ix]
					path_ix := draw_monoid.PathIdx
					path := paths[path_ix]
					dx := int32(path.Bbox[0]) - int32(bin_tile_x)
					dy := int32(path.Bbox[1]) - int32(bin_tile_y)
					x0 := jmath.Clamp(dx, 0, N_TILE_X)
					y0 := jmath.Clamp(dy, 0, N_TILE_Y)
					x1 := jmath.Clamp(int32(path.Bbox[2])-int32(bin_tile_x), 0, N_TILE_X)
					y1 := jmath.Clamp(int32(path.Bbox[3])-int32(bin_tile_y), 0, N_TILE_Y)
					for y := y0; y < y1; y++ {
						for x := x0; x < x1; x++ {
							compacted[(y*N_TILE_X + x)] = append(compacted[(y*N_TILE_X+x)], drawobj_ix)
						}
					}
				}
			}
		}
		// compacted now has the list of draw objects for each tile.
		// While the WGSL source does at most 256 draw objects at a time,
		// this version does all the draw objects in a tile.
		for tile_ix := range N_TILE {
			tile_x := uint32(tile_ix % N_TILE_X)
			tile_y := uint32(tile_ix / N_TILE_X)
			this_tile_ix := (bin_tile_y+tile_y)*width_in_tiles + bin_tile_x + tile_x
			tile_state := NewTileState(this_tile_ix)
			blend_offset := tile_state.cmd_offset
			tile_state.cmd_offset += 1
			clip_depth := 0
			clip_zero_depth := 0
			for _, drawobj_ix := range compacted[tile_ix] {
				drawtag := scene[(drawtag_base + drawobj_ix)]
				if clip_zero_depth == 0 {
					draw_monoid := draw_monoids[drawobj_ix]
					path_ix := draw_monoid.PathIdx
					path := paths[path_ix]
					bbox := path.Bbox
					stride := bbox[2] - bbox[0]
					x := bin_tile_x + tile_x - bbox[0]
					y := bin_tile_y + tile_y - bbox[1]
					tile := &tiles[path.Tiles+y*stride+x]
					is_clip := (drawtag & 1) != 0
					is_blend := false
					dd := config.Layout.DrawDataBase + draw_monoid.SceneOffset
					di := draw_monoid.InfoOffset
					if is_clip {
						const BLEND_CLIP = (128 << 8) | 3
						blend := scene[dd]
						is_blend = blend != BLEND_CLIP
					}

					draw_flags := info_bin_data[di]
					even_odd := (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0
					n_segs := tile.SegmentCountOrIdx

					// If this draw object represents an even-odd fill and we
					// know that no line segment crosses this tile and then this
					// draw object should not contribute to the tile if its
					// backdrop (i.e. the winding number of its top-left corner)
					// is even.
					backdrop_clear := (even_odd && jmath.AbsInt32(tile.Backdrop)&1 == 0) || (!even_odd && tile.Backdrop == 0)
					include_tile := n_segs != 0 || (backdrop_clear == is_clip) || is_blend
					if include_tile {
						switch encoding.DrawTag(drawtag) {
						case encoding.DrawTagColor:
							tile_state.write_path(config, bump, ptcl, tile, draw_flags)
							rgba_color := scene[dd]
							tile_state.write_color(config, bump, ptcl, rgba_color)

						case encoding.DrawTagImage:
							tile_state.write_path(config, bump, ptcl, tile, draw_flags)
							tile_state.write_image(config, bump, ptcl, di+1)

						case encoding.DrawTagLinearGradient:
							tile_state.write_path(config, bump, ptcl, tile, draw_flags)
							index := scene[dd]
							tile_state.write_grad(
								config,
								bump,
								ptcl,
								CMD_LIN_GRAD,
								index,
								di+1,
							)

						case encoding.DrawTagRadialGradient:
							tile_state.write_path(config, bump, ptcl, tile, draw_flags)
							index := scene[dd]
							tile_state.write_grad(
								config,
								bump,
								ptcl,
								CMD_RAD_GRAD,
								index,
								di+1,
							)

						case encoding.DrawTagSweepGradient:
							tile_state.write_path(config, bump, ptcl, tile, draw_flags)
							index := scene[dd]
							tile_state.write_grad(
								config,
								bump,
								ptcl,
								CMD_SWEEP_GRAD,
								index,
								di+1,
							)

						case encoding.DrawTagBeginClip:
							if tile.SegmentCountOrIdx == 0 && tile.Backdrop == 0 {
								clip_zero_depth = clip_depth + 1
							} else {
								tile_state.write_begin_clip(config, bump, ptcl)
								// TODO: update blend depth
							}
							clip_depth++

						case encoding.DrawTagEndClip:
							clip_depth--
							// A clip shape is always a non-zero fill (draw_flags=0).
							tile_state.write_path(config, bump, ptcl, tile, 0)
							blend := scene[dd]
							alpha := math.Float32frombits(scene[dd+1])
							tile_state.write_end_clip(config, bump, ptcl, blend, alpha)

						default:
							panic("unreachable")
						}
					}
				} else {
					// In "clip zero" state, suppress all drawing
					switch encoding.DrawTag(drawtag) {
					case encoding.DrawTagBeginClip:
						clip_depth++
					case encoding.DrawTagEndClip:
						if clip_depth == clip_zero_depth {
							clip_zero_depth = 0
						}
						clip_depth--
					}
				}
			}

			if bin_tile_x+tile_x < width_in_tiles && bin_tile_y+tile_y < height_in_tiles {
				ptcl[tile_state.cmd_offset] = CMD_END
				scratch_size := uint32(0) // TODO: actually compute blend depth
				ptcl[blend_offset] = bump.Blend
				bump.Blend += scratch_size
			}
		}
	}
}

func two_point_to_unit_line(p0 Vec2, p1 Vec2) Transform {
	tmp1 := from_poly2(p0, p1)
	inv := tmp1.inverse()
	tmp2 := from_poly2(Vec2{}, Vec2{1.0, 0.0})
	return tmp2.Mul(inv)
}

func from_poly2(p0 Vec2, p1 Vec2) Transform {
	return Transform{
		p1.y - p0.y,
		p0.x - p1.x,
		p1.x - p0.x,
		p1.y - p0.y,
		p0.x,
		p0.y}
}

func DrawLeaf(n_wg uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.DrawMonoid](resources[2].(CPUBuffer))
	path_bbox := safeish.SliceCast[[]renderer.PathBbox](resources[3].(CPUBuffer))
	draw_monoid := safeish.SliceCast[[]renderer.DrawMonoid](resources[4].(CPUBuffer))
	info := safeish.SliceCast[[]uint32](resources[5].(CPUBuffer))
	clip_inp := safeish.SliceCast[[]renderer.Clip](resources[6].(CPUBuffer))

	num_blocks_total := (config.Layout.NumDrawObjects + (WG_SIZE - 1)) / WG_SIZE
	n_blocks_base := num_blocks_total / WG_SIZE
	remainder := num_blocks_total % WG_SIZE
	prefix := renderer.DrawMonoid{}
	for i := range n_wg {
		first_block := n_blocks_base*i + min(i, remainder)
		n_blocks := n_blocks_base
		if i < remainder {
			n_blocks++
		}
		m := prefix
		for j := range WG_SIZE * n_blocks {
			ix := uint32(first_block*WG_SIZE) + uint32(j)
			tag_raw := read_draw_tag_from_scene(config, scene, ix)
			tag_word := encoding.DrawTag(tag_raw)
			// store exclusive prefix sum
			if ix < config.Layout.NumDrawObjects {
				draw_monoid[ix] = m
			}
			m_next := m.Combine(renderer.NewDrawMonoid(tag_word))
			dd := config.Layout.DrawDataBase + m.SceneOffset
			di := m.InfoOffset
			if tag_word == encoding.DrawTagColor ||
				tag_word == encoding.DrawTagLinearGradient ||
				tag_word == encoding.DrawTagRadialGradient ||
				tag_word == encoding.DrawTagSweepGradient ||
				tag_word == encoding.DrawTagImage ||
				tag_word == encoding.DrawTagBeginClip {
				bbox := path_bbox[m.PathIdx]
				transform := transformRead(config.Layout.TransformBase, bbox.TransIdx, scene)
				draw_flags := bbox.DrawFlags
				switch tag_word {
				case encoding.DrawTagColor:
					info[di] = draw_flags

				case encoding.DrawTagLinearGradient:
					info[di] = draw_flags
					p0_ := Vec2{
						math.Float32frombits(scene[dd+1]),
						math.Float32frombits(scene[dd+2]),
					}
					p1_ := Vec2{
						math.Float32frombits(scene[dd+3]),
						math.Float32frombits(scene[dd+4]),
					}
					p0 := transform.apply(p0_)
					p1 := transform.apply(p1_)
					dxy := p1.Sub(p0)
					scale := 1.0 / dxy.dot(dxy)
					line_xy := dxy.Mul(scale)
					line_c := -p0.dot(line_xy)
					info[di+1] = math.Float32bits(line_xy.x)
					info[di+2] = math.Float32bits(line_xy.y)
					info[di+3] = math.Float32bits(line_c)

				case encoding.DrawTagRadialGradient:
					const GRADIENT_EPSILON = 1.0 / (1 << 12)
					info[di] = draw_flags
					p0 := Vec2{
						math.Float32frombits(scene[dd+1]),
						math.Float32frombits(scene[dd+2]),
					}
					p1 := Vec2{
						math.Float32frombits(scene[dd+3]),
						math.Float32frombits(scene[dd+4]),
					}
					r0 := math.Float32frombits(scene[dd+5])
					r1 := math.Float32frombits(scene[dd+6])
					user_to_gradient := transform.inverse()
					var xform Transform
					var focal_x float32
					var radius float32
					var kind uint32
					var flags uint32
					if jmath.Abs32(r0-r1) < GRADIENT_EPSILON {
						// When the radii are the same, emit a strip gradient
						kind = RAD_GRAD_KIND_STRIP
						scaled := r0 / p0.distance(p1)
						xform = two_point_to_unit_line(p0, p1).Mul(user_to_gradient)
						radius = scaled * scaled
					} else {
						// Assume a two point conical gradient unless the centers
						// are equal.
						kind = RAD_GRAD_KIND_CONE
						if p0 == p1 {
							kind = RAD_GRAD_KIND_CIRCULAR
							// Nudge p0 a bit to avoid denormals.
							p0.x += GRADIENT_EPSILON
						}
						if r1 == 0.0 {
							// If r1 == 0.0, swap the points and radii
							flags |= RAD_GRAD_SWAPPED
							p0, p1 = p1, p0
						}
						focal_x = r0 / (r0 - r1)
						cf := p0.Mul(1.0 - focal_x).Add(p1.Mul(focal_x))
						radius = r1 / cf.distance(p1)
						user_to_unit_line :=
							two_point_to_unit_line(cf, p1).Mul(user_to_gradient)
						var user_to_scaled Transform
						// When r == 1.0, focal point is on circle
						if jmath.Abs32(radius-1.0) <= GRADIENT_EPSILON {
							kind = RAD_GRAD_KIND_FOCAL_ON_CIRCLE
							scale := 0.5 * jmath.Abs32(1.0-focal_x)
							user_to_scaled = Transform{scale, 0.0, 0.0, scale, 0.0, 0.0}.Mul(user_to_unit_line)
						} else {
							a := radius*radius - 1.0
							scale_ratio := jmath.Abs32(1.0-focal_x) / a
							scale_x := radius * scale_ratio
							scale_y := jmath.Sqrt32(jmath.Abs32(a)) * scale_ratio
							user_to_scaled = Transform{scale_x, 0.0, 0.0, scale_y, 0.0, 0.0}.Mul(user_to_unit_line)
						}
						xform = user_to_scaled
					}
					info[di+1] = math.Float32bits(xform[0])
					info[di+2] = math.Float32bits(xform[1])
					info[di+3] = math.Float32bits(xform[2])
					info[di+4] = math.Float32bits(xform[3])
					info[di+5] = math.Float32bits(xform[4])
					info[di+6] = math.Float32bits(xform[5])
					info[di+7] = math.Float32bits(focal_x)
					info[di+8] = math.Float32bits(radius)
					info[di+9] = (flags << 3) | kind

				case encoding.DrawTagSweepGradient:
					info[di] = draw_flags
					p0 := Vec2{
						math.Float32frombits(scene[dd+1]),
						math.Float32frombits(scene[dd+2]),
					}
					xform :=
						(transform.Mul(Transform{1.0, 0.0, 0.0, 1.0, p0.x, p0.y})).inverse()
					info[di+1] = math.Float32bits(xform[0])
					info[di+2] = math.Float32bits(xform[1])
					info[di+3] = math.Float32bits(xform[2])
					info[di+4] = math.Float32bits(xform[3])
					info[di+5] = math.Float32bits(xform[4])
					info[di+6] = math.Float32bits(xform[5])
					info[di+7] = scene[dd+3]
					info[di+8] = scene[dd+4]

				case encoding.DrawTagImage:
					info[di] = draw_flags
					xform := transform.inverse()
					info[di+1] = math.Float32bits(xform[0])
					info[di+2] = math.Float32bits(xform[1])
					info[di+3] = math.Float32bits(xform[2])
					info[di+4] = math.Float32bits(xform[3])
					info[di+5] = math.Float32bits(xform[4])
					info[di+6] = math.Float32bits(xform[5])
					info[di+7] = scene[dd]
					info[di+8] = scene[dd+1]

				case encoding.DrawTagBeginClip:
				default:
					panic(fmt.Sprintf("unhandled draw tag %v", tag_word))
				}
			}
			switch tag_word {
			case encoding.DrawTagBeginClip:
				path_ix := int32(m.PathIdx)
				clip_inp[m.ClipIdx] = renderer.Clip{Idx: ix, PathIdx: path_ix}
			case encoding.DrawTagEndClip:
				path_ix := int32(^ix)
				clip_inp[m.ClipIdx] = renderer.Clip{Idx: ix, PathIdx: path_ix}
			}
			m = m_next
		}
		prefix = prefix.Combine(reduced[i])
	}
}

func transformRead(transform_base uint32, ix uint32, data []uint32) Transform {
	z := make([]float32, 6)
	base := (transform_base + ix*6)
	for i := range uint32(6) {
		z[i] = math.Float32frombits(data[base+i])
	}
	return Transform(z)
}

func PathTilingSetup(_ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[0].(CPUBuffer))
	indirect := fromBytes[renderer.IndirectCount](resources[1].(CPUBuffer))
	segments := bump.SegCounts
	indirect.X = (segments + (WG_SIZE - 1)) / WG_SIZE
	indirect.Y = 1
	indirect.Z = 1
}
