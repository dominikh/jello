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
	"honnef.co/go/jello/mem"
	"honnef.co/go/jello/renderer"
	"honnef.co/go/safeish"
)

const wgSize = 256

const ptclInitialAlloc = 64

// Radial gradient kinds
const radGradKindCircular = 1
const radGradKindStrip = 2
const radGradKindFocalOnCircle = 3
const radGradKindCone = 4

// Radial gradient flags
const radGradSwapped = 1

const tileWidth = 16
const tileHeight = 16
const tileScale = 1.0 / 16.0
const numTileX = 16
const numTileY = 16
const numTile = numTileX * numTileY

// Keep in sync with config.wgsl
const blendStackSplit = 4

const ptclIncrement = 256
const ptclHeadroom = 2

// Tags for PTCL commands
const cmdEnd = 0
const cmdFill = 1

const cmdSolid = 3
const cmdColor = 5
const cmdLinGrad = 6
const cmdRadGrad = 7
const cmdSweepGrad = 8
const cmdImage = 9
const cmdBeginClip = 10
const cmdEndClip = 11
const cmdJump = 12

const pathTagSegType = 3
const pathTagPath = 0x10
const pathTagLineTo = 1
const pathTagQuadTo = 2
const pathTagCubicTo = 3
const pathTagF32 = 8

const drawInfoFlagsFillRuleBit = 1

// The largest floating point value strictly less than 1.
//
// This value is used to limit the value of b so that its floor is strictly less
// than 1. That guarantees that floor(a * i + b) == 0 for i == 0, which lands on
// the correct first tile.
const oneMinusULP = 0.99999994

// An epsilon to be applied in path numerical robustness.
//
// When floor(a * (n - 1) + b) does not match the expected value (the width in
// grid cells minus one), this delta is applied to a to push it in the correct
// direction. The theory is that a is not off by more than a few ulp, and it's
// always in the range of 0..1.
const robustEpsilon = 2e-7

func assert(b bool) {
	if !b {
		panic("failed assert")
	}
}

func span(a, b float32) uint32 {
	return uint32(max(jmath.Ceil32(max(a, b))-jmath.Floor32(min(a, b)), 1))
}

type transform [6]float32

var identity = transform{1, 0, 0, 1, 0, 0}

func (t transform) apply(p vec2) vec2 {
	z := t
	x := z[0]*p.x + z[2]*p.y + z[4]
	y := z[1]*p.x + z[3]*p.y + z[5]
	return vec2{x, y}
}

func (t transform) inverse() transform {
	z := t
	invDet := 1.0 / (z[0]*z[3] - z[1]*z[2])
	invMat := [4]float32{
		z[3] * invDet,
		-z[1] * invDet,
		-z[2] * invDet,
		z[0] * invDet,
	}
	return transform{
		invMat[0],
		invMat[1],
		invMat[2],
		invMat[3],
		-(invMat[0]*z[4] + invMat[2]*z[5]),
		-(invMat[1]*z[4] + invMat[3]*z[5]),
	}
}

func (t transform) Mul(other transform) transform {
	return transform{
		t[0]*other[0] + t[2]*other[1],
		t[1]*other[0] + t[3]*other[1],
		t[0]*other[2] + t[2]*other[3],
		t[1]*other[2] + t[3]*other[3],
		t[0]*other[4] + t[2]*other[5] + t[4],
		t[1]*other[4] + t[3]*other[5] + t[5],
	}
}

type vec2 struct {
	x, y float32
}

func (v vec2) add(o vec2) vec2 {
	return vec2{
		v.x + o.x,
		v.y + o.y,
	}
}

func (v vec2) sub(o vec2) vec2 {
	return vec2{
		v.x - o.x,
		v.y - o.y,
	}
}

func (v vec2) mul(f float32) vec2 {
	return vec2{
		v.x * f,
		v.y * f,
	}
}

func (v vec2) dot(other vec2) float32 {
	return v.x*other.x + v.y*other.y
}

func (v vec2) distance(other vec2) float32 {
	return v.sub(other).length()
}

func (v vec2) length() float32 {
	return jmath.Hypot32(v.x, v.y)
}

func (v vec2) lengthSquared() float32 {
	return v.dot(v)
}

func (v vec2) nan() bool {
	return math.IsNaN(float64(v.x)) || math.IsNaN(float64(v.y))
}

func (v vec2) mix(other vec2, t float32) vec2 {
	x := v.x + (other.x-v.x)*t
	y := v.y + (other.y-v.y)*t
	return vec2{x, y}
}

func (v vec2) normalize() vec2 {
	return vec2{
		v.x / v.length(),
		v.y / v.length(),
	}
}

func (v vec2) atan2() float32 {
	return jmath.Atan232(v.y, v.x)
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

func Backdrop(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[2].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[3].(CPUBuffer))

	for drawobjIdx := range config.Layout.NumDrawObjects {
		path := paths[drawobjIdx]
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

func BboxClear(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	pathBboxes := safeish.SliceCast[[]renderer.PathBbox](resources[1].(CPUBuffer))
	for i := range config.Layout.NumPaths {
		pathBboxes[i].X0 = 0x7fff_ffff
		pathBboxes[i].Y0 = 0x7fff_ffff
		pathBboxes[i].X1 = -0x8000_0000
		pathBboxes[i].Y1 = -0x8000_0000
	}
}

func PathTagReduce(_ *mem.Arena, numWgs uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.PathMonoid](resources[2].(CPUBuffer))

	pathTagBase := config.Layout.PathTagBase
	for i := range int(numWgs) {
		m := renderer.PathMonoid{}
		for j := range wgSize {
			tag := scene[(int(pathTagBase)+i*wgSize)+j]
			m = m.Combine(renderer.NewPathMonoid(tag))
		}
		reduced[i] = m
	}
}

func TileAlloc(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	const SX = 1.0 / (tileWidth)
	const SY = 1.0 / (tileHeight)

	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	drawBboxes := safeish.SliceCast[[][4]float32](resources[2].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[3].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[4].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[5].(CPUBuffer))

	drawTagBase := config.Layout.DrawTagBase
	widthInTiles := int32(config.WidthInTiles)
	heightInTiles := int32(config.HeightInTiles)
	for drawobjIdx := range config.Layout.NumDrawObjects {
		drawtag := encoding.DrawTag(scene[(drawTagBase + drawobjIdx)])
		var x0 int32
		var y0 int32
		var x1 int32
		var y1 int32
		if drawtag != encoding.DrawTagNop && drawtag != encoding.DrawTagEndClip {
			bbox := drawBboxes[drawobjIdx]
			if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
				x0 = int32(jmath.Floor32((bbox[0] * SX)))
				y0 = int32(jmath.Floor32((bbox[1] * SY)))
				x1 = int32(jmath.Ceil32((bbox[2] * SX)))
				y1 = int32(jmath.Ceil32((bbox[3] * SY)))
			}
		}
		ux0 := uint32(jmath.Clamp(widthInTiles, 0, x0))
		uy0 := uint32(jmath.Clamp(heightInTiles, 0, y0))
		ux1 := uint32(jmath.Clamp(widthInTiles, 0, x1))
		uy1 := uint32(jmath.Clamp(heightInTiles, 0, y1))
		tileCount := (ux1 - ux0) * (uy1 - uy0)
		offset := bump.Tile
		bump.Tile += tileCount
		// We construct it this way because padding is private.
		var path renderer.Path
		path.Bbox = [4]uint32{ux0, uy0, ux1, uy1}
		path.Tiles = offset
		paths[drawobjIdx] = path
		for i := range tileCount {
			tiles[(offset + i)] = renderer.Tile{}
		}
	}
}

func Binning(arena *mem.Arena, numWgs uint32, resources []CPUBinding) {
	const SX = 1.0 / (numTileX * tileWidth)
	const SY = 1.0 / (numTileY * tileHeight)

	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	drawMonoids := safeish.SliceCast[[]renderer.DrawMonoid](resources[1].(CPUBuffer))
	pathBboxBuf := safeish.SliceCast[[]renderer.PathBbox](resources[2].(CPUBuffer))
	clipBboxBuf := safeish.SliceCast[[][4]float32](resources[3].(CPUBuffer))
	intersectedBbox := safeish.SliceCast[[][4]float32](resources[4].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[5].(CPUBuffer))
	binData := safeish.SliceCast[[]uint32](resources[6].(CPUBuffer))
	binHeader := safeish.SliceCast[[]renderer.BinHeader](resources[7].(CPUBuffer))

	for wg := range numWgs {
		counts := mem.NewSlice[[]uint32](arena, wgSize, wgSize)
		bboxes := mem.NewSlice[[][4]int32](arena, wgSize, wgSize)
		widthInBins := int32((config.WidthInTiles + numTileX - 1) / numTileX)
		heightInBins := int32((config.HeightInTiles + numTileY - 1) / numTileY)
		for localIdx := range uint32(wgSize) {
			elementIdx := wg*wgSize + localIdx
			var x0 int32
			var y0 int32
			var x1 int32
			var y1 int32
			if elementIdx < config.Layout.NumDrawObjects {
				drawMonoid := drawMonoids[elementIdx]
				clipBbox := [4]float32{-1e9, -1e9, 1e9, 1e9}
				if drawMonoid.ClipIdx > 0 {
					if drawMonoid.ClipIdx-1 >= config.Layout.NumClips {
						panic("unreachable")
					}
					clipBbox = clipBboxBuf[drawMonoid.ClipIdx-1]
				}
				pathBbox := pathBboxBuf[drawMonoid.PathIdx]
				pb := [4]float32{
					float32(pathBbox.X0),
					float32(pathBbox.Y0),
					float32(pathBbox.X1),
					float32(pathBbox.Y1),
				}
				bbox := bboxIntersect(clipBbox, pb)
				intersectedBbox[elementIdx] = bbox
				if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
					x0 = int32(math.Floor(float64(bbox[0] * SX)))
					y0 = int32(math.Floor(float64(bbox[1] * SY)))
					x1 = int32(math.Ceil(float64(bbox[2] * SX)))
					y1 = int32(math.Ceil(float64(bbox[3] * SY)))
				}
			}
			x0 = jmath.Clamp(x0, 0, widthInBins)
			y0 = jmath.Clamp(y0, 0, heightInBins)
			x1 = jmath.Clamp(x1, 0, widthInBins)
			y1 = jmath.Clamp(y1, 0, heightInBins)
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					counts[y*widthInBins+x]++
				}
			}
			bboxes[localIdx] = [4]int32{x0, y0, x1, y1}
		}
		chunkOffset := mem.NewSlice[[]uint32](arena, wgSize, wgSize)
		for localIdx := range uint32(wgSize) {
			globalIdx := wg*wgSize + localIdx
			chunkOffset[localIdx] = bump.Binning
			bump.Binning += counts[localIdx]
			binHeader[globalIdx] = renderer.BinHeader{
				ElementCount: counts[localIdx],
				ChunkOffset:  chunkOffset[localIdx],
			}
		}
		for localIdx := range uint32(wgSize) {
			elementIdx := wg*wgSize + localIdx
			bbox := bboxes[localIdx]
			for y := bbox[1]; y < bbox[3]; y++ {
				for x := bbox[0]; x < bbox[2]; x++ {
					binIdx := (y*widthInBins + x)
					idx := config.Layout.BinDataStart + chunkOffset[binIdx]
					binData[idx] = elementIdx
					chunkOffset[binIdx]++
				}
			}
		}
	}
}

func bboxIntersect(a, b [4]float32) [4]float32 {
	return [4]float32{
		max(a[0], b[0]),
		max(a[1], b[1]),
		min(a[2], b[2]),
		min(a[3], b[3]),
	}
}

func ClipLeaf(arena *mem.Arena, _ uint32, resources []CPUBinding) {
	type clipStackElement struct {
		// index of draw object
		parentIdx uint32
		pathIdx   uint32
		bbox      [4]float32
	}

	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	clipInp := safeish.SliceCast[[]renderer.Clip](resources[1].(CPUBuffer))
	pathBboxes := safeish.SliceCast[[]renderer.PathBbox](resources[2].(CPUBuffer))
	drawMonoids := safeish.SliceCast[[]renderer.DrawMonoid](resources[5].(CPUBuffer))
	clipBboxes := safeish.SliceCast[[][4]float32](resources[6].(CPUBuffer))

	var stack []clipStackElement
	for globalIdx := range config.Layout.NumClips {
		clipEl := clipInp[globalIdx]
		if clipEl.PathIdx >= 0 {
			// begin clip
			pathIdx := uint32(clipEl.PathIdx)
			pathBbox := pathBboxes[pathIdx]
			pBbox := [4]float32{
				float32(pathBbox.X0),
				float32(pathBbox.Y0),
				float32(pathBbox.X1),
				float32(pathBbox.Y1),
			}
			var bbox [4]float32
			if len(stack) > 0 {
				last := stack[len(stack)-1]
				bbox = [4]float32{
					max(pBbox[0], last.bbox[0]),
					max(pBbox[1], last.bbox[1]),
					max(pBbox[2], last.bbox[2]),
					max(pBbox[3], last.bbox[3]),
				}
			} else {
				bbox = pBbox
			}
			clipBboxes[globalIdx] = bbox
			parentIdx := clipEl.Idx
			stack = mem.Append(arena, stack, clipStackElement{
				parentIdx,
				pathIdx,
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
			clipBboxes[globalIdx] = bbox
			drawMonoids[clipEl.Idx].PathIdx = tos.pathIdx
			drawMonoids[clipEl.Idx].SceneOffset =
				drawMonoids[tos.parentIdx].SceneOffset
		}
	}
}

func ClipReduce(arena *mem.Arena, numWgs uint32, resources []CPUBinding) {
	clipInp := safeish.SliceCast[[]renderer.Clip](resources[0].(CPUBuffer))
	pathBboxes := safeish.SliceCast[[]renderer.PathBbox](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.ClipBic](resources[2].(CPUBuffer))
	clipOut := safeish.SliceCast[[]renderer.ClipElement](resources[3].(CPUBuffer))

	scratch := mem.NewSlice[[]uint32](arena, 0, wgSize)
	for wgIdx := range numWgs {
		scratch = scratch[:0]
		var bicReduced renderer.ClipBic
		// reverse scan
		for localIdx := wgSize - 1; localIdx >= 0; localIdx-- {
			globalIdx := wgIdx*wgSize + uint32(localIdx)
			inp := clipInp[globalIdx].PathIdx
			var isPush uint32
			if inp >= 0 {
				isPush = 1
			}
			bic := renderer.ClipBic{A: 1 - isPush, B: isPush}
			if isPush != 0 && bicReduced.A == 0 {
				scratch = mem.Append(arena, scratch, globalIdx)
			}
			bicReduced = bic.Combine(bicReduced)
		}
		reduced[wgIdx] = bicReduced
		for i := len(scratch) - 1; i >= 0; i-- {
			parentIdx := scratch[i]
			var clipEl renderer.ClipElement
			clipEl.ParentIdx = parentIdx
			pathIdx := clipInp[parentIdx].PathIdx
			pathBbox := pathBboxes[pathIdx]
			clipEl.Bbox = [4]float32{
				float32(pathBbox.X0),
				float32(pathBbox.Y0),
				float32(pathBbox.X1),
				float32(pathBbox.Y1),
			}
			globalIdx := wgIdx*wgSize + uint32(i)
			clipOut[globalIdx] = clipEl
		}
	}
}

func DrawReduce(_ *mem.Arena, numWgs uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.DrawMonoid](resources[2].(CPUBuffer))

	numBlocksTotal := (config.Layout.NumDrawObjects + (wgSize - 1)) / wgSize
	numBlocksBase := numBlocksTotal / wgSize
	remainder := numBlocksTotal % wgSize
	for i := range numWgs {
		firstBlock := numBlocksBase*i + min(i, remainder)
		var b uint32
		if i < remainder {
			b = 1
		}
		numBlocks := numBlocksBase + b
		var m renderer.DrawMonoid
		for j := range wgSize * numBlocks {
			idx := (firstBlock * wgSize) + j
			tag := readDrawTagFromScene(config, scene, idx)
			m = m.Combine(renderer.NewDrawMonoid(encoding.DrawTag(tag)))
		}
		reduced[i] = m
	}
}

const drawTagNop = 0

// Read draw tag, guarded by number of draw objects.
//
// The idx argument is allowed to exceed the number of draw objects,
// in which case a NOP is returned.
func readDrawTagFromScene(config *renderer.ConfigUniform, scene []uint32, idx uint32) uint32 {
	if idx < config.Layout.NumDrawObjects {
		tagIdx := config.Layout.DrawTagBase + idx
		return scene[tagIdx]
	} else {
		return drawTagNop
	}
}

func PathCountSetup(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[0].(CPUBuffer))
	indirect := fromBytes[renderer.IndirectCount](resources[1].(CPUBuffer))

	lines := bump.Lines
	indirect.X = (lines + (wgSize - 1)) / wgSize
	indirect.Y = 1
	indirect.Z = 1
}

func PathTagScan(_ *mem.Arena, numWgs uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.PathMonoid](resources[2].(CPUBuffer))
	tagMonoids := safeish.SliceCast[[]renderer.PathMonoid](resources[3].(CPUBuffer))

	pathTagBase := config.Layout.PathTagBase
	var prefix renderer.PathMonoid
	for i := range uint32(numWgs) {
		m := prefix
		for j := range uint32(wgSize) {
			idx := (i * wgSize) + j
			tagMonoids[idx] = m
			tag := scene[pathTagBase+idx]
			m = m.Combine(renderer.NewPathMonoid(tag))
		}
		prefix = prefix.Combine(reduced[i])
	}
}

func PathTiling(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[0].(CPUBuffer))
	segCounts := safeish.SliceCast[[]renderer.SegmentCount](resources[1].(CPUBuffer))
	lines := safeish.SliceCast[[]renderer.LineSoup](resources[2].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[3].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[4].(CPUBuffer))
	segments := safeish.SliceCast[[]renderer.PathSegment](resources[5].(CPUBuffer))

	for segIdx := range bump.SegCounts {
		segCount := segCounts[segIdx]
		line := lines[segCount.LineIdx]
		counts := segCount.Counts
		segWithinSlice := counts >> 16
		segWithinLine := counts & 0xffff

		// coarse rasterization logic
		p0 := vec2{line.P0[0], line.P0[1]}
		p1 := vec2{line.P1[0], line.P1[1]}
		isDown := p1.y >= p0.y
		var xy0, xy1 vec2
		if isDown {
			xy0, xy1 = p0, p1
		} else {
			xy0, xy1 = p1, p0
		}
		s0 := xy0.mul(tileScale)
		s1 := xy1.mul(tileScale)
		countX := span(s0.x, s1.x) - 1
		count := countX + span(s0.y, s1.y)

		dx := jmath.Abs32(s1.x - s0.x)
		dy := s1.y - s0.y
		idxdy := 1.0 / (dx + dy)
		a := dx * idxdy
		isPositiveSlope := s1.x >= s0.x
		var sign float32
		if isPositiveSlope {
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
		b := min(((dy*c + dx*(ytop-s0.y)) * idxdy), oneMinusULP)
		robustError := jmath.Floor32(a*(float32(count)-1.0)+b) - float32(countX)
		if robustError != 0.0 {
			a -= jmath.Copysign32(robustEpsilon, robustError)
		}
		x0 := xt0 * sign
		if isPositiveSlope {
			x0 += 0.0
		} else {
			x0 += -1.0
		}
		z := jmath.Floor32(a*float32(segWithinLine) + b)
		x := int32(x0) + int32(sign*z)
		y := int32(y0 + float32(segWithinLine) - z)

		path := paths[line.PathIdx]
		bboxu := path.Bbox
		bbox := [4]int32{
			int32(bboxu[0]),
			int32(bboxu[1]),
			int32(bboxu[2]),
			int32(bboxu[3]),
		}
		stride := bbox[2] - bbox[0]
		tileIdx := int32(path.Tiles) + (y-bbox[1])*stride + x - bbox[0]
		tile := tiles[tileIdx]
		segStart := ^tile.SegmentCountOrIdx
		if (int32(segStart)) < 0 {
			continue
		}
		tileXy := vec2{float32(x) * tileWidth, float32(y) * tileHeight}
		tileXy1 := tileXy.add(vec2{tileWidth, tileHeight})

		if segWithinLine > 0 {
			zPrev := jmath.Floor32(a*(float32(segWithinLine)-1.0) + b)
			if z == zPrev {
				// Top edge is clipped
				xt := xy0.x + (xy1.x-xy0.x)*(tileXy.y-xy0.y)/(xy1.y-xy0.y)
				xt = jmath.Clamp(xt, tileXy.x+1e-3, tileXy1.x)
				xy0 = vec2{xt, tileXy.y}
			} else {
				// If isPositiveSlope, left edge is clipped, otherwise right
				var xClip float32
				if isPositiveSlope {
					xClip = tileXy.x
				} else {
					xClip = tileXy1.x
				}
				yt := xy0.y + (xy1.y-xy0.y)*(xClip-xy0.x)/(xy1.x-xy0.x)
				yt = jmath.Clamp(yt, tileXy.y+1e-3, tileXy1.y)
				xy0 = vec2{xClip, yt}
			}
		}
		if segWithinLine < count-1 {
			zNext := jmath.Floor32(a*(float32(segWithinLine)+1.0) + b)
			if z == zNext {
				// Bottom edge is clipped
				xt := xy0.x + (xy1.x-xy0.x)*(tileXy1.y-xy0.y)/(xy1.y-xy0.y)
				xt = jmath.Clamp(xt, tileXy.x+1e-3, tileXy1.x)
				xy1 = vec2{xt, tileXy1.y}
			} else {
				// If isPositiveSlope, right edge is clipped, otherwise left
				var xClip float32
				if isPositiveSlope {
					xClip = tileXy1.x
				} else {
					xClip = tileXy.x
				}
				yt := xy0.y + (xy1.y-xy0.y)*(xClip-xy0.x)/(xy1.x-xy0.x)
				yt = jmath.Clamp(yt, tileXy.y+1e-3, tileXy1.y)
				xy1 = vec2{xClip, yt}
			}
		}
		yEdge := float32(1e9)
		// Apply numerical robustness logic
		p0 = xy0.sub(tileXy)
		p1 = xy1.sub(tileXy)
		const EPSILON = 1e-6
		if p0.x == 0.0 {
			if p1.x == 0.0 {
				p0.x = EPSILON
				if p0.y == 0.0 {
					// Entire tile
					p1.x = EPSILON
					p1.y = tileHeight
				} else {
					// Make segment disappear
					p1.x = 2.0 * EPSILON
					p1.y = p0.y
				}
			} else if p0.y == 0.0 {
				p0.x = EPSILON
			} else {
				yEdge = p0.y
			}
		} else if p1.x == 0.0 {
			if p1.y == 0.0 {
				p1.x = EPSILON
			} else {
				yEdge = p1.y
			}
		}
		if p0.x == jmath.Floor32(p0.x) && p0.x != 0.0 {
			p0.x -= EPSILON
		}
		if p1.x == jmath.Floor32(p1.x) && p1.x != 0.0 {
			p1.x -= EPSILON
		}
		if !isDown {
			p0, p1 = p1, p0
		}
		segment := renderer.PathSegment{
			Point0: [2]float32{p0.x, p0.y},
			Point1: [2]float32{p1.x, p1.y},
			YEdge:  yEdge,
		}
		assert(p0.x >= 0.0 && p0.x <= tileWidth)
		assert(p0.y >= 0.0 && p0.y <= tileHeight)
		assert(p1.x >= 0.0 && p1.x <= tileWidth)
		assert(p1.y >= 0.0 && p1.y <= tileHeight)
		segments[(segStart + segWithinSlice)] = segment
	}
}

func PathCount(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[1].(CPUBuffer))
	lines := safeish.SliceCast[[]renderer.LineSoup](resources[2].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[3].(CPUBuffer))
	tile := safeish.SliceCast[[]renderer.Tile](resources[4].(CPUBuffer))
	segCounts := safeish.SliceCast[[]renderer.SegmentCount](resources[5].(CPUBuffer))

	for lineIdx := range bump.Lines {
		line := lines[lineIdx]
		p0 := vec2{line.P0[0], line.P0[1]}
		p1 := vec2{line.P1[0], line.P1[1]}
		isDown := p1.y >= p0.y
		var xy0, xy1 vec2
		if isDown {
			xy0, xy1 = p0, p1
		} else {
			xy0, xy1 = p1, p0
		}
		s0 := xy0.mul(tileScale)
		s1 := xy1.mul(tileScale)
		countX := span(s0.x, s1.x) - 1
		count := countX + span(s0.y, s1.y)

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
		isPositiveSlope := s1.x >= s0.x
		var sign float32
		if isPositiveSlope {
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
		b := min(((dy*c + dx*(ytop-s0.y)) * idxdy), oneMinusULP)
		robustError := jmath.Floor32(a*(float32(count)-1.0)+b) - float32(countX)
		if robustError != 0.0 {
			a -= jmath.Copysign32(robustEpsilon, robustError)
		}
		x0 := xt0 * sign
		if isPositiveSlope {
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
		if isDown {
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
			if isPositiveSlope {
				fudge = 0.0
			} else {
				fudge = 1.0
			}
			if xmin < float32(bbox[0]) {
				f := jmath.Round32((sign*(float32(bbox[0])-x0) - b + fudge) / a)
				if (x0+sign*jmath.Floor32(a*f+b) < float32(bbox[0])) == isPositiveSlope {
					f += 1.0
				}
				ynext := int32(y0 + f - jmath.Floor32(a*f+b) + 1.0)
				if isPositiveSlope {
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
				if (x0+sign*jmath.Floor32(a*f+b) < float32(bbox[2])) == isPositiveSlope {
					f += 1.0
				}
				if isPositiveSlope {
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
		lastZ := jmath.Floor32(a*(float32(imin)-1.0) + b)
		segBase := bump.SegCounts
		bump.SegCounts += imax - imin
		for i := imin; i < imax; i++ {
			zf := a*float32(i) + b
			z := jmath.Floor32(zf)
			y := int32(y0 + float32(i) - z)
			x := int32(x0 + sign*z)
			base := int32(path.Tiles) + (y-bbox[1])*stride - bbox[0]
			var topEdge bool
			if i == 0 {
				topEdge = y0 == s0.y
			} else {
				topEdge = lastZ == z
			}
			if topEdge && x+1 < bbox[2] {
				xBump := max((x + 1), bbox[0])
				tile[(base + xBump)].Backdrop += delta
			}
			// .segments is another name for the .count field; it's overloaded
			segWithinSlice := tile[(base + x)].SegmentCountOrIdx
			tile[(base + x)].SegmentCountOrIdx += 1
			counts := (segWithinSlice << 16) | i
			segCount := renderer.SegmentCount{LineIdx: lineIdx, Counts: counts}
			segCounts[(segBase + i - imin)] = segCount
			lastZ = z
		}
	}
}

type tileState struct {
	cmdOffset uint32
	cmdLimit  uint32
}

func newTileState(tileIdx uint32) tileState {
	cmdOffset := tileIdx * ptclInitialAlloc
	cmdLimit := cmdOffset + (ptclInitialAlloc - ptclHeadroom)
	return tileState{
		cmdOffset,
		cmdLimit,
	}
}

func (ts *tileState) allocCmd(
	size uint32,
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
) {
	if ts.cmdOffset+size >= ts.cmdLimit {
		ptclDynStart := config.WidthInTiles * config.HeightInTiles * ptclInitialAlloc
		chunkSize := max(ptclIncrement, size+ptclHeadroom)
		newCmd := ptclDynStart + bump.Ptcl
		bump.Ptcl += chunkSize
		ptcl[ts.cmdOffset] = cmdJump
		ptcl[ts.cmdOffset+1] = newCmd
		ts.cmdOffset = newCmd
		ts.cmdLimit = newCmd + (ptclIncrement - ptclHeadroom)
	}
}

func (ts *tileState) write(ptcl []uint32, offset uint32, value uint32) {
	ptcl[ts.cmdOffset+offset] = value
}

func (ts *tileState) writePath(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	tile *renderer.Tile,
	drawFlags uint32,
) {
	numSegs := tile.SegmentCountOrIdx
	if numSegs != 0 {
		segIdx := bump.Segments
		tile.SegmentCountOrIdx = ^segIdx
		bump.Segments += numSegs
		ts.allocCmd(4, config, bump, ptcl)
		ts.write(ptcl, 0, cmdFill)
		var evenOdd uint32
		if (drawFlags & drawInfoFlagsFillRuleBit) != 0 {
			evenOdd = 1
		}
		sizeAndRule := (numSegs << 1) | evenOdd
		ts.write(ptcl, 1, sizeAndRule)
		ts.write(ptcl, 2, segIdx)
		ts.write(ptcl, 3, uint32(tile.Backdrop))
		ts.cmdOffset += 4
	} else {
		ts.allocCmd(1, config, bump, ptcl)
		ts.write(ptcl, 0, cmdSolid)
		ts.cmdOffset += 1
	}
}

func (ts *tileState) writeColor(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	rgbaColor [4]float32,
) {
	ts.allocCmd(5, config, bump, ptcl)
	ts.write(ptcl, 0, cmdColor)
	ts.write(ptcl, 1, math.Float32bits(rgbaColor[0]))
	ts.write(ptcl, 2, math.Float32bits(rgbaColor[1]))
	ts.write(ptcl, 3, math.Float32bits(rgbaColor[2]))
	ts.write(ptcl, 4, math.Float32bits(rgbaColor[3]))
	ts.cmdOffset += 5
}

func (ts *tileState) writeImage(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	infoOffset uint32,
) {
	ts.allocCmd(2, config, bump, ptcl)
	ts.write(ptcl, 0, cmdImage)
	ts.write(ptcl, 1, infoOffset)
	ts.cmdOffset += 2
}

func (ts *tileState) writeGrad(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	ty uint32,
	index uint32,
	infoOffset uint32,
) {
	ts.allocCmd(3, config, bump, ptcl)
	ts.write(ptcl, 0, ty)
	ts.write(ptcl, 1, index)
	ts.write(ptcl, 2, infoOffset)
	ts.cmdOffset += 3
}

func (ts *tileState) writeBeginClip(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
) {
	ts.allocCmd(1, config, bump, ptcl)
	ts.write(ptcl, 0, cmdBeginClip)
	ts.cmdOffset += 1
}

func (ts *tileState) writeEndClip(
	config *renderer.ConfigUniform,
	bump *renderer.BumpAllocators,
	ptcl []uint32,
	blend uint32,
	alpha float32,
) {
	ts.allocCmd(3, config, bump, ptcl)
	ts.write(ptcl, 0, cmdEndClip)
	ts.write(ptcl, 1, blend)
	ts.write(ptcl, 2, math.Float32bits(alpha))
	ts.cmdOffset += 3
}

func Coarse(arena *mem.Arena, _ uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	drawMonoids := safeish.SliceCast[[]renderer.DrawMonoid](resources[2].(CPUBuffer))
	binHeaders := safeish.SliceCast[[]renderer.BinHeader](resources[3].(CPUBuffer))
	infoBinData := safeish.SliceCast[[]uint32](resources[4].(CPUBuffer))
	paths := safeish.SliceCast[[]renderer.Path](resources[5].(CPUBuffer))
	tiles := safeish.SliceCast[[]renderer.Tile](resources[6].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[7].(CPUBuffer))
	ptcl := safeish.SliceCast[[]uint32](resources[8].(CPUBuffer))

	widthInTiles := config.WidthInTiles
	heightInTiles := config.HeightInTiles
	widthInBins := (widthInTiles + numTileX - 1) / numTileX
	heightInBins := (heightInTiles + numTileY - 1) / numTileY
	numBins := widthInBins * heightInBins
	binDataStart := config.Layout.BinDataStart
	drawTagBase := config.Layout.DrawTagBase
	compacted := mem.NewSlice[[][]uint32](arena, numTile, numTile)
	numPartitions := (config.Layout.NumDrawObjects + numTile - 1) / numTile
	for bin := range numBins {
		for i := range compacted {
			compacted[i] = compacted[i][:0]
		}
		binX := bin % widthInBins
		binY := bin / widthInBins
		binTileX := numTileX * binX
		binTileY := numTileY * binY
		for part := range numPartitions {
			inIdx := part*numTile + bin
			binHeader := binHeaders[inIdx]
			start := binDataStart + binHeader.ChunkOffset
			for i := range binHeader.ElementCount {
				drawobjIdx := infoBinData[start+i]
				tag := scene[drawTagBase+drawobjIdx]
				if encoding.DrawTag(tag) != encoding.DrawTagNop {
					drawMonoid := drawMonoids[drawobjIdx]
					pathIdx := drawMonoid.PathIdx
					path := paths[pathIdx]
					dx := int32(path.Bbox[0]) - int32(binTileX)
					dy := int32(path.Bbox[1]) - int32(binTileY)
					x0 := jmath.Clamp(dx, 0, numTileX)
					y0 := jmath.Clamp(dy, 0, numTileY)
					x1 := jmath.Clamp(int32(path.Bbox[2])-int32(binTileX), 0, numTileX)
					y1 := jmath.Clamp(int32(path.Bbox[3])-int32(binTileY), 0, numTileY)
					for y := y0; y < y1; y++ {
						for x := x0; x < x1; x++ {
							compacted[(y*numTileX + x)] = mem.Append(arena, compacted[(y*numTileX+x)], drawobjIdx)
						}
					}
				}
			}
		}
		// compacted now has the list of draw objects for each tile.
		// While the WGSL source does at most 256 draw objects at a time,
		// this version does all the draw objects in a tile.
		for tileIdx := range numTile {
			tileX := uint32(tileIdx % numTileX)
			tileY := uint32(tileIdx / numTileX)
			thisTileIdx := (binTileY+tileY)*widthInTiles + binTileX + tileX
			tileState := newTileState(thisTileIdx)
			blendOffset := tileState.cmdOffset
			tileState.cmdOffset += 1
			clipDepth := 0
			renderBlendDepth := 0
			maxBlendDepth := 0
			clipZeroDepth := 0
			for _, drawobjIdx := range compacted[tileIdx] {
				drawtag := scene[(drawTagBase + drawobjIdx)]
				if clipZeroDepth == 0 {
					drawMonoid := drawMonoids[drawobjIdx]
					pathIdx := drawMonoid.PathIdx
					path := paths[pathIdx]
					bbox := path.Bbox
					stride := bbox[2] - bbox[0]
					x := binTileX + tileX - bbox[0]
					y := binTileY + tileY - bbox[1]
					tile := &tiles[path.Tiles+y*stride+x]
					isClip := (drawtag & 1) != 0
					isBlend := false
					dd := config.Layout.DrawDataBase + drawMonoid.SceneOffset
					di := drawMonoid.InfoOffset
					if isClip {
						const blendClip = (128 << 8) | 3
						blend := scene[dd]
						isBlend = blend != blendClip
					}

					drawFlags := infoBinData[di]
					evenOdd := (drawFlags & drawInfoFlagsFillRuleBit) != 0
					numSegs := tile.SegmentCountOrIdx

					// If this draw object represents an even-odd fill and we
					// know that no line segment crosses this tile and then this
					// draw object should not contribute to the tile if its
					// backdrop (i.e. the winding number of its top-left corner)
					// is even.
					backdropClear := (evenOdd && jmath.AbsInt32(tile.Backdrop)&1 == 0) || (!evenOdd && tile.Backdrop == 0)
					includeTile := numSegs != 0 || (backdropClear == isClip) || isBlend
					if includeTile {
						switch encoding.DrawTag(drawtag) {
						case encoding.DrawTagColor:
							tileState.writePath(config, bump, ptcl, tile, drawFlags)
							r := math.Float32frombits(scene[dd])
							g := math.Float32frombits(scene[dd+1])
							b := math.Float32frombits(scene[dd+2])
							a := math.Float32frombits(scene[dd+3])
							tileState.writeColor(config, bump, ptcl, [4]float32{r, g, b, a})

						case encoding.DrawTagImage:
							tileState.writePath(config, bump, ptcl, tile, drawFlags)
							tileState.writeImage(config, bump, ptcl, di+1)

						case encoding.DrawTagLinearGradient:
							tileState.writePath(config, bump, ptcl, tile, drawFlags)
							index := scene[dd]
							tileState.writeGrad(
								config,
								bump,
								ptcl,
								cmdLinGrad,
								index,
								di+1,
							)

						case encoding.DrawTagRadialGradient:
							tileState.writePath(config, bump, ptcl, tile, drawFlags)
							index := scene[dd]
							tileState.writeGrad(
								config,
								bump,
								ptcl,
								cmdRadGrad,
								index,
								di+1,
							)

						case encoding.DrawTagSweepGradient:
							tileState.writePath(config, bump, ptcl, tile, drawFlags)
							index := scene[dd]
							tileState.writeGrad(
								config,
								bump,
								ptcl,
								cmdSweepGrad,
								index,
								di+1,
							)

						case encoding.DrawTagBeginClip:
							if tile.SegmentCountOrIdx == 0 && tile.Backdrop == 0 {
								clipZeroDepth = clipDepth + 1
							} else {
								tileState.writeBeginClip(config, bump, ptcl)
								// TODO: Do we need to track this separately, seems like it
								// is always the same as clip_depth in this code path
								renderBlendDepth++
								maxBlendDepth = max(renderBlendDepth, maxBlendDepth)
							}
							clipDepth++

						case encoding.DrawTagEndClip:
							clipDepth--
							// A clip shape is always a non-zero fill (drawFlags=0).
							tileState.writePath(config, bump, ptcl, tile, 0)
							blend := scene[dd]
							alpha := math.Float32frombits(scene[dd+1])
							tileState.writeEndClip(config, bump, ptcl, blend, alpha)
							renderBlendDepth--

						default:
							panic("unreachable")
						}
					}
				} else {
					// In "clip zero" state, suppress all drawing
					switch encoding.DrawTag(drawtag) {
					case encoding.DrawTagBeginClip:
						clipDepth++
					case encoding.DrawTagEndClip:
						if clipDepth == clipZeroDepth {
							clipZeroDepth = 0
						}
						clipDepth--
					}
				}
			}

			if binTileX+tileX < widthInTiles && binTileY+tileY < heightInTiles {
				ptcl[tileState.cmdOffset] = cmdEnd
				scratchSize := uint32(max((maxBlendDepth-blendStackSplit), 0) * tileWidth * tileHeight)
				ptcl[blendOffset] = bump.Blend
				bump.Blend += scratchSize
			}
		}
	}
}

func twoPointToUnitLine(p0 vec2, p1 vec2) transform {
	tmp1 := fromPoly2(p0, p1)
	inv := tmp1.inverse()
	tmp2 := fromPoly2(vec2{}, vec2{1.0, 0.0})
	return tmp2.Mul(inv)
}

func fromPoly2(p0 vec2, p1 vec2) transform {
	return transform{
		p1.y - p0.y,
		p0.x - p1.x,
		p1.x - p0.x,
		p1.y - p0.y,
		p0.x,
		p0.y}
}

func DrawLeaf(_ *mem.Arena, numWgs uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	reduced := safeish.SliceCast[[]renderer.DrawMonoid](resources[2].(CPUBuffer))
	pathBbox := safeish.SliceCast[[]renderer.PathBbox](resources[3].(CPUBuffer))
	drawMonoid := safeish.SliceCast[[]renderer.DrawMonoid](resources[4].(CPUBuffer))
	info := safeish.SliceCast[[]uint32](resources[5].(CPUBuffer))
	clipInp := safeish.SliceCast[[]renderer.Clip](resources[6].(CPUBuffer))

	numBlocksTotal := (config.Layout.NumDrawObjects + (wgSize - 1)) / wgSize
	numBlocksBase := numBlocksTotal / wgSize
	remainder := numBlocksTotal % wgSize
	prefix := renderer.DrawMonoid{}
	for i := range numWgs {
		firstBlock := numBlocksBase*i + min(i, remainder)
		numBlocks := numBlocksBase
		if i < remainder {
			numBlocks++
		}
		m := prefix
		for j := range wgSize * numBlocks {
			idx := uint32(firstBlock*wgSize) + uint32(j)
			tagRaw := readDrawTagFromScene(config, scene, idx)
			tagWord := encoding.DrawTag(tagRaw)
			// store exclusive prefix sum
			if idx < config.Layout.NumDrawObjects {
				drawMonoid[idx] = m
			}
			mNext := m.Combine(renderer.NewDrawMonoid(tagWord))
			dd := config.Layout.DrawDataBase + m.SceneOffset
			di := m.InfoOffset
			if tagWord == encoding.DrawTagColor ||
				tagWord == encoding.DrawTagLinearGradient ||
				tagWord == encoding.DrawTagRadialGradient ||
				tagWord == encoding.DrawTagSweepGradient ||
				tagWord == encoding.DrawTagImage ||
				tagWord == encoding.DrawTagBeginClip {
				bbox := pathBbox[m.PathIdx]
				trans := transformRead(config.Layout.TransformBase, bbox.TransIdx, scene)
				drawFlags := bbox.DrawFlags
				switch tagWord {
				case encoding.DrawTagColor:
					info[di] = drawFlags

				case encoding.DrawTagLinearGradient:
					info[di] = drawFlags
					p0_ := vec2{
						math.Float32frombits(scene[dd+1]),
						math.Float32frombits(scene[dd+2]),
					}
					p1_ := vec2{
						math.Float32frombits(scene[dd+3]),
						math.Float32frombits(scene[dd+4]),
					}
					p0 := trans.apply(p0_)
					p1 := trans.apply(p1_)
					dxy := p1.sub(p0)
					scale := 1.0 / dxy.dot(dxy)
					lineXY := dxy.mul(scale)
					lineC := -p0.dot(lineXY)
					info[di+1] = math.Float32bits(lineXY.x)
					info[di+2] = math.Float32bits(lineXY.y)
					info[di+3] = math.Float32bits(lineC)

				case encoding.DrawTagRadialGradient:
					const gradientEpsilon = 1.0 / (1 << 12)
					info[di] = drawFlags
					p0 := vec2{
						math.Float32frombits(scene[dd+1]),
						math.Float32frombits(scene[dd+2]),
					}
					p1 := vec2{
						math.Float32frombits(scene[dd+3]),
						math.Float32frombits(scene[dd+4]),
					}
					r0 := math.Float32frombits(scene[dd+5])
					r1 := math.Float32frombits(scene[dd+6])
					userToGradient := trans.inverse()
					var xform transform
					var focalX float32
					var radius float32
					var kind uint32
					var flags uint32
					if jmath.Abs32(r0-r1) < gradientEpsilon {
						// When the radii are the same, emit a strip gradient
						kind = radGradKindStrip
						scaled := r0 / p0.distance(p1)
						xform = twoPointToUnitLine(p0, p1).Mul(userToGradient)
						radius = scaled * scaled
					} else {
						// Assume a two point conical gradient unless the centers
						// are equal.
						kind = radGradKindCone
						if p0 == p1 {
							kind = radGradKindCircular
							// Nudge p0 a bit to avoid denormals.
							p0.x += gradientEpsilon
						}
						if r1 == 0.0 {
							// If r1 == 0.0, swap the points and radii
							flags |= radGradSwapped
							p0, p1 = p1, p0
						}
						focalX = r0 / (r0 - r1)
						cf := p0.mul(1.0 - focalX).add(p1.mul(focalX))
						radius = r1 / cf.distance(p1)
						userToUnitLine :=
							twoPointToUnitLine(cf, p1).Mul(userToGradient)
						var userToScaled transform
						// When r == 1.0, focal point is on circle
						if jmath.Abs32(radius-1.0) <= gradientEpsilon {
							kind = radGradKindFocalOnCircle
							scale := 0.5 * jmath.Abs32(1.0-focalX)
							userToScaled = transform{scale, 0.0, 0.0, scale, 0.0, 0.0}.Mul(userToUnitLine)
						} else {
							a := radius*radius - 1.0
							scaleRatio := jmath.Abs32(1.0-focalX) / a
							scaleX := radius * scaleRatio
							scaleY := jmath.Sqrt32(jmath.Abs32(a)) * scaleRatio
							userToScaled = transform{scaleX, 0.0, 0.0, scaleY, 0.0, 0.0}.Mul(userToUnitLine)
						}
						xform = userToScaled
					}
					info[di+1] = math.Float32bits(xform[0])
					info[di+2] = math.Float32bits(xform[1])
					info[di+3] = math.Float32bits(xform[2])
					info[di+4] = math.Float32bits(xform[3])
					info[di+5] = math.Float32bits(xform[4])
					info[di+6] = math.Float32bits(xform[5])
					info[di+7] = math.Float32bits(focalX)
					info[di+8] = math.Float32bits(radius)
					info[di+9] = (flags << 3) | kind

				case encoding.DrawTagSweepGradient:
					info[di] = drawFlags
					p0 := vec2{
						math.Float32frombits(scene[dd+1]),
						math.Float32frombits(scene[dd+2]),
					}
					xform :=
						(trans.Mul(transform{1.0, 0.0, 0.0, 1.0, p0.x, p0.y})).inverse()
					info[di+1] = math.Float32bits(xform[0])
					info[di+2] = math.Float32bits(xform[1])
					info[di+3] = math.Float32bits(xform[2])
					info[di+4] = math.Float32bits(xform[3])
					info[di+5] = math.Float32bits(xform[4])
					info[di+6] = math.Float32bits(xform[5])
					info[di+7] = scene[dd+3]
					info[di+8] = scene[dd+4]

				case encoding.DrawTagImage:
					info[di] = drawFlags
					xform := trans.inverse()
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
					panic(fmt.Sprintf("unhandled draw tag %v", tagWord))
				}
			}
			switch tagWord {
			case encoding.DrawTagBeginClip:
				pathIdx := int32(m.PathIdx)
				clipInp[m.ClipIdx] = renderer.Clip{Idx: idx, PathIdx: pathIdx}
			case encoding.DrawTagEndClip:
				pathIdx := int32(^idx)
				clipInp[m.ClipIdx] = renderer.Clip{Idx: idx, PathIdx: pathIdx}
			}
			m = mNext
		}
		prefix = prefix.Combine(reduced[i])
	}
}

func transformRead(transformBase uint32, idx uint32, data []uint32) transform {
	var z [6]float32
	base := (transformBase + idx*6)
	for i := range uint32(6) {
		z[i] = math.Float32frombits(data[base+i])
	}
	return transform(z)
}

func PathTilingSetup(_ *mem.Arena, _ uint32, resources []CPUBinding) {
	bump := fromBytes[renderer.BumpAllocators](resources[0].(CPUBuffer))
	indirect := fromBytes[renderer.IndirectCount](resources[1].(CPUBuffer))
	segments := bump.SegCounts
	indirect.X = (segments + (wgSize - 1)) / wgSize
	indirect.Y = 1
	indirect.Z = 1
}
