// Copyright 2023 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package renderer

import (
	"structs"
	"unsafe"

	"golang.org/x/exp/constraints"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/jello/mem"
)

type WorkgroupSize [3]uint32

// ConfigUniform contains uniform render configuration data used by all GPU stages.
//
// This data structure must be kept in sync with the definition in
// `shaders/shared/config.wgsl`.
type ConfigUniform struct {
	_ structs.HostLayout

	// Width of the scene in tiles.
	WidthInTiles uint32
	// Height of the scene in tiles.
	HeightInTiles uint32
	// Width of the target in pixels.
	TargetWidth uint32
	// Height of the target in pixels.
	TargetHeight uint32
	// The base background color applied to the target before any blends.
	BaseColor uint32
	// Layout of packed scene data.
	Layout Layout
	// Size of line soup buffer allocation (in [`LineSoup`]s)
	LinesSize uint32
	// Size of binning buffer allocation (in `uint32`s).
	BinningSize uint32
	// Size of tile buffer allocation (in [`Tile`]s).
	TilesSize uint32
	// Size of segment count buffer allocation (in [`SegmentCount`]s).
	SegCountsSize uint32
	// Size of segment buffer allocation (in [`PathSegment`]s).
	SegmentsSize uint32
	// Size of per-tile command list buffer allocation (in `uint32`s).
	PtclSize uint32
}

type Layout struct {
	_ structs.HostLayout

	// Number of draw objects.
	NumDrawObjects uint32
	// Number of paths.
	NumPaths uint32
	// Number of clips.
	NumClips uint32
	// Start of binning data.
	BinDataStart uint32
	// Start of path tag stream.
	PathTagBase uint32
	// Start of path data stream.
	PathDataBase uint32
	// Start of draw tag stream.
	DrawTagBase uint32
	// Start of draw data stream.
	DrawDataBase uint32
	// Start of transform stream.
	TransformBase uint32
	// Start of style stream.
	StyleBase uint32
}

func (l *Layout) pathTagsSize() uint32 {
	start := l.PathTagBase * 4
	end := l.PathDataBase * 4
	return end - start
}

type RenderConfig struct {
	gpu             ConfigUniform
	workgroupCounts WorkgroupCounts
	bufferSizes     BufferSizes
}

func NewRenderConfig(arena *mem.Arena, layout *Layout, width, height uint32, baseColor gfx.Color) *RenderConfig {
	newWidth := nextMultipleOf(width, tileWidth)
	newHeight := nextMultipleOf(height, tileHeight)
	widthInTiles := newWidth / tileWidth
	heightInTiles := newHeight / tileHeight
	numPathTags := layout.pathTagsSize()
	workgroupCounts := NewWorkgroupCounts(layout, widthInTiles, heightInTiles, numPathTags)
	bufferSizes := NewBufferSizes(layout, &workgroupCounts)
	out := mem.New[RenderConfig](arena)
	*out = RenderConfig{
		gpu: ConfigUniform{
			WidthInTiles:  widthInTiles,
			HeightInTiles: heightInTiles,
			TargetWidth:   width,
			TargetHeight:  height,
			BaseColor:     baseColor.LinearSRGB().PremulUint32(),
			LinesSize:     uint32(bufferSizes.Lines),
			BinningSize:   uint32(bufferSizes.BinData) - layout.BinDataStart,
			TilesSize:     uint32(bufferSizes.Tiles),
			SegCountsSize: uint32(bufferSizes.SegCounts),
			SegmentsSize:  uint32(bufferSizes.Segments),
			PtclSize:      uint32(bufferSizes.Ptcl),
			Layout:        *layout,
		},
		workgroupCounts: workgroupCounts,
		bufferSizes:     bufferSizes,
	}
	return out
}

func NewBufferSizes(layout *Layout, workgroups *WorkgroupCounts) BufferSizes {
	numPaths := layout.NumPaths
	numDrawObjects := layout.NumDrawObjects
	numClips := layout.NumClips
	pathTagWgs := workgroups.PathReduce[0]
	var reducedSize uint32
	if workgroups.UseLargePathScan {
		reducedSize = jmath.AlignUp(pathTagWgs, pathReduceWg)
	} else {
		reducedSize = pathTagWgs
	}
	drawMonoidWgs := workgroups.DrawReduce[0]

	binningWgs := workgroups.Binning[0]
	numPathsALigned := jmath.AlignUp(numPaths, 256)

	// The following buffer sizes have been hand picked to accommodate the vello test scenes as
	// well as paris-30k. These should instead get derived from the scene layout using
	// reasonable heuristics.
	binData := NewBufferSize[uint32](1 << 18)
	tiles := NewBufferSize[Tile](1 << 21)
	lines := NewBufferSize[LineSoup](1 << 21)
	segCounts := NewBufferSize[SegmentCount](1 << 21)
	segments := NewBufferSize[PathSegment](1 << 21)
	ptcl := NewBufferSize[uint32](1 << 23)
	return BufferSizes{
		PathReduced:     NewBufferSize[PathMonoid](reducedSize),
		PathReduced2:    NewBufferSize[PathMonoid](pathReduceWg),
		PathReducedScan: NewBufferSize[PathMonoid](reducedSize),
		PathMonoids:     NewBufferSize[PathMonoid](pathTagWgs * pathReduceWg),
		PathBboxes:      NewBufferSize[PathBbox](numPaths),
		DrawReduced:     NewBufferSize[DrawMonoid](drawMonoidWgs),
		DrawMonoids:     NewBufferSize[DrawMonoid](numDrawObjects),
		Info:            NewBufferSize[uint32](layout.BinDataStart),
		ClipInps:        NewBufferSize[Clip](numClips),
		ClipEls:         NewBufferSize[ClipElement](numClips),
		ClipBics:        NewBufferSize[ClipBic](numClips / clipReduceWg),
		ClipBboxes:      NewBufferSize[ClipBbox](numClips),
		DrawBboxes:      NewBufferSize[DrawBbox](numPaths),
		BumpAlloc:       NewBufferSize[BumpAllocators](1),
		IndirectCount:   NewBufferSize[IndirectCount](1),
		BinHeaders:      NewBufferSize[BinHeader](binningWgs * 256),
		Paths:           NewBufferSize[Path](numPathsALigned),

		Lines:     lines,
		BinData:   binData,
		Tiles:     tiles,
		SegCounts: segCounts,
		Segments:  segments,
		Ptcl:      ptcl,
	}
}

func NewWorkgroupCounts(
	layout *Layout,
	widthInTiles uint32,
	heightInTiles uint32,
	numPathTags uint32,
) WorkgroupCounts {
	numPaths := layout.NumPaths
	numDrawObjects := layout.NumDrawObjects
	numClips := layout.NumClips
	pathTagPadded := jmath.AlignUp(numPathTags, 4*pathReduceWg)
	pathTagWgs := pathTagPadded / (4 * pathReduceWg)
	useLargePathScan := pathTagWgs > pathReduceWg
	var reducedSize uint32
	if useLargePathScan {
		reducedSize = jmath.AlignUp(pathTagWgs, pathReduceWg)
	} else {
		reducedSize = pathTagWgs
	}
	drawObjectWgs := (numDrawObjects + pathBboxWg - 1) / pathBboxWg
	drawMonoidWgs := min(drawObjectWgs, pathBboxWg)
	flattenWgs := (numPathTags + flattenWg - 1) / flattenWg
	numClipsMinusOne := numClips
	if numClips > 0 {
		numClipsMinusOne--
	}
	clipReduceWgs := numClipsMinusOne / clipReduceWg
	clipWgs := (numClips + clipReduceWg - 1) / clipReduceWg
	pathWgs := (numPaths + pathBboxWg - 1) / pathBboxWg
	widthInBins := (widthInTiles + 15) / 16
	heightInBins := (heightInTiles + 15) / 16
	return WorkgroupCounts{
		UseLargePathScan: useLargePathScan,
		PathReduce:       [3]uint32{pathTagWgs, 1, 1},
		PathReduce2:      [3]uint32{pathReduceWg, 1, 1},
		PathScan1:        [3]uint32{reducedSize / pathReduceWg, 1, 1},
		PathScan:         [3]uint32{pathTagWgs, 1, 1},
		BboxClear:        [3]uint32{drawObjectWgs, 1, 1},
		Flatten:          [3]uint32{flattenWgs, 1, 1},
		DrawReduce:       [3]uint32{drawMonoidWgs, 1, 1},
		DrawLeaf:         [3]uint32{drawMonoidWgs, 1, 1},
		ClipReduce:       [3]uint32{clipReduceWgs, 1, 1},
		ClipLeaf:         [3]uint32{clipWgs, 1, 1},
		Binning:          [3]uint32{drawObjectWgs, 1, 1},
		TileAlloc:        [3]uint32{pathWgs, 1, 1},
		PathCountSetup:   [3]uint32{1, 1, 1},
		Backdrop:         [3]uint32{pathWgs, 1, 1},
		Coarse:           [3]uint32{widthInBins, heightInBins, 1},
		PathTilingSetup:  [3]uint32{1, 1, 1},
		Fine:             [3]uint32{widthInTiles, heightInTiles, 1},
	}
}

func nextMultipleOf[T constraints.Integer](x, y T) T {
	r := x % y
	if r == 0 {
		return x
	} else {
		return x + y - r
	}
}

const pathReduceWg = 256
const pathBboxWg = 256
const flattenWg = 256
const tileWidth = 16
const tileHeight = 16
const clipReduceWg = 256

type BufferSizes struct {
	// Known size buffers
	PathReduced     BufferSize[PathMonoid]
	PathReduced2    BufferSize[PathMonoid]
	PathReducedScan BufferSize[PathMonoid]
	PathMonoids     BufferSize[PathMonoid]
	PathBboxes      BufferSize[PathBbox]
	DrawReduced     BufferSize[DrawMonoid]
	DrawMonoids     BufferSize[DrawMonoid]
	Info            BufferSize[uint32]
	ClipInps        BufferSize[Clip]
	ClipEls         BufferSize[ClipElement]
	ClipBics        BufferSize[ClipBic]
	ClipBboxes      BufferSize[ClipBbox]
	DrawBboxes      BufferSize[DrawBbox]
	BumpAlloc       BufferSize[BumpAllocators]
	IndirectCount   BufferSize[IndirectCount]
	BinHeaders      BufferSize[BinHeader]
	Paths           BufferSize[Path]
	// Bump allocated buffers
	Lines     BufferSize[LineSoup]
	BinData   BufferSize[uint32]
	Tiles     BufferSize[Tile]
	SegCounts BufferSize[SegmentCount]
	Segments  BufferSize[PathSegment]
	Ptcl      BufferSize[uint32]
}

type WorkgroupCounts struct {
	UseLargePathScan bool
	PathReduce       WorkgroupSize
	PathReduce2      WorkgroupSize
	PathScan1        WorkgroupSize
	PathScan         WorkgroupSize
	BboxClear        WorkgroupSize
	Flatten          WorkgroupSize
	DrawReduce       WorkgroupSize
	DrawLeaf         WorkgroupSize
	ClipReduce       WorkgroupSize
	ClipLeaf         WorkgroupSize
	Binning          WorkgroupSize
	TileAlloc        WorkgroupSize
	PathCountSetup   WorkgroupSize
	// Note `pathCount` must use an indirect dispatch
	Backdrop        WorkgroupSize
	Coarse          WorkgroupSize
	PathTilingSetup WorkgroupSize
	// Note `pathTiling` must use an indirect dispatch
	Fine WorkgroupSize
}

type BumpAllocators struct {
	_ structs.HostLayout

	Failed    uint32
	Binning   uint32
	Ptcl      uint32
	Tile      uint32
	SegCounts uint32
	Segments  uint32
	Blend     uint32
	Lines     uint32
}

func (ba *BumpAllocators) Memory() BumpAllocatorMemory {
	binning := NewBufferSize[uint32](ba.Binning)
	ptcl := NewBufferSize[uint32](ba.Ptcl)
	tile := NewBufferSize[Tile](ba.Tile)
	segCounts := NewBufferSize[SegmentCount](ba.SegCounts)
	segments := NewBufferSize[PathSegment](ba.Segments)
	lines := NewBufferSize[LineSoup](ba.Lines)

	total := binning.sizeInBytes() +
		ptcl.sizeInBytes() +
		tile.sizeInBytes() +
		segCounts.sizeInBytes() +
		segments.sizeInBytes() +
		lines.sizeInBytes()

	return BumpAllocatorMemory{
		Total:     total,
		Binning:   binning,
		Ptcl:      ptcl,
		Tile:      tile,
		SegCounts: segCounts,
		Segments:  segments,
		Lines:     lines,
	}
}

type BufferSize[T any] uint32

func NewBufferSize[T any](x uint32) BufferSize[T] {
	return BufferSize[T](max(x, 1))
}

func (s BufferSize[T]) sizeInBytes() uint32 {
	return uint32(s) * uint32(unsafe.Sizeof(*new(T)))
}

type BumpAllocatorMemory struct {
	Total     uint32
	Binning   BufferSize[uint32]
	Ptcl      BufferSize[uint32]
	Tile      BufferSize[Tile]
	SegCounts BufferSize[SegmentCount]
	Segments  BufferSize[PathSegment]
	Lines     BufferSize[LineSoup]
}

// IndirectCount stores indirect dispatch size values.
//
// The original plan was to reuse [BumpAllocators], but the WebGPU compatible
// usage list rules forbid that being used as indirect counts while also
// bound as writable.
type IndirectCount struct {
	_ structs.HostLayout

	X uint32
	Y uint32
	Z uint32
	_ uint32 // padding
}
