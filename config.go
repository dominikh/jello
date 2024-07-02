package jello

import (
	"structs"
	"unsafe"

	"golang.org/x/exp/constraints"
	"honnef.co/go/brush"
)

const PATH_REDUCE_WG = 256
const PATH_BBOX_WG = 256
const FLATTEN_WG = 256
const TILE_WIDTH = 16
const TILE_HEIGHT = 16
const CLIP_REDUCE_WG = 256

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
	seg_counts := NewBufferSize[SegmentCount](ba.SegCounts)
	segments := NewBufferSize[PathSegment](ba.Segments)
	lines := NewBufferSize[LineSoup](ba.Lines)

	total := binning.size_in_bytes() +
		ptcl.size_in_bytes() +
		tile.size_in_bytes() +
		seg_counts.size_in_bytes() +
		segments.size_in_bytes() +
		lines.size_in_bytes()

	return BumpAllocatorMemory{
		Total:     total,
		Binning:   binning,
		Ptcl:      ptcl,
		Tile:      tile,
		SegCounts: seg_counts,
		Segments:  segments,
		Lines:     lines,
	}
}

type BufferSize[T any] uint32

func NewBufferSize[T any](x uint32) BufferSize[T] {
	return BufferSize[T](max(x, 1))
}

func (s BufferSize[T]) size_in_bytes() uint32 {
	// XXX can we avoid using unsafe for this?
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

type WorkgroupSize [3]uint32

// / Storage of indirect dispatch size values.
// /
// / The original plan was to reuse [`BumpAllocators`], but the WebGPU compatible
// / usage list rules forbid that being used as indirect counts while also
// / bound as writable.
type IndirectCount struct {
	_ structs.HostLayout

	Count_x uint32
	Count_y uint32
	Count_z uint32
	_       uint32
}

// / Uniform render configuration data used by all GPU stages.
// /
// / This data structure must be kept in sync with the definition in
// / `shaders/shared/config.wgsl`.
type ConfigUniform struct {
	_ structs.HostLayout

	/// Width of the scene in tiles.
	WidthInTiles uint32
	/// Height of the scene in tiles.
	HeightInTiles uint32
	/// Width of the target in pixels.
	TargetWidth uint32
	/// Height of the target in pixels.
	TargetHeight uint32
	/// The base background color applied to the target before any blends.
	BaseColor uint32
	/// Layout of packed scene data.
	Layout Layout
	/// Size of line soup buffer allocation (in [`LineSoup`]s)
	LinesSize uint32
	/// Size of binning buffer allocation (in `uint32`s).
	BinningSize uint32
	/// Size of tile buffer allocation (in [`Tile`]s).
	TilesSize uint32
	/// Size of segment count buffer allocation (in [`SegmentCount`]s).
	SegCountsSize uint32
	/// Size of segment buffer allocation (in [`PathSegment`]s).
	SegmentsSize uint32
	/// Size of per-tile command list buffer allocation (in `uint32`s).
	PtclSize uint32
}

type RenderConfig struct {
	gpu              ConfigUniform
	workgroup_counts WorkgroupCounts
	buffer_sizes     BufferSizes
}

func NewRenderConfig(layout *Layout, width, height uint32, baseColor brush.Color) RenderConfig {
	new_width := next_multiple_of(width, TILE_WIDTH)
	new_height := next_multiple_of(height, TILE_HEIGHT)
	width_in_tiles := new_width / TILE_WIDTH
	height_in_tiles := new_height / TILE_HEIGHT
	n_path_tags := layout.path_tags_size()
	workgroup_counts := NewWorkgroupCounts(layout, width_in_tiles, height_in_tiles, n_path_tags)
	buffer_sizes := NewBufferSizes(layout, &workgroup_counts)
	return RenderConfig{
		gpu: ConfigUniform{
			WidthInTiles:  width_in_tiles,
			HeightInTiles: height_in_tiles,
			TargetWidth:   width,
			TargetHeight:  height,
			BaseColor:     baseColor.PremulUint32(),
			LinesSize:     uint32(buffer_sizes.Lines),
			BinningSize:   uint32(buffer_sizes.Bin_data) - layout.bin_data_start,
			TilesSize:     uint32(buffer_sizes.Tiles),
			SegCountsSize: uint32(buffer_sizes.Seg_counts),
			SegmentsSize:  uint32(buffer_sizes.Segments),
			PtclSize:      uint32(buffer_sizes.Ptcl),
			Layout:        *layout,
		},
		workgroup_counts: workgroup_counts,
		buffer_sizes:     buffer_sizes,
	}
}

type BufferSizes struct {
	// Known size buffers
	Path_reduced      BufferSize[PathMonoid]
	Path_reduced2     BufferSize[PathMonoid]
	Path_reduced_scan BufferSize[PathMonoid]
	Path_monoids      BufferSize[PathMonoid]
	Path_bboxes       BufferSize[PathBbox]
	Draw_reduced      BufferSize[DrawMonoid]
	Draw_monoids      BufferSize[DrawMonoid]
	Info              BufferSize[uint32]
	Clip_inps         BufferSize[Clip]
	Clip_els          BufferSize[ClipElement]
	Clip_bics         BufferSize[ClipBic]
	Clip_bboxes       BufferSize[ClipBbox]
	Draw_bboxes       BufferSize[DrawBbox]
	Bump_alloc        BufferSize[BumpAllocators]
	Indirect_count    BufferSize[IndirectCount]
	Bin_headers       BufferSize[BinHeader]
	Paths             BufferSize[Path]
	// Bump allocated buffers
	Lines      BufferSize[LineSoup]
	Bin_data   BufferSize[uint32]
	Tiles      BufferSize[Tile]
	Seg_counts BufferSize[SegmentCount]
	Segments   BufferSize[PathSegment]
	Ptcl       BufferSize[uint32]
}

func NewBufferSizes(layout *Layout, workgroups *WorkgroupCounts) BufferSizes {
	n_paths := layout.n_paths
	n_draw_objects := layout.n_draw_objects
	n_clips := layout.n_clips
	path_tag_wgs := workgroups.Path_reduce[0]
	var reduced_size uint32
	if workgroups.Use_large_path_scan {
		reduced_size = align_upu32(path_tag_wgs, PATH_REDUCE_WG)
	} else {
		reduced_size = path_tag_wgs
	}
	draw_monoid_wgs := workgroups.Draw_reduce[0]

	binning_wgs := workgroups.Binning[0]
	n_paths_aligned := align_upu32(n_paths, 256)

	// The following buffer sizes have been hand picked to accommodate the vello test scenes as
	// well as paris-30k. These should instead get derived from the scene layout using
	// reasonable heuristics.
	bin_data := NewBufferSize[uint32](1 << 18)
	tiles := NewBufferSize[Tile](1 << 21)
	lines := NewBufferSize[LineSoup](1 << 21)
	seg_counts := NewBufferSize[SegmentCount](1 << 21)
	segments := NewBufferSize[PathSegment](1 << 21)
	ptcl := NewBufferSize[uint32](1 << 23)
	return BufferSizes{
		Path_reduced:      NewBufferSize[PathMonoid](reduced_size),
		Path_reduced2:     NewBufferSize[PathMonoid](PATH_REDUCE_WG),
		Path_reduced_scan: NewBufferSize[PathMonoid](reduced_size),
		Path_monoids:      NewBufferSize[PathMonoid](path_tag_wgs * PATH_REDUCE_WG),
		Path_bboxes:       NewBufferSize[PathBbox](n_paths),
		Draw_reduced:      NewBufferSize[DrawMonoid](draw_monoid_wgs),
		Draw_monoids:      NewBufferSize[DrawMonoid](n_draw_objects),
		Info:              NewBufferSize[uint32](layout.bin_data_start),
		Clip_inps:         NewBufferSize[Clip](n_clips),
		Clip_els:          NewBufferSize[ClipElement](n_clips),
		Clip_bics:         NewBufferSize[ClipBic](n_clips / CLIP_REDUCE_WG),
		Clip_bboxes:       NewBufferSize[ClipBbox](n_clips),
		Draw_bboxes:       NewBufferSize[DrawBbox](n_paths),
		Bump_alloc:        NewBufferSize[BumpAllocators](1),
		Indirect_count:    NewBufferSize[IndirectCount](1),
		Bin_headers:       NewBufferSize[BinHeader](binning_wgs * 256),
		Paths:             NewBufferSize[Path](n_paths_aligned),

		Lines:      lines,
		Bin_data:   bin_data,
		Tiles:      tiles,
		Seg_counts: seg_counts,
		Segments:   segments,
		Ptcl:       ptcl,
	}
}

type WorkgroupCounts struct {
	Use_large_path_scan bool
	Path_reduce         WorkgroupSize
	Path_reduce2        WorkgroupSize
	Path_scan1          WorkgroupSize
	Path_scan           WorkgroupSize
	Bbox_clear          WorkgroupSize
	Flatten             WorkgroupSize
	Draw_reduce         WorkgroupSize
	Draw_leaf           WorkgroupSize
	Clip_reduce         WorkgroupSize
	Clip_leaf           WorkgroupSize
	Binning             WorkgroupSize
	Tile_alloc          WorkgroupSize
	Path_count_setup    WorkgroupSize
	// Note `path_count` must use an indirect dispatch
	Backdrop          WorkgroupSize
	Coarse            WorkgroupSize
	Path_tiling_setup WorkgroupSize
	// Note `path_tiling` must use an indirect dispatch
	Fine WorkgroupSize
}

func NewWorkgroupCounts(
	layout *Layout,
	width_in_tiles uint32,
	height_in_tiles uint32,
	n_path_tags uint32,
) WorkgroupCounts {
	n_paths := layout.n_paths
	n_draw_objects := layout.n_draw_objects
	n_clips := layout.n_clips
	path_tag_padded := align_upu32(n_path_tags, 4*PATH_REDUCE_WG)
	path_tag_wgs := path_tag_padded / (4 * PATH_REDUCE_WG)
	use_large_path_scan := path_tag_wgs > PATH_REDUCE_WG
	var reduced_size uint32
	if use_large_path_scan {
		reduced_size = align_upu32(path_tag_wgs, PATH_REDUCE_WG)
	} else {
		reduced_size = path_tag_wgs
	}
	draw_object_wgs := (n_draw_objects + PATH_BBOX_WG - 1) / PATH_BBOX_WG
	draw_monoid_wgs := min(draw_object_wgs, PATH_BBOX_WG)
	flatten_wgs := (n_path_tags + FLATTEN_WG - 1) / FLATTEN_WG
	n_clips_minus_one := n_clips
	if n_clips > 0 {
		n_clips_minus_one--
	}
	clip_reduce_wgs := n_clips_minus_one / CLIP_REDUCE_WG
	clip_wgs := (n_clips + CLIP_REDUCE_WG - 1) / CLIP_REDUCE_WG
	path_wgs := (n_paths + PATH_BBOX_WG - 1) / PATH_BBOX_WG
	width_in_bins := (width_in_tiles + 15) / 16
	height_in_bins := (height_in_tiles + 15) / 16
	return WorkgroupCounts{
		Use_large_path_scan: use_large_path_scan,
		Path_reduce:         [3]uint32{path_tag_wgs, 1, 1},
		Path_reduce2:        [3]uint32{PATH_REDUCE_WG, 1, 1},
		Path_scan1:          [3]uint32{reduced_size / PATH_REDUCE_WG, 1, 1},
		Path_scan:           [3]uint32{path_tag_wgs, 1, 1},
		Bbox_clear:          [3]uint32{draw_object_wgs, 1, 1},
		Flatten:             [3]uint32{flatten_wgs, 1, 1},
		Draw_reduce:         [3]uint32{draw_monoid_wgs, 1, 1},
		Draw_leaf:           [3]uint32{draw_monoid_wgs, 1, 1},
		Clip_reduce:         [3]uint32{clip_reduce_wgs, 1, 1},
		Clip_leaf:           [3]uint32{clip_wgs, 1, 1},
		Binning:             [3]uint32{draw_object_wgs, 1, 1},
		Tile_alloc:          [3]uint32{path_wgs, 1, 1},
		Path_count_setup:    [3]uint32{1, 1, 1},
		Backdrop:            [3]uint32{path_wgs, 1, 1},
		Coarse:              [3]uint32{width_in_bins, height_in_bins, 1},
		Path_tiling_setup:   [3]uint32{1, 1, 1},
		Fine:                [3]uint32{width_in_tiles, height_in_tiles, 1},
	}
}

func next_multiple_of[T constraints.Integer](x, y T) T {
	r := x % y
	if r == 0 {
		return x
	} else {
		return x + y - r
	}
}
