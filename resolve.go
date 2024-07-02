package jello

import (
	"encoding/binary"
	"slices"
	"structs"
	"unsafe"

	"honnef.co/go/safeish"
)

type Layout struct {
	_ structs.HostLayout

	/// Number of draw objects.
	n_draw_objects uint32
	/// Number of paths.
	n_paths uint32
	/// Number of clips.
	n_clips uint32
	/// Start of binning data.
	bin_data_start uint32
	/// Start of path tag stream.
	path_tag_base uint32
	/// Start of path data stream.
	path_data_base uint32
	/// Start of draw tag stream.
	draw_tag_base uint32
	/// Start of draw data stream.
	draw_data_base uint32
	/// Start of transform stream.
	transform_base uint32
	/// Start of style stream.
	style_base uint32
}

func (l *Layout) path_tags_size() uint32 {
	start := l.path_tag_base * 4
	end := l.path_data_base * 4
	return end - start
}

type Resolver struct{}

func NewResolver() *Resolver {
	return &Resolver{}
}

func (r *Resolver) Resolve(encoding *Encoding, packed []byte) (Layout, Ramps, Images, []byte) {
	// XXX implement late bound resources
	layout, packed := ResolveSolidPathsOnly(encoding, packed)
	return layout, Ramps{}, Images{}, packed
}

func ResolveSolidPathsOnly(encoding *Encoding, data []byte) (Layout, []byte) {
	// assert!(
	//     encoding.resources.patches.is_empty(),
	//     "this resolve function doesn't support late bound resources"
	// );
	data = data[:0]
	layout := Layout{
		n_paths: encoding.NumPaths,
		n_clips: encoding.NumClips,
	}
	sbs := computeSceneBufferSizes(encoding, &StreamOffsets{})
	buffer_size := sbs.bufferSize
	path_tag_padded := sbs.pathTagPadded
	data = slices.Grow(data, buffer_size)
	// Path tag stream
	layout.path_tag_base = size_to_words(len(data))

	data = append(data, safeish.SliceCast[[]byte](encoding.PathTags)...)
	for range encoding.NumOpenClips {
		data = append(data, byte(PATH))
	}
	if len(data) < path_tag_padded {
		data = slices.Grow(data, path_tag_padded-len(data))[:path_tag_padded]
	} else if len(data) > path_tag_padded {
		data = data[:path_tag_padded]
	}
	// Path data stream
	layout.path_data_base = size_to_words(len(data))
	data = append(data, encoding.PathData...)
	// Draw tag stream
	layout.draw_tag_base = size_to_words(len(data))
	// Bin data follows draw info
	for _, tag := range encoding.DrawTags {
		layout.bin_data_start += tag.InfoSize()
	}
	data = append(data, safeish.SliceCast[[]byte](encoding.DrawTags)...)
	for range encoding.NumOpenClips {
		data = binary.LittleEndian.AppendUint32(data, uint32(END_CLIP))
	}
	// Draw data stream
	layout.draw_data_base = size_to_words(len(data))
	data = append(data, encoding.DrawData...)
	// Transform stream
	layout.transform_base = size_to_words(len(data))
	data = append(data, safeish.SliceCast[[]byte](encoding.Transforms)...)
	// Style stream
	layout.style_base = size_to_words(len(data))
	data = append(data, safeish.SliceCast[[]byte](encoding.Styles)...)
	layout.n_draw_objects = layout.n_paths
	if buffer_size != len(data) {
		panic("invalid encoding")
	}
	return layout, data
}

type sceneBufferSizes struct {
	bufferSize    int
	pathTagPadded int
}

func computeSceneBufferSizes(encoding *Encoding, patch_sizes *StreamOffsets) sceneBufferSizes {
	n_path_tags := len(encoding.PathTags) + patch_sizes.PathTags + int(encoding.NumOpenClips)
	path_tag_padded := align_up(n_path_tags, 4*PATH_REDUCE_WG)
	buffer_size := path_tag_padded +
		slice_size_in_bytes(encoding.PathData, patch_sizes.PathData) +
		slice_size_in_bytes(
			encoding.DrawTags,
			patch_sizes.DrawTags+int(encoding.NumOpenClips),
		) +
		slice_size_in_bytes(encoding.DrawData, patch_sizes.DrawData) +
		slice_size_in_bytes(encoding.Transforms, patch_sizes.Transforms) +
		slice_size_in_bytes(encoding.Styles, patch_sizes.Styles)
	return sceneBufferSizes{
		buffer_size,
		path_tag_padded,
	}
}

func size_to_words(n int) uint32 {
	return uint32(n) / 4
}

func slice_size_in_bytes[E any, T ~[]E](slice T, extra int) int {
	return (len(slice) + extra) * int(unsafe.Sizeof(*new(E)))
}

func align_up(len int, alignment int) int {
	return (len + alignment - 1) & -alignment
}

func align_upu32(len uint32, alignment uint32) uint32 {
	return (len + alignment - 1) & -alignment
}
