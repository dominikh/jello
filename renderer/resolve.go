package renderer

import (
	"encoding/binary"
	"slices"
	"unsafe"

	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/safeish"
)

type Resolver struct{}

func NewResolver() *Resolver {
	return &Resolver{}
}

func (r *Resolver) Resolve(encoding *encoding.Encoding, packed []byte) (Layout, Ramps, Images, []byte) {
	// XXX implement late bound resources
	layout, packed := ResolveSolidPathsOnly(encoding, packed)
	return layout, Ramps{}, Images{}, packed
}

func ResolveSolidPathsOnly(enc *encoding.Encoding, data []byte) (Layout, []byte) {
	// assert!(
	//     encoding.resources.patches.is_empty(),
	//     "this resolve function doesn't support late bound resources"
	// );
	data = data[:0]
	layout := Layout{
		NumPaths: enc.NumPaths,
		NumClips: enc.NumClips,
	}
	sbs := computeSceneBufferSizes(enc, &encoding.StreamOffsets{})
	bufferSize := sbs.bufferSize
	pathTagPadded := sbs.pathTagPadded
	data = slices.Grow(data, bufferSize)
	// Path tag stream
	layout.PathTagBase = sizeToWords(len(data))

	data = append(data, safeish.SliceCast[[]byte](enc.PathTags)...)
	for range enc.NumOpenClips {
		data = append(data, byte(encoding.PathTagPath))
	}
	if len(data) < pathTagPadded {
		data = slices.Grow(data, pathTagPadded-len(data))[:pathTagPadded]
	} else if len(data) > pathTagPadded {
		data = data[:pathTagPadded]
	}
	// Path data stream
	layout.PathDataBase = sizeToWords(len(data))
	data = append(data, enc.PathData...)
	// Draw tag stream
	layout.DrawTagBase = sizeToWords(len(data))
	// Bin data follows draw info
	for _, tag := range enc.DrawTags {
		layout.BinDataStart += tag.InfoSize()
	}
	data = append(data, safeish.SliceCast[[]byte](enc.DrawTags)...)
	for range enc.NumOpenClips {
		data = binary.LittleEndian.AppendUint32(data, uint32(encoding.DrawTagEndClip))
	}
	// Draw data stream
	layout.DrawDataBase = sizeToWords(len(data))
	data = append(data, enc.DrawData...)
	// Transform stream
	layout.TransformBase = sizeToWords(len(data))
	data = append(data, safeish.SliceCast[[]byte](enc.Transforms)...)
	// Style stream
	layout.StyleBase = sizeToWords(len(data))
	data = append(data, safeish.SliceCast[[]byte](enc.Styles)...)
	layout.NumDrawObjects = layout.NumPaths
	if bufferSize != len(data) {
		panic("invalid encoding")
	}
	return layout, data
}

type sceneBufferSizes struct {
	bufferSize    int
	pathTagPadded int
}

func computeSceneBufferSizes(encoding *encoding.Encoding, patchSizes *encoding.StreamOffsets) sceneBufferSizes {
	numPathTags := len(encoding.PathTags) + patchSizes.PathTags + int(encoding.NumOpenClips)
	pathTagPadded := jmath.AlignUp(numPathTags, 4*pathReduceWg)
	bufferSize := pathTagPadded +
		sliceSizeInBytes(encoding.PathData, patchSizes.PathData) +
		sliceSizeInBytes(
			encoding.DrawTags,
			patchSizes.DrawTags+int(encoding.NumOpenClips),
		) +
		sliceSizeInBytes(encoding.DrawData, patchSizes.DrawData) +
		sliceSizeInBytes(encoding.Transforms, patchSizes.Transforms) +
		sliceSizeInBytes(encoding.Styles, patchSizes.Styles)
	return sceneBufferSizes{
		bufferSize,
		pathTagPadded,
	}
}

func sizeToWords(n int) uint32 {
	return uint32(n) / 4
}

func sliceSizeInBytes[E any, T ~[]E](slice T, extra int) int {
	return (len(slice) + extra) * int(unsafe.Sizeof(*new(E)))
}
