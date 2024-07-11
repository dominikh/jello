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
	NumDrawObjects uint32
	/// Number of paths.
	NumPaths uint32
	/// Number of clips.
	NumClips uint32
	/// Start of binning data.
	BinDataStart uint32
	/// Start of path tag stream.
	PathTagBase uint32
	/// Start of path data stream.
	PathDataBase uint32
	/// Start of draw tag stream.
	DrawTagBase uint32
	/// Start of draw data stream.
	DrawDataBase uint32
	/// Start of transform stream.
	TransformBase uint32
	/// Start of style stream.
	StyleBase uint32
}

func (l *Layout) pathTagsSize() uint32 {
	start := l.PathTagBase * 4
	end := l.PathDataBase * 4
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
		NumPaths: encoding.NumPaths,
		NumClips: encoding.NumClips,
	}
	sbs := computeSceneBufferSizes(encoding, &StreamOffsets{})
	bufferSize := sbs.bufferSize
	pathTagPadded := sbs.pathTagPadded
	data = slices.Grow(data, bufferSize)
	// Path tag stream
	layout.PathTagBase = sizeToWords(len(data))

	data = append(data, safeish.SliceCast[[]byte](encoding.PathTags)...)
	for range encoding.NumOpenClips {
		data = append(data, byte(PathTagPath))
	}
	if len(data) < pathTagPadded {
		data = slices.Grow(data, pathTagPadded-len(data))[:pathTagPadded]
	} else if len(data) > pathTagPadded {
		data = data[:pathTagPadded]
	}
	// Path data stream
	layout.PathDataBase = sizeToWords(len(data))
	data = append(data, encoding.PathData...)
	// Draw tag stream
	layout.DrawTagBase = sizeToWords(len(data))
	// Bin data follows draw info
	for _, tag := range encoding.DrawTags {
		layout.BinDataStart += tag.InfoSize()
	}
	data = append(data, safeish.SliceCast[[]byte](encoding.DrawTags)...)
	for range encoding.NumOpenClips {
		data = binary.LittleEndian.AppendUint32(data, uint32(DrawTagEndClip))
	}
	// Draw data stream
	layout.DrawDataBase = sizeToWords(len(data))
	data = append(data, encoding.DrawData...)
	// Transform stream
	layout.TransformBase = sizeToWords(len(data))
	data = append(data, safeish.SliceCast[[]byte](encoding.Transforms)...)
	// Style stream
	layout.StyleBase = sizeToWords(len(data))
	data = append(data, safeish.SliceCast[[]byte](encoding.Styles)...)
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

func computeSceneBufferSizes(encoding *Encoding, patchSizes *StreamOffsets) sceneBufferSizes {
	numPathTags := len(encoding.PathTags) + patchSizes.PathTags + int(encoding.NumOpenClips)
	pathTagPadded := alignUp(numPathTags, 4*pathReduceWg)
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

func alignUp(len int, alignment int) int {
	return (len + alignment - 1) & -alignment
}

// TODO(dh): make alignUp generic and remove alignUpU32
func alignUpU32(len uint32, alignment uint32) uint32 {
	return (len + alignment - 1) & -alignment
}
