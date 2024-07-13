package renderer

import (
	"encoding/binary"
	"slices"
	"unsafe"

	"honnef.co/go/brush"
	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/safeish"
)

type Resolver struct {
	rampCache rampCache
	patches   []ResolvedPatch
}

func NewResolver() *Resolver {
	return &Resolver{
		rampCache: rampCache{
			mapping: make(map[string]*rampCacheEntry),
		},
	}
}

type ResolvedPatch interface {
	isResolvedPatch()
}

// XXX support glyphs and images

type ResolvedPatchRamp struct {
	DrawDataOffset int
	RampID         uint32
	Extend         brush.Extend
}

func (ResolvedPatchRamp) isResolvedPatch() {}

func (r *Resolver) Resolve(enc *encoding.Encoding, packed []byte) (Layout, Ramps, Images, []byte) {
	resources := enc.Resources
	if len(resources.Patches) == 0 {
		layout, packed := resolveSolidPathsOnly(enc, packed)
		return layout, Ramps{}, Images{}, packed
	}
	// TODO(dh): check if we can deduplicate code between this function and resolveSolidPathsOnly

	patchSizes := r.resolvePatches(enc)
	// XXX support images
	// r.resolvePendingImages()
	data := packed[:0]
	layout := Layout{
		NumPaths: enc.NumPaths,
		NumClips: enc.NumClips,
	}
	sbs := computeSceneBufferSizes(enc, &patchSizes)
	data = slices.Grow(data, sbs.bufferSize)
	{
		// Path tag stream
		layout.PathTagBase = sizeToWords(len(data))
		pos := 0
		stream := enc.PathTags
		// XXX glyph stuff
		if pos < len(stream) {
			data = append(data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
		for range enc.NumOpenClips {
			data = append(data, byte(encoding.PathTagPath))
		}
		if len(data) < sbs.pathTagPadded {
			data = slices.Grow(data, sbs.pathTagPadded-len(data))[:sbs.pathTagPadded]
		} else {
			data = data[:sbs.pathTagPadded]
		}
	}
	{
		// Path data stream
		layout.PathDataBase = sizeToWords(len(data))
		pos := 0
		stream := enc.PathData
		// XXX glyph stuff
		if pos < len(stream) {
			data = append(data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	// Draw tag stream
	layout.DrawTagBase = sizeToWords(len(data))
	{
		// Bin data follows draw info
		var sum uint32
		for _, tag := range enc.DrawTags {
			sum += tag.InfoSize()
		}
		layout.BinDataStart = sum
		data = append(data, safeish.SliceCast[[]byte](enc.DrawTags)...)
		for range enc.NumOpenClips {
			data = append(data, byte(encoding.DrawTagEndClip))
		}
	}
	{
		// Draw data stream
		layout.DrawDataBase = sizeToWords(len(data))
		pos := 0
		stream := enc.DrawData
		for _, patch := range r.patches {
			// XXX support glyphs and images
			switch patch := patch.(type) {
			case ResolvedPatchRamp:
				if pos < patch.DrawDataOffset {
					data = append(data, enc.DrawData[pos:patch.DrawDataOffset]...)
				}
				indexMode := (patch.RampID << 2) | uint32(patch.Extend)
				data = binary.LittleEndian.AppendUint32(data, indexMode)
				pos = patch.DrawDataOffset + 4
			}
		}
		if pos < len(stream) {
			data = append(data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	{
		// Transform stream
		layout.TransformBase = sizeToWords(len(data))
		pos := 0
		stream := enc.Transforms
		// XXX glyph stuff
		if pos < len(stream) {
			data = append(data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	{
		// Style stream
		layout.StyleBase = sizeToWords(len(data))
		pos := 0
		stream := enc.Styles
		// XXX glyph stuff
		if pos < len(stream) {
			data = append(data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	// XXX glyph stuff
	layout.NumDrawObjects = layout.NumPaths
	if sbs.bufferSize != len(data) {
		panic("internal error: buffer size mismatch")
	}
	// XXX support images
	return layout, r.rampCache.ramps(), Images{}, data
}

func (r *Resolver) resolvePatches(enc *encoding.Encoding) encoding.StreamOffsets {
	r.rampCache.maintain()
	// XXX glyph stuff
	// XXX image stuff
	// OPT(dh): make sure we actually need to clear r.patches; does it store pointers?
	clear(r.patches)
	r.patches = r.patches[:0]
	var sizes encoding.StreamOffsets
	resources := enc.Resources
	for _, patch := range resources.Patches {
		// XXX glyph stuff
		// XXX image stuff
		switch patch := patch.(type) {
		case encoding.RampPatch:
			rampID := r.rampCache.add(resources.ColorStops[patch.Stops[0]:patch.Stops[1]])
			r.patches = append(r.patches, ResolvedPatchRamp{
				DrawDataOffset: patch.DrawDataOffset + sizes.DrawData,
				RampID:         rampID,
				Extend:         patch.Extend,
			})
		}
	}
	return sizes
}

func resolveSolidPathsOnly(enc *encoding.Encoding, data []byte) (Layout, []byte) {
	if len(enc.Resources.Patches) != 0 {
		panic("this function doesn't support late bound resources")
	}
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
