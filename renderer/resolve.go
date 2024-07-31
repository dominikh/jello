// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package renderer

import (
	"encoding/binary"
	"fmt"
	"image"
	"unsafe"

	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/jello/mem"
	"honnef.co/go/safeish"
)

type Resolver struct {
	rampCache rampCache
	patches   []ResolvedPatch
	images    map[image.Image]uint32
}

func NewResolver() *Resolver {
	return &Resolver{
		rampCache: rampCache{
			mapping: make(map[string]*rampCacheEntry),
		},
		images: make(map[image.Image]uint32),
	}
}

type ResolvedPatchKind int

const (
	ResolvedPatchKindRamp ResolvedPatchKind = iota + 1
	ResolvedPatchKindImage
)

type ResolvedPatch struct {
	Kind  ResolvedPatchKind
	Ramp  ResolvedPatchRamp
	Image ResolvedPatchImage
}

// XXX support glyphs

type ResolvedPatchRamp struct {
	DrawDataOffset int
	RampID         uint32
	Extend         gfx.Extend
}

type ResolvedPatchImage struct {
	// index of image in array
	Index uint32
	Image gfx.Image
	// offset to the index in the draw data stream
	DrawDataOffset int
}

func (r *Resolver) Resolve(arena *mem.Arena, enc *encoding.Encoding) (Layout, Ramps, []gfx.Image, []byte) {
	resources := enc.Resources
	if len(resources.Patches) == 0 {
		layout, packed := resolveSolidPathsOnly(arena, enc)
		return layout, Ramps{}, nil, packed
	}
	// TODO(dh): check if we can deduplicate code between this function and resolveSolidPathsOnly

	var data []byte
	patchSizes, imgs := r.resolvePatches(enc)
	layout := Layout{
		NumPaths: enc.NumPaths,
		NumClips: enc.NumClips,
	}
	sbs := computeSceneBufferSizes(enc, &patchSizes)
	data = mem.Grow(arena, data, sbs.bufferSize)
	{
		// Path tag stream
		layout.PathTagBase = sizeToWords(len(data))
		pos := 0
		stream := enc.PathTags
		// XXX glyph stuff
		if pos < len(stream) {
			data = mem.Append(arena, data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
		for range enc.NumOpenClips {
			data = mem.Append(arena, data, byte(encoding.PathTagPath))
		}
		if len(data) < sbs.pathTagPadded {
			data = mem.Grow(arena, data, sbs.pathTagPadded-len(data))[:sbs.pathTagPadded]
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
			data = mem.Append(arena, data, safeish.SliceCast[[]byte](stream[pos:])...)
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
		data = mem.Append(arena, data, safeish.SliceCast[[]byte](enc.DrawTags)...)
		for range enc.NumOpenClips {
			data = mem.Append(arena, data, byte(encoding.DrawTagEndClip))
		}
	}
	{
		// Draw data stream. This code replaces parts of the existing data
		// stream (enc.DrawData) with information computed during resolving.
		layout.DrawDataBase = sizeToWords(len(data))
		pos := 0
		stream := enc.DrawData
		for _, patch := range r.patches {
			// XXX support glyphs
			switch patch.Kind {
			case ResolvedPatchKindRamp:
				if pos < patch.Ramp.DrawDataOffset {
					data = mem.Append(arena, data, enc.DrawData[pos:patch.Ramp.DrawDataOffset]...)
				}
				indexMode := (patch.Ramp.RampID << 2) | uint32(patch.Ramp.Extend)
				data = mem.Grow(arena, data, 4)
				data = binary.LittleEndian.AppendUint32(data, indexMode)
				pos = patch.Ramp.DrawDataOffset + 4
			case ResolvedPatchKindImage:
				if pos < patch.Image.DrawDataOffset {
					data = mem.Append(arena, data, enc.DrawData[pos:patch.Image.DrawDataOffset]...)
				}
				data = mem.Grow(arena, data, 4)
				data = binary.LittleEndian.AppendUint32(data, patch.Image.Index)
				pos = patch.Image.DrawDataOffset + 4
			default:
				panic(fmt.Sprintf("unhandled kind %d", patch.Kind))
			}
		}
		if pos < len(stream) {
			data = mem.Append(arena, data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	{
		// Transform stream
		layout.TransformBase = sizeToWords(len(data))
		pos := 0
		stream := enc.Transforms
		// XXX glyph stuff
		if pos < len(stream) {
			data = mem.Append(arena, data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	{
		// Style stream
		layout.StyleBase = sizeToWords(len(data))
		pos := 0
		stream := enc.Styles
		// XXX glyph stuff
		if pos < len(stream) {
			data = mem.Append(arena, data, safeish.SliceCast[[]byte](stream[pos:])...)
		}
	}
	// XXX glyph stuff
	layout.NumDrawObjects = layout.NumPaths
	if sbs.bufferSize != len(data) {
		panic("internal error: buffer size mismatch")
	}
	return layout, r.rampCache.ramps(), imgs, data
}

func (r *Resolver) resolvePatches(enc *encoding.Encoding) (encoding.StreamOffsets, []gfx.Image) {
	r.rampCache.maintain()
	// XXX glyph stuff
	clear(r.images)
	clear(r.patches)
	r.patches = r.patches[:0]
	var sizes encoding.StreamOffsets
	resources := enc.Resources

	var imgs []gfx.Image
	var imgIdx uint32
	for _, patch := range resources.Patches {
		// XXX glyph stuff
		switch patch := patch.(type) {
		case encoding.RampPatch:
			rampID := r.rampCache.add(resources.ColorStops[patch.Stops[0]:patch.Stops[1]])
			r.patches = append(r.patches, ResolvedPatch{
				Kind: ResolvedPatchKindRamp,
				Ramp: ResolvedPatchRamp{
					DrawDataOffset: patch.DrawDataOffset + sizes.DrawData,
					RampID:         rampID,
					Extend:         patch.Extend,
				},
			})
		case encoding.ImagePatch:
			var idx uint32
			if id, ok := r.images[patch.Image.Image]; ok {
				idx = id
			} else {
				idx = imgIdx
				imgIdx++
				imgs = append(imgs, patch.Image)
				r.images[patch.Image.Image] = idx
			}
			r.patches = append(r.patches, ResolvedPatch{
				Kind: ResolvedPatchKindImage,
				Image: ResolvedPatchImage{
					Index:          idx,
					Image:          patch.Image,
					DrawDataOffset: patch.DrawDataOffset + sizes.DrawData,
				},
			})
		default:
			panic(fmt.Sprintf("unhandled type %T", patch))
		}
	}
	return sizes, imgs
}

func resolveSolidPathsOnly(arena *mem.Arena, enc *encoding.Encoding) (Layout, []byte) {
	if len(enc.Resources.Patches) != 0 {
		panic("this function doesn't support late bound resources")
	}
	layout := Layout{
		NumPaths: enc.NumPaths,
		NumClips: enc.NumClips,
	}
	sbs := computeSceneBufferSizes(enc, &encoding.StreamOffsets{})
	bufferSize := sbs.bufferSize
	pathTagPadded := sbs.pathTagPadded
	data := mem.NewSlice[[]byte](arena, 0, bufferSize)
	// Path tag stream
	layout.PathTagBase = sizeToWords(len(data))

	data = mem.Append(arena, data, safeish.SliceCast[[]byte](enc.PathTags)...)
	for range enc.NumOpenClips {
		data = mem.Append(arena, data, byte(encoding.PathTagPath))
	}
	if len(data) < pathTagPadded {
		data = mem.Grow(arena, data, pathTagPadded-len(data))[:pathTagPadded]
	} else if len(data) > pathTagPadded {
		data = data[:pathTagPadded]
	}
	// Path data stream
	layout.PathDataBase = sizeToWords(len(data))
	data = mem.Append(arena, data, enc.PathData...)
	// Draw tag stream
	layout.DrawTagBase = sizeToWords(len(data))
	// Bin data follows draw info
	for _, tag := range enc.DrawTags {
		layout.BinDataStart += tag.InfoSize()
	}
	data = mem.Append(arena, data, safeish.SliceCast[[]byte](enc.DrawTags)...)
	for range enc.NumOpenClips {
		data = mem.Grow(arena, data, 4)
		data = binary.LittleEndian.AppendUint32(data, uint32(encoding.DrawTagEndClip))
	}
	// Draw data stream
	layout.DrawDataBase = sizeToWords(len(data))
	data = mem.Append(arena, data, enc.DrawData...)
	// Transform stream
	layout.TransformBase = sizeToWords(len(data))
	data = mem.Append(arena, data, safeish.SliceCast[[]byte](enc.Transforms)...)
	// Style stream
	layout.StyleBase = sizeToWords(len(data))
	data = mem.Append(arena, data, safeish.SliceCast[[]byte](enc.Styles)...)
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
