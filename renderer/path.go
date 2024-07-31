// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package renderer

import (
	"math/bits"
	"structs"
	"unsafe"

	"honnef.co/go/jello/encoding"
)

type PathMonoid struct {
	_ structs.HostLayout

	// Index into transform stream.
	TransIdx uint32
	// Path segment index.
	PathSegIdx uint32
	// Offset into path segment stream.
	PathSegOffset uint32
	// Index into style stream.
	StyleIdx uint32
	// Index of containing path.
	PathIdx uint32
}

func NewPathMonoid(tagWord uint32) PathMonoid {
	var c PathMonoid
	point_count := tagWord & 0x3030303
	c.PathSegIdx = uint32(bits.OnesCount32(((point_count * 7) & 0x4040404)))
	c.TransIdx = uint32(bits.OnesCount32((tagWord & (uint32(encoding.PathTagTransform) * 0x1010101))))
	n_points := point_count + ((tagWord >> 2) & 0x1010101)
	a := n_points + (n_points & (((tagWord >> 3) & 0x1010101) * 15))
	a += a >> 8
	a += a >> 16
	c.PathSegOffset = a & 0xff
	c.PathIdx = uint32(bits.OnesCount32((tagWord & (uint32(encoding.PathTagPath) * 0x1010101))))
	style_size := int(unsafe.Sizeof(encoding.Style{}) / 4)
	c.StyleIdx = uint32(bits.OnesCount32((tagWord & (uint32(encoding.PathTagStyle) * 0x1010101))) * style_size)
	return c
}

func (m PathMonoid) Combine(other PathMonoid) PathMonoid {
	return PathMonoid{
		TransIdx:      m.TransIdx + other.TransIdx,
		PathSegIdx:    m.PathSegIdx + other.PathSegIdx,
		PathSegOffset: m.PathSegOffset + other.PathSegOffset,
		StyleIdx:      m.StyleIdx + other.StyleIdx,
		PathIdx:       m.PathIdx + other.PathIdx,
	}
}

type PathBbox struct {
	_ structs.HostLayout

	// Minimum x value.
	X0 int32
	// Minimum y value.
	Y0 int32
	// Maximum x value.
	X1 int32
	// Maximum y value.
	Y1 int32
	// Style flags
	DrawFlags uint32
	// Index into the transform stream.
	TransIdx uint32
}

type Path struct {
	_ structs.HostLayout

	Bbox  [4]uint32
	Tiles uint32
	_     [3]uint32
}

type LineSoup struct {
	_ structs.HostLayout

	PathIdx uint32
	_       uint32 // padding
	P0      [2]float32
	P1      [2]float32
}

type PathSegment struct {
	_ structs.HostLayout

	Point0 [2]float32
	Point1 [2]float32
	YEdge  float32
	_      uint32 // padding
}

type Tile struct {
	_ structs.HostLayout

	Backdrop          int32
	SegmentCountOrIdx uint32
}

type SegmentCount struct {
	_ structs.HostLayout

	LineIdx uint32
	Counts  uint32
}
