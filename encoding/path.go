// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package encoding

import (
	"encoding/binary"
	"iter"
	"math"
	"structs"

	"honnef.co/go/curve"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/jmath"
)

type Style struct {
	_ structs.HostLayout

	// Encodes the stroke and fill style parameters. This field is split into two 16-bit
	// parts:
	//
	// - `flags: u16` - encodes fill vs stroke, even-odd vs non-zero fill mode for fills and cap
	//                  and join style for strokes. See the FLAGS_* constants below for more
	//                  information.
	// ```text
	// flags: |style|fill|join|start cap|end cap|reserved|
	//  bits:  0     1    2-3  4-5       6-7     8-15
	// ```
	//
	// - `miter_limit: u16` - The miter limit for a stroke, encoded in binary16 (half) floating
	//                        point representation. This field is only meaningful for the
	//                        `Join::Miter` join style. It's ignored for other stroke styles and
	//                        fills.
	FlagsAndMiterLimits uint32
	LineWidth           float32
}

const (
	// 0 for a fill, 1 for a stroke
	FlagsStyleBit uint32 = 0x8000_0000

	// 0 for non-zero, 1 for even-odd
	FlagsFillBit uint32 = 0x4000_0000

	// Encodings for join style:
	//    - 0b00 -> bevel
	//    - 0b01 -> miter
	//    - 0b10 -> round
	FlagsJoinBitsBevel uint32 = 0
	FlagsJoinBitsMiter uint32 = 0x1000_0000
	FlagsJoinBitsRound uint32 = 0x2000_0000
	FlagsJoinMask      uint32 = 0x3000_0000

	// Encodings for cap style:
	//    - 0b00 -> butt
	//    - 0b01 -> square
	//    - 0b10 -> round
	flagsCapBitsButt   uint32 = 0
	FlagsCapBitsSquare uint32 = 0x0100_0000
	FlagsCapBitsRound  uint32 = 0x0200_0000

	flagsStartCapBitsButt   uint32 = flagsCapBitsButt << 2
	flagsStartCapBitsSquare uint32 = FlagsCapBitsSquare << 2
	flagsStartCapBitsRound  uint32 = FlagsCapBitsRound << 2
	flagsEndCapBitsButt     uint32 = flagsCapBitsButt
	flagsEndCapBitsSquare   uint32 = FlagsCapBitsSquare
	flagsEndCapBitsRound    uint32 = FlagsCapBitsRound

	FlagsStartCapMask uint32 = 0x0C00_0000
	FlagsEndCapMask   uint32 = 0x0300_0000
	MiterLimitMask    uint32 = 0xFFFF
)

func styleFromFill(fill gfx.Fill) Style {
	var fillBit uint32
	if fill == gfx.EvenOdd {
		fillBit = FlagsFillBit
	}
	return Style{
		FlagsAndMiterLimits: fillBit,
		LineWidth:           0,
	}
}

func styleFromStroke(stroke curve.Stroke) Style {
	style := FlagsStyleBit
	var join uint32
	switch stroke.Join {
	case curve.BevelJoin:
		join = FlagsJoinBitsBevel
	case curve.MiterJoin:
		join = FlagsJoinBitsMiter
	case curve.RoundJoin:
		join = FlagsJoinBitsRound
	}
	var startCap uint32
	switch stroke.StartCap {
	case curve.ButtCap:
		startCap = flagsStartCapBitsButt
	case curve.SquareCap:
		startCap = flagsStartCapBitsSquare
	case curve.RoundCap:
		startCap = flagsStartCapBitsRound
	}
	var endCap uint32
	switch stroke.EndCap {
	case curve.ButtCap:
		endCap = flagsEndCapBitsButt
	case curve.SquareCap:
		endCap = flagsEndCapBitsSquare
	case curve.RoundCap:
		endCap = flagsEndCapBitsRound
	}
	miterLimit := uint32(jmath.Float16(float32(stroke.MiterLimit)))
	return Style{
		FlagsAndMiterLimits: style | join | startCap | endCap | miterLimit,
		LineWidth:           float32(stroke.Width),
	}
}

type pathSegmentType uint8

const (
	pathSegmentTypeLineTo  pathSegmentType = 0x1
	pathSegmentTypeQuadTo  pathSegmentType = 0x2
	pathSegmentTypeCubicTo pathSegmentType = 0x3
)

type PathTag uint8

const (
	// 32-bit floating point line segment.
	//
	// This is equivalent to `(PathSegmentType::LineTo | PathTag::F32_BIT)`.
	PathTagLineToF32 PathTag = 0x9

	// 32-bit floating point quadratic segment.
	//
	// This is equivalent to `(PathSegmentType::QUAD_TO | PathTag::F32_BIT)`.
	PathTagQuadToF32 PathTag = 0xa

	// 32-bit floating point cubic segment.
	//
	// This is equivalent to `(PathSegmentType::CUBIC_TO | PathTag::F32_BIT)`.
	PathTagCubicToF32 PathTag = 0xb

	// 16-bit integral line segment.
	PathTagLineToI16 PathTag = 0x1

	// 16-bit integral quadratic segment.
	PathTagQuadToI16 PathTag = 0x2

	// 16-bit integral cubic segment.
	PathTagCubicToI16 PathTag = 0x3

	// Transform marker.
	PathTagTransform PathTag = 0x20

	// Path marker.
	PathTagPath PathTag = 0x10

	// Style setting.
	PathTagStyle PathTag = 0x40

	// Bit that marks a segment that is the end of a subpath.
	PathTagSubpathEndBit PathTag = 0x4

	// Bit for path segments that are represented as f32 values. If unset
	// they are represented as i16.
	PathTagF32Bit PathTag = 0x8

	// Mask for bottom 3 bits that contain the [`PathSegmentType`].
	PathTagSegmentMask PathTag = 0x3
)

func (tag *PathTag) setSubpathEnd() { *tag |= PathTagSubpathEndBit }
func (tag PathTag) pathSegmentType() pathSegmentType {
	return pathSegmentType(tag & PathTagSegmentMask)
}

type pathEncoder struct {
	tags                 *[]PathTag
	data                 *[]byte
	numSegments          *uint32
	numPaths             *uint32
	firstPoint           [2]float32
	firstStartTangentEnd [2]float32
	state                pathState
	numEncodedSegments   uint32
	isFill               bool
}

type pathState int

const (
	pathStateStart pathState = iota
	pathStateMoveTo
	pathStateNonemptySubpath
)

func (enc *pathEncoder) lastPoint() ([2]float32, bool) {
	n := len(*enc.data)
	if n < 8 {
		return [2]float32{}, false
	}
	x := binary.LittleEndian.Uint32((*enc.data)[n-8 : n-4])
	y := binary.LittleEndian.Uint32((*enc.data)[n-4 : n])
	return [2]float32{math.Float32frombits(x), math.Float32frombits(y)}, true
}

func (enc *pathEncoder) MoveTo(x, y float32) {
	if enc.isFill {
		enc.Close()
	}
	if enc.state == pathStateMoveTo {
		*enc.data = (*enc.data)[:len(*enc.data)-8]
	} else if enc.state == pathStateNonemptySubpath {
		if !enc.isFill {
			enc.insertStrokeCapMarkerSegment(false)
		}
		if len(*enc.tags) != 0 {
			(*enc.tags)[len(*enc.tags)-1].setSubpathEnd()
		}
	}
	enc.firstPoint = [2]float32{x, y}
	var bytes [8]byte
	binary.LittleEndian.PutUint32(bytes[0:], math.Float32bits(x))
	binary.LittleEndian.PutUint32(bytes[4:], math.Float32bits(y))
	*enc.data = append(*enc.data, bytes[:]...)
	enc.state = pathStateMoveTo
}

func (enc *pathEncoder) isZeroLengthSegment(p1 [2]float32, p2_, p3_ *[2]float32) bool {
	p0, ok := enc.lastPoint()
	if !ok {
		panic("unreachable")
	}
	p2 := p1
	p3 := p1
	if p2_ != nil {
		p2 = *p2_
	}
	if p3_ != nil {
		p3 = *p3_
	}

	xMin := min(p0[0], p1[0], p2[0], p3[0])
	xMax := max(p0[0], p1[0], p2[0], p3[0])
	yMin := min(p0[1], p1[1], p2[1], p3[1])
	yMax := max(p0[1], p1[1], p2[1], p3[1])

	return !(xMax-xMin > jmath.Epsilon || yMax-yMin > jmath.Epsilon)
}

func (enc *pathEncoder) startTangentForCurve(p1 [2]float32, p2_, p3_ *[2]float32) ([2]float32, bool) {
	p0 := [2]float32{enc.firstPoint[0], enc.firstPoint[1]}
	p2 := p0
	p3 := p0
	if p2_ != nil {
		p2 = *p2_
	}
	if p3_ != nil {
		p3 = *p3_
	}

	if jmath.Abs32(p1[0]-p0[0]) > jmath.Epsilon || jmath.Abs32(p1[1]-p0[1]) > jmath.Epsilon {
		return p1, true
	} else if jmath.Abs32(p2[0]-p0[0]) > jmath.Epsilon || jmath.Abs32(p2[1]-p0[1]) > jmath.Epsilon {
		return p2, true
	} else if jmath.Abs32(p3[0]-p0[0]) > jmath.Epsilon || jmath.Abs32(p3[1]-p0[1]) > jmath.Epsilon {
		return p3, true
	} else {
		return [2]float32{}, false
	}
}

func (enc *pathEncoder) LineTo(x, y float32) {
	if enc.state == pathStateStart {
		if enc.numEncodedSegments == 0 {
			// This copies the behavior of kurbo which treats an initial line, quad
			// or curve as a move.
			enc.MoveTo(x, y)
			return
		}
		enc.MoveTo(enc.firstPoint[0], enc.firstPoint[1])
	}
	if enc.state == pathStateMoveTo {
		// Ensure that we don't end up with a zero-length start tangent.
		if pt, ok := enc.startTangentForCurve(
			[2]float32{x, y},
			nil,
			nil,
		); ok {
			enc.firstStartTangentEnd = pt
		} else {
			return
		}
	}
	// Drop the segment if its length is zero
	if enc.isZeroLengthSegment([2]float32{x, y}, nil, nil) {
		return
	}
	var bytes [8]byte
	binary.LittleEndian.PutUint32(bytes[0:], math.Float32bits(x))
	binary.LittleEndian.PutUint32(bytes[4:], math.Float32bits(y))
	*enc.data = append(*enc.data, bytes[:]...)
	*enc.tags = append(*enc.tags, PathTagLineToF32)
	enc.state = pathStateNonemptySubpath
	enc.numEncodedSegments++
}

func (enc *pathEncoder) QuadTo(x1, y1, x2, y2 float32) {
	if enc.state == pathStateStart {
		if enc.numEncodedSegments == 0 {
			enc.MoveTo(x2, y2)
			return
		}
		enc.MoveTo(enc.firstPoint[0], enc.firstPoint[1])
	}
	if enc.state == pathStateMoveTo {
		// Ensure that we don't end up with a zero-length start tangent.
		xy, ok := enc.startTangentForCurve([2]float32{x1, y1}, &[2]float32{x2, y2}, &[2]float32{})
		if !ok {
			return
		}
		enc.firstStartTangentEnd = xy
	}
	// Drop the segment if its length is zero
	if enc.isZeroLengthSegment([2]float32{x1, y1}, &[2]float32{x2, y2}, nil) {
		return
	}
	var buf [16]byte
	binary.LittleEndian.PutUint32(buf[0:], math.Float32bits(x1))
	binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(y1))
	binary.LittleEndian.PutUint32(buf[8:], math.Float32bits(x2))
	binary.LittleEndian.PutUint32(buf[12:], math.Float32bits(y2))
	*enc.data = append(*enc.data, buf[:]...)
	*enc.tags = append(*enc.tags, PathTagQuadToF32)
	enc.state = pathStateNonemptySubpath
	enc.numEncodedSegments++
}

func (enc *pathEncoder) CubicTo(x1, y1, x2, y2, x3, y3 float32) {
	if enc.state == pathStateStart {
		if enc.numEncodedSegments == 0 {
			enc.MoveTo(x3, y3)
			return
		}
		enc.MoveTo(enc.firstPoint[0], enc.firstPoint[1])
	}
	if enc.state == pathStateMoveTo {
		// Ensure that we don't end up with a zero-length start tangent.
		xy, ok := enc.startTangentForCurve([2]float32{x1, y1}, &[2]float32{x2, y2}, &[2]float32{x3, y3})
		if !ok {
			return
		}
		enc.firstStartTangentEnd = xy
	}
	// Drop the segment if its length is zero
	if enc.isZeroLengthSegment([2]float32{x1, y1}, &[2]float32{x2, y2}, &[2]float32{x3, y3}) {
		return
	}
	var buf [24]byte
	binary.LittleEndian.PutUint32(buf[0:], math.Float32bits(x1))
	binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(y1))
	binary.LittleEndian.PutUint32(buf[8:], math.Float32bits(x2))
	binary.LittleEndian.PutUint32(buf[12:], math.Float32bits(y2))
	binary.LittleEndian.PutUint32(buf[16:], math.Float32bits(x3))
	binary.LittleEndian.PutUint32(buf[20:], math.Float32bits(y3))
	*enc.data = append(*enc.data, buf[:]...)
	*enc.tags = append(*enc.tags, PathTagCubicToF32)
	enc.state = pathStateNonemptySubpath
	enc.numEncodedSegments++
}

func (enc *pathEncoder) Close() {
	switch enc.state {
	case pathStateStart:
		return
	case pathStateMoveTo:
		*enc.data = (*enc.data)[:len(*enc.data)-8]
		enc.state = pathStateStart
		return
	}
	if len(*enc.data) < 8 {
		// can't happen
		return
	}
	var firstBytes [8]byte
	binary.LittleEndian.PutUint32(firstBytes[0:], math.Float32bits(enc.firstPoint[0]))
	binary.LittleEndian.PutUint32(firstBytes[4:], math.Float32bits(enc.firstPoint[1]))
	if ([8]byte)((*enc.data)[len(*enc.data)-8:]) != firstBytes {
		*enc.data = append(*enc.data, firstBytes[:]...)
		*enc.tags = append(*enc.tags, PathTagLineToF32)
		enc.numEncodedSegments++
	}
	if !enc.isFill {
		enc.insertStrokeCapMarkerSegment(true)
	}
	if len(*enc.tags) > 0 {
		(*enc.tags)[len(*enc.tags)-1].setSubpathEnd()
	}
	enc.state = pathStateStart
}

func (enc *pathEncoder) PathElements(path iter.Seq[curve.PathElement]) {
	for el := range path {
		switch el.Kind {
		case curve.MoveToKind:
			enc.MoveTo(float32(el.P0.X), float32(el.P0.Y))
		case curve.LineToKind:
			enc.LineTo(float32(el.P0.X), float32(el.P0.Y))
		case curve.QuadToKind:
			p0 := el.P0
			p1 := el.P1
			enc.QuadTo(float32(p0.X), float32(p0.Y), float32(p1.X), float32(p1.Y))
		case curve.CubicToKind:
			p0 := el.P0
			p1 := el.P1
			p2 := el.P2
			enc.CubicTo(
				float32(p0.X),
				float32(p0.Y),
				float32(p1.X),
				float32(p1.Y),
				float32(p2.X),
				float32(p2.Y),
			)
		case curve.ClosePathKind:
			enc.Close()
		}
	}
}

func (enc *pathEncoder) Finish(insertPathMarker bool) uint32 {
	if enc.isFill {
		enc.Close()
	}
	if enc.state == pathStateMoveTo {
		*enc.data = (*enc.data)[:len(*enc.data)-8]
	}
	if enc.numEncodedSegments != 0 {
		if !enc.isFill && enc.state == pathStateNonemptySubpath {
			enc.insertStrokeCapMarkerSegment(false)
		}
		if len(*enc.tags) > 0 {
			(*enc.tags)[len(*enc.tags)-1].setSubpathEnd()
		}
		*enc.numSegments += enc.numEncodedSegments
		if insertPathMarker {
			*enc.tags = append(*enc.tags, PathTagPath)
			*enc.numPaths += 1
		}
	}
	return enc.numEncodedSegments
}

func (enc *pathEncoder) insertStrokeCapMarkerSegment(isClosed bool) {
	if enc.isFill {
		panic("invalid state")
	}
	if enc.state != pathStateNonemptySubpath {
		panic("invalid state")
	}
	if isClosed {
		// We expect that the most recently encoded pair of coordinates in the path data stream
		// contain the first control point in the path segment (see `PathEncoder::close`).
		// Hence a line-to encoded here should embed the subpath's start tangent.
		enc.LineTo(
			enc.firstStartTangentEnd[0],
			enc.firstStartTangentEnd[1],
		)
	} else {
		enc.QuadTo(
			enc.firstPoint[0],
			enc.firstPoint[1],
			enc.firstStartTangentEnd[0],
			enc.firstStartTangentEnd[1],
		)
	}
}
