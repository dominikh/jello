package encoding

import (
	"encoding/binary"
	"iter"
	"math"
	"structs"

	"honnef.co/go/brush"
	"honnef.co/go/curve"
	"honnef.co/go/jello/jmath"
)

type Style struct {
	_ structs.HostLayout

	FlagsAndMiterLimits uint32
	LineWidth           float32
}

const (
	/// 0 for a fill, 1 for a stroke
	flagsStyleBit uint32 = 0x8000_0000

	/// 0 for non-zero, 1 for even-odd
	flagsFillBit uint32 = 0x4000_0000

	/// Encodings for join style:
	///    - 0b00 -> bevel
	///    - 0b01 -> miter
	///    - 0b10 -> round
	flagsJoinBitsBevel uint32 = 0
	flagsJoinBitsMiter uint32 = 0x1000_0000
	flagsJoinBitsRound uint32 = 0x2000_0000
	flagsJoinMask      uint32 = 0x3000_0000

	/// Encodings for cap style:
	///    - 0b00 -> butt
	///    - 0b01 -> square
	///    - 0b10 -> round
	flagsCapBitsButt   uint32 = 0
	flagsCapBitsSquare uint32 = 0x0100_0000
	flagsCapBitsRound  uint32 = 0x0200_0000

	flagsStartCapBitsButt   uint32 = flagsCapBitsButt << 2
	flagsStartCapBitsSquare uint32 = flagsCapBitsSquare << 2
	flagsStartCapBitsRound  uint32 = flagsCapBitsRound << 2
	flagsEndCapBitsButt     uint32 = flagsCapBitsButt
	flagsEndCapBitsSquare   uint32 = flagsCapBitsSquare
	flagsEndCapBitsRound    uint32 = flagsCapBitsRound

	flagsStartCapMask uint32 = 0x0C00_0000
	flagsEndCapMask   uint32 = 0x0300_0000
	miterLimitMask    uint32 = 0xFFFF
)

func StyleFromFill(fill brush.Fill) Style {
	var fillBit uint32
	if fill == brush.EvenOdd {
		fillBit = flagsFillBit
	}
	return Style{
		FlagsAndMiterLimits: fillBit,
		LineWidth:           0,
	}
}

func StyleFromStroke(stroke curve.Stroke) Style {
	style := flagsStyleBit
	var join uint32
	switch stroke.Join {
	case curve.BevelJoin:
		join = flagsJoinBitsBevel
	case curve.MiterJoin:
		join = flagsJoinBitsMiter
	case curve.RoundJoin:
		join = flagsJoinBitsRound
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

type PathSegmentType uint8

const (
	PathSegmentTypeLineTo  PathSegmentType = 0x1
	PathSegmentTypeQuadTo  PathSegmentType = 0x2
	PathSegmentTypeCubicTo PathSegmentType = 0x3
)

type PathTag uint8

const (
	/// 32-bit floating point line segment.
	///
	/// This is equivalent to `(PathSegmentType::LineTo | PathTag::F32_BIT)`.
	PathTagLineToF32 PathTag = 0x9

	/// 32-bit floating point quadratic segment.
	///
	/// This is equivalent to `(PathSegmentType::QUAD_TO | PathTag::F32_BIT)`.
	PathTagQuadToF32 PathTag = 0xa

	/// 32-bit floating point cubic segment.
	///
	/// This is equivalent to `(PathSegmentType::CUBIC_TO | PathTag::F32_BIT)`.
	PathTagCubicToF32 PathTag = 0xb

	/// 16-bit integral line segment.
	PathTagLineToI16 PathTag = 0x1

	/// 16-bit integral quadratic segment.
	PathTagQuadToI16 PathTag = 0x2

	/// 16-bit integral cubic segment.
	PathTagCubicToI16 PathTag = 0x3

	/// Transform marker.
	PathTagTransform PathTag = 0x20

	/// Path marker.
	PathTagPath PathTag = 0x10

	/// Style setting.
	PathTagStyle PathTag = 0x40

	/// Bit that marks a segment that is the end of a subpath.
	PathTagSubpathEndBit PathTag = 0x4

	/// Bit for path segments that are represented as f32 values. If unset
	/// they are represented as i16.
	PathTagF32Bit PathTag = 0x8

	/// Mask for bottom 3 bits that contain the [`PathSegmentType`].
	PathTagSegmentMask PathTag = 0x3
)

func (tag PathTag) IsPathSegment() bool { return tag.PathSegmentType() != 0 }
func (tag PathTag) IsFloat32() bool     { return tag&PathTagF32Bit != 0 }
func (tag PathTag) IsSubpathEnd() bool  { return tag&PathTagSubpathEndBit != 0 }
func (tag *PathTag) SetSubpathEnd()     { *tag |= PathTagSubpathEndBit }
func (tag PathTag) PathSegmentType() PathSegmentType {
	return PathSegmentType(tag & PathTagSegmentMask)
}

type PathEncoder struct {
	tags                 *[]PathTag
	data                 *[]byte
	numSegments          *uint32
	numPaths             *uint32
	firstPoint           [2]float32
	firstStartTangentEnd [2]float32
	state                PathState
	numEncodedSegments   uint32
	isFill               bool
}

type PathState int

const (
	PathStateStart PathState = iota
	PathStateMoveTo
	PathStateNonemptySubpath
)

func (enc *PathEncoder) lastPoint() ([2]float32, bool) {
	n := len(*enc.data)
	if n < 8 {
		return [2]float32{}, false
	}
	x := binary.LittleEndian.Uint32((*enc.data)[n-8 : n-4])
	y := binary.LittleEndian.Uint32((*enc.data)[n-4 : n])
	return [2]float32{math.Float32frombits(x), math.Float32frombits(y)}, true
}

func (enc *PathEncoder) MoveTo(x, y float32) {
	if enc.isFill {
		enc.Close()
	}
	if enc.state == PathStateMoveTo {
		*enc.data = (*enc.data)[:len(*enc.data)-8]
	} else if enc.state == PathStateNonemptySubpath {
		if !enc.isFill {
			enc.insertStrokeCapMarkerSegment(false)
		}
		if len(*enc.tags) != 0 {
			(*enc.tags)[len(*enc.tags)-1].SetSubpathEnd()
		}
	}
	enc.firstPoint = [2]float32{x, y}
	var bytes [8]byte
	binary.LittleEndian.PutUint32(bytes[0:], math.Float32bits(x))
	binary.LittleEndian.PutUint32(bytes[4:], math.Float32bits(y))
	*enc.data = append(*enc.data, bytes[:]...)
	enc.state = PathStateMoveTo
}

func (enc *PathEncoder) isZeroLengthSegment(p1 [2]float32, p2_, p3_ *[2]float32) bool {
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

func (enc *PathEncoder) startTangentForCurve(p1 [2]float32, p2_, p3_ *[2]float32) ([2]float32, bool) {
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

func (enc *PathEncoder) LineTo(x, y float32) {
	if enc.state == PathStateStart {
		if enc.numEncodedSegments == 0 {
			// This copies the behavior of kurbo which treats an initial line, quad
			// or curve as a move.
			enc.MoveTo(x, y)
			return
		}
		enc.MoveTo(enc.firstPoint[0], enc.firstPoint[1])
	}
	if enc.state == PathStateMoveTo {
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
	enc.state = PathStateNonemptySubpath
	enc.numEncodedSegments++
}

func (enc *PathEncoder) QuadTo(x1, y1, x2, y2 float32) {
	if enc.state == PathStateStart {
		if enc.numEncodedSegments == 0 {
			enc.MoveTo(x2, y2)
			return
		}
		enc.MoveTo(enc.firstPoint[0], enc.firstPoint[1])
	}
	if enc.state == PathStateMoveTo {
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
	enc.state = PathStateNonemptySubpath
	enc.numEncodedSegments++
}

func (enc *PathEncoder) CubicTo(x1, y1, x2, y2, x3, y3 float32) {
	if enc.state == PathStateStart {
		if enc.numEncodedSegments == 0 {
			enc.MoveTo(x3, y3)
			return
		}
		enc.MoveTo(enc.firstPoint[0], enc.firstPoint[1])
	}
	if enc.state == PathStateMoveTo {
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
	enc.state = PathStateNonemptySubpath
	enc.numEncodedSegments++
}

func (enc *PathEncoder) Close() {
	switch enc.state {
	case PathStateStart:
		return
	case PathStateMoveTo:
		*enc.data = (*enc.data)[:len(*enc.data)-8]
		enc.state = PathStateStart
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
		(*enc.tags)[len(*enc.tags)-1].SetSubpathEnd()
	}
	enc.state = PathStateStart
}

func (enc *PathEncoder) Shape(shape curve.Shape) {
	enc.PathElements(shape.PathElements(0.1))
}

func (enc *PathEncoder) PathElements(path iter.Seq[curve.PathElement]) {
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

func (enc *PathEncoder) Finish(insertPathMarker bool) uint32 {
	if enc.isFill {
		enc.Close()
	}
	if enc.state == PathStateMoveTo {
		*enc.data = (*enc.data)[:len(*enc.data)-8]
	}
	if enc.numEncodedSegments != 0 {
		if !enc.isFill && enc.state == PathStateNonemptySubpath {
			enc.insertStrokeCapMarkerSegment(false)
		}
		if len(*enc.tags) > 0 {
			(*enc.tags)[len(*enc.tags)-1].SetSubpathEnd()
		}
		*enc.numSegments += enc.numEncodedSegments
		if insertPathMarker {
			*enc.tags = append(*enc.tags, PathTagPath)
			*enc.numPaths += 1
		}
	}
	return enc.numEncodedSegments
}

func (enc *PathEncoder) insertStrokeCapMarkerSegment(isClosed bool) {
	if enc.isFill {
		panic("invalid state")
	}
	if enc.state != PathStateNonemptySubpath {
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
