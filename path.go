package jello

import (
	"encoding/binary"
	"iter"
	"math"
	"structs"

	"honnef.co/go/brush"
	"honnef.co/go/curve"
)

type Style struct {
	_ structs.HostLayout

	FlagsAndMiterLimits uint32
	LineWidth           float32
}

const (
	/// 0 for a fill, 1 for a stroke
	FLAGS_STYLE_BIT uint32 = 0x8000_0000

	/// 0 for non-zero, 1 for even-odd
	FLAGS_FILL_BIT uint32 = 0x4000_0000

	/// Encodings for join style:
	///    - 0b00 -> bevel
	///    - 0b01 -> miter
	///    - 0b10 -> round
	FLAGS_JOIN_BITS_BEVEL uint32 = 0
	FLAGS_JOIN_BITS_MITER uint32 = 0x1000_0000
	FLAGS_JOIN_BITS_ROUND uint32 = 0x2000_0000
	FLAGS_JOIN_MASK       uint32 = 0x3000_0000

	/// Encodings for cap style:
	///    - 0b00 -> butt
	///    - 0b01 -> square
	///    - 0b10 -> round
	FLAGS_CAP_BITS_BUTT   uint32 = 0
	FLAGS_CAP_BITS_SQUARE uint32 = 0x0100_0000
	FLAGS_CAP_BITS_ROUND  uint32 = 0x0200_0000

	FLAGS_START_CAP_BITS_BUTT   uint32 = FLAGS_CAP_BITS_BUTT << 2
	FLAGS_START_CAP_BITS_SQUARE uint32 = FLAGS_CAP_BITS_SQUARE << 2
	FLAGS_START_CAP_BITS_ROUND  uint32 = FLAGS_CAP_BITS_ROUND << 2
	FLAGS_END_CAP_BITS_BUTT     uint32 = FLAGS_CAP_BITS_BUTT
	FLAGS_END_CAP_BITS_SQUARE   uint32 = FLAGS_CAP_BITS_SQUARE
	FLAGS_END_CAP_BITS_ROUND    uint32 = FLAGS_CAP_BITS_ROUND

	FLAGS_START_CAP_MASK uint32 = 0x0C00_0000
	FLAGS_END_CAP_MASK   uint32 = 0x0300_0000
	MITER_LIMIT_MASK     uint32 = 0xFFFF
)

func StyleFromFill(fill brush.Fill) Style {
	var fill_bit uint32
	if fill == brush.EvenOdd {
		fill_bit = FLAGS_FILL_BIT
	}
	return Style{
		FlagsAndMiterLimits: fill_bit,
		LineWidth:           0,
	}
}

func StyleFromStroke(stroke curve.Stroke) Style {
	style := FLAGS_STYLE_BIT
	var join uint32
	switch stroke.Join {
	case curve.BevelJoin:
		join = FLAGS_JOIN_BITS_BEVEL
	case curve.MiterJoin:
		join = FLAGS_JOIN_BITS_MITER
	case curve.RoundJoin:
		join = FLAGS_JOIN_BITS_ROUND
	}
	var start_cap uint32
	switch stroke.StartCap {
	case curve.ButtCap:
		start_cap = FLAGS_START_CAP_BITS_BUTT
	case curve.SquareCap:
		start_cap = FLAGS_START_CAP_BITS_SQUARE
	case curve.RoundCap:
		start_cap = FLAGS_START_CAP_BITS_ROUND
	}
	var end_cap uint32
	switch stroke.EndCap {
	case curve.ButtCap:
		end_cap = FLAGS_END_CAP_BITS_BUTT
	case curve.SquareCap:
		end_cap = FLAGS_END_CAP_BITS_SQUARE
	case curve.RoundCap:
		end_cap = FLAGS_END_CAP_BITS_ROUND
	}
	miter_limit := uint32(f32_to_f16(float32(stroke.MiterLimit)))
	return Style{
		FlagsAndMiterLimits: style | join | start_cap | end_cap | miter_limit,
		LineWidth:           float32(stroke.Width),
	}
}

type PathSegmentType uint8

const (
	LINE_TO  PathSegmentType = 0x1
	QUAD_TO  PathSegmentType = 0x2
	CUBIC_TO PathSegmentType = 0x3
)

type PathTag uint8

const (
	/// 32-bit floating point line segment.
	///
	/// This is equivalent to `(PathSegmentType::LINE_TO | PathTag::F32_BIT)`.
	LINE_TO_F32 PathTag = 0x9

	/// 32-bit floating point quadratic segment.
	///
	/// This is equivalent to `(PathSegmentType::QUAD_TO | PathTag::F32_BIT)`.
	QUAD_TO_F32 PathTag = 0xa

	/// 32-bit floating point cubic segment.
	///
	/// This is equivalent to `(PathSegmentType::CUBIC_TO | PathTag::F32_BIT)`.
	CUBIC_TO_F32 PathTag = 0xb

	/// 16-bit integral line segment.
	LINE_TO_I16 PathTag = 0x1

	/// 16-bit integral quadratic segment.
	QUAD_TO_I16 PathTag = 0x2

	/// 16-bit integral cubic segment.
	CUBIC_TO_I16 PathTag = 0x3

	/// Transform marker.
	TRANSFORM PathTag = 0x20

	/// Path marker.
	PATH PathTag = 0x10

	/// Style setting.
	STYLE PathTag = 0x40

	/// Bit that marks a segment that is the end of a subpath.
	SUBPATH_END_BIT PathTag = 0x4

	/// Bit for path segments that are represented as f32 values. If unset
	/// they are represented as i16.
	_F32_BIT PathTag = 0x8

	/// Mask for bottom 3 bits that contain the [`PathSegmentType`].
	_SEGMENT_MASK PathTag = 0x3
)

func (tag PathTag) IsPathSegment() bool              { return tag.PathSegmentType() != 0 }
func (tag PathTag) IsFloat32() bool                  { return tag&_F32_BIT != 0 }
func (tag PathTag) IsSubpathEnd() bool               { return tag&SUBPATH_END_BIT != 0 }
func (tag *PathTag) SetSubpathEnd()                  { *tag |= SUBPATH_END_BIT }
func (tag PathTag) PathSegmentType() PathSegmentType { return PathSegmentType(tag & _SEGMENT_MASK) }

type PathMonoid struct {
	_ structs.HostLayout

	/// Index into transform stream.
	Trans_ix uint32
	/// Path segment index.
	Pathseg_ix uint32
	/// Offset into path segment stream.
	Pathseg_offset uint32
	/// Index into style stream.
	Style_ix uint32
	/// Index of containing path.
	Path_ix uint32
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

func (self *PathEncoder) lastPoint() ([2]float32, bool) {
	n := len(*self.data)
	if n < 8 {
		return [2]float32{}, false
	}
	x := binary.LittleEndian.Uint32((*self.data)[n-8 : n-4])
	y := binary.LittleEndian.Uint32((*self.data)[n-4 : n])
	return [2]float32{math.Float32frombits(x), math.Float32frombits(y)}, true
}

func (self *PathEncoder) MoveTo(x, y float32) {
	if self.isFill {
		self.Close()
	}
	if self.state == PathStateMoveTo {
		*self.data = (*self.data)[:len(*self.data)-8]
	} else if self.state == PathStateNonemptySubpath {
		if !self.isFill {
			self.insertStrokeCapMarkerSegment(false)
		}
		if len(*self.tags) != 0 {
			(*self.tags)[len(*self.tags)-1].SetSubpathEnd()
		}
	}
	self.firstPoint = [2]float32{x, y}
	var bytes [8]byte
	binary.LittleEndian.PutUint32(bytes[0:], math.Float32bits(x))
	binary.LittleEndian.PutUint32(bytes[4:], math.Float32bits(y))
	*self.data = append(*self.data, bytes[:]...)
	self.state = PathStateMoveTo
}

func (self *PathEncoder) isZeroLengthSegment(p1 [2]float32, p2_, p3_ option[[2]float32]) bool {
	p0, ok := self.lastPoint()
	if !ok {
		panic("unreachable")
	}
	p2 := p2_.unwrapOr(p1)
	p3 := p3_.unwrapOr(p1)

	x_min := min(p0[0], p1[0], p2[0], p3[0])
	x_max := max(p0[0], p1[0], p2[0], p3[0])
	y_min := min(p0[1], p1[1], p2[1], p3[1])
	y_max := max(p0[1], p1[1], p2[1], p3[1])

	return !(x_max-x_min > epsilon || y_max-y_min > epsilon)
}

func (self *PathEncoder) startTangentForCurve(p1 [2]float32, p2_, p3_ option[[2]float32]) ([2]float32, bool) {
	p0 := [2]float32{self.firstPoint[0], self.firstPoint[1]}
	p2 := p2_.unwrapOr(p0)
	p3 := p3_.unwrapOr(p0)
	if abs32(p1[0]-p0[0]) > epsilon || abs32(p1[1]-p0[1]) > epsilon {
		return p1, true
	} else if abs32(p2[0]-p0[0]) > epsilon || abs32(p2[1]-p0[1]) > epsilon {
		return p2, true
	} else if abs32(p3[0]-p0[0]) > epsilon || abs32(p3[1]-p0[1]) > epsilon {
		return p3, true
	} else {
		return [2]float32{}, false
	}
}

func (self *PathEncoder) LineTo(x, y float32) {
	if self.state == PathStateStart {
		if self.numEncodedSegments == 0 {
			// This copies the behavior of kurbo which treats an initial line, quad
			// or curve as a move.
			self.MoveTo(x, y)
			return
		}
		self.MoveTo(self.firstPoint[0], self.firstPoint[1])
	}
	if self.state == PathStateMoveTo {
		// Ensure that we don't end up with a zero-length start tangent.
		if pt, ok := self.startTangentForCurve(
			[2]float32{x, y},
			option[[2]float32]{},
			option[[2]float32]{},
		); ok {
			self.firstStartTangentEnd = pt
		} else {
			return
		}
	}
	// Drop the segment if its length is zero
	if self.isZeroLengthSegment([2]float32{x, y}, option[[2]float32]{}, option[[2]float32]{}) {
		return
	}
	var bytes [8]byte
	binary.LittleEndian.PutUint32(bytes[0:], math.Float32bits(x))
	binary.LittleEndian.PutUint32(bytes[4:], math.Float32bits(y))
	*self.data = append(*self.data, bytes[:]...)
	*self.tags = append(*self.tags, LINE_TO_F32)
	self.state = PathStateNonemptySubpath
	self.numEncodedSegments++
}

func (self *PathEncoder) QuadTo(x1, y1, x2, y2 float32) {
	if self.state == PathStateStart {
		if self.numEncodedSegments == 0 {
			self.MoveTo(x2, y2)
			return
		}
		self.MoveTo(self.firstPoint[0], self.firstPoint[1])
	}
	if self.state == PathStateMoveTo {
		// Ensure that we don't end up with a zero-length start tangent.
		xy, ok := self.startTangentForCurve([2]float32{x1, y1}, some([2]float32{x2, y2}), option[[2]float32]{})
		if !ok {
			return
		}
		self.firstStartTangentEnd = xy
	}
	// Drop the segment if its length is zero
	if self.isZeroLengthSegment([2]float32{x1, y1}, some([2]float32{x2, y2}), option[[2]float32]{}) {
		return
	}
	var buf [16]byte
	binary.LittleEndian.PutUint32(buf[0:], math.Float32bits(x1))
	binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(y1))
	binary.LittleEndian.PutUint32(buf[8:], math.Float32bits(x2))
	binary.LittleEndian.PutUint32(buf[12:], math.Float32bits(y2))
	*self.data = append(*self.data, buf[:]...)
	*self.tags = append(*self.tags, QUAD_TO_F32)
	self.state = PathStateNonemptySubpath
	self.numEncodedSegments++
}

func (self *PathEncoder) CubicTo(x1, y1, x2, y2, x3, y3 float32) {
	if self.state == PathStateStart {
		if self.numEncodedSegments == 0 {
			self.MoveTo(x3, y3)
			return
		}
		self.MoveTo(self.firstPoint[0], self.firstPoint[1])
	}
	if self.state == PathStateMoveTo {
		// Ensure that we don't end up with a zero-length start tangent.
		xy, ok := self.startTangentForCurve([2]float32{x1, y1}, some([2]float32{x2, y2}), some([2]float32{x3, y3}))
		if !ok {
			return
		}
		self.firstStartTangentEnd = xy
	}
	// Drop the segment if its length is zero
	if self.isZeroLengthSegment([2]float32{x1, y1}, some([2]float32{x2, y2}), some([2]float32{x3, y3})) {
		return
	}
	var buf [24]byte
	binary.LittleEndian.PutUint32(buf[0:], math.Float32bits(x1))
	binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(y1))
	binary.LittleEndian.PutUint32(buf[8:], math.Float32bits(x2))
	binary.LittleEndian.PutUint32(buf[12:], math.Float32bits(y2))
	binary.LittleEndian.PutUint32(buf[16:], math.Float32bits(x3))
	binary.LittleEndian.PutUint32(buf[20:], math.Float32bits(y3))
	*self.data = append(*self.data, buf[:]...)
	*self.tags = append(*self.tags, CUBIC_TO_F32)
	self.state = PathStateNonemptySubpath
	self.numEncodedSegments++
}

func (self *PathEncoder) Close() {
	switch self.state {
	case PathStateStart:
		return
	case PathStateMoveTo:
		*self.data = (*self.data)[:len(*self.data)-8]
		self.state = PathStateStart
		return
	}
	if len(*self.data) < 8 {
		// can't happen
		return
	}
	var first_bytes [8]byte
	binary.LittleEndian.PutUint32(first_bytes[0:], math.Float32bits(self.firstPoint[0]))
	binary.LittleEndian.PutUint32(first_bytes[4:], math.Float32bits(self.firstPoint[1]))
	if ([8]byte)((*self.data)[len(*self.data)-8:]) != first_bytes {
		*self.data = append(*self.data, first_bytes[:]...)
		*self.tags = append(*self.tags, LINE_TO_F32)
		self.numEncodedSegments++
	}
	if !self.isFill {
		self.insertStrokeCapMarkerSegment(true)
	}
	if len(*self.tags) > 0 {
		(*self.tags)[len(*self.tags)-1].SetSubpathEnd()
	}
	self.state = PathStateStart
}

func (self *PathEncoder) Shape(shape curve.Shape) {
	self.PathElements(shape.PathElements(0.1))
}

func (self *PathEncoder) PathElements(path iter.Seq[curve.PathElement]) {
	for el := range path {
		switch el.Kind {
		case curve.MoveToKind:
			self.MoveTo(float32(el.P0.X), float32(el.P0.Y))
		case curve.LineToKind:
			self.LineTo(float32(el.P0.X), float32(el.P0.Y))
		case curve.QuadToKind:
			p0 := el.P0
			p1 := el.P1
			self.QuadTo(float32(p0.X), float32(p0.Y), float32(p1.X), float32(p1.Y))
		case curve.CubicToKind:
			p0 := el.P0
			p1 := el.P1
			p2 := el.P2
			self.CubicTo(
				float32(p0.X),
				float32(p0.Y),
				float32(p1.X),
				float32(p1.Y),
				float32(p2.X),
				float32(p2.Y),
			)
		case curve.ClosePathKind:
			self.Close()
		}
	}
}

func (self *PathEncoder) Finish(insertPathMarker bool) uint32 {
	if self.isFill {
		self.Close()
	}
	if self.state == PathStateMoveTo {
		*self.data = (*self.data)[:len(*self.data)-8]
	}
	if self.numEncodedSegments != 0 {
		if !self.isFill && self.state == PathStateNonemptySubpath {
			self.insertStrokeCapMarkerSegment(false)
		}
		if len(*self.tags) > 0 {
			(*self.tags)[len(*self.tags)-1].SetSubpathEnd()
		}
		*self.numSegments += self.numEncodedSegments
		if insertPathMarker {
			*self.tags = append(*self.tags, PATH)
			*self.numPaths += 1
		}
	}
	return self.numEncodedSegments
}

func (self *PathEncoder) insertStrokeCapMarkerSegment(isClosed bool) {
	if self.isFill {
		panic("invalid state")
	}
	if self.state != PathStateNonemptySubpath {
		panic("invalid state")
	}
	if isClosed {
		// We expect that the most recently encoded pair of coordinates in the path data stream
		// contain the first control point in the path segment (see `PathEncoder::close`).
		// Hence a line-to encoded here should embed the subpath's start tangent.
		self.LineTo(
			self.firstStartTangentEnd[0],
			self.firstStartTangentEnd[1],
		)
	} else {
		self.QuadTo(
			self.firstPoint[0],
			self.firstPoint[1],
			self.firstStartTangentEnd[0],
			self.firstStartTangentEnd[1],
		)
	}
}

type Tile struct {
	_ structs.HostLayout

	Backdrop          int32
	SegmentCountOrIdx uint32
}

type LineSoup struct {
	_ structs.HostLayout

	PathIdx uint32
	_       uint32 // padding
	P0      [2]float32
	P1      [2]float32
}

type SegmentCount struct {
	_ structs.HostLayout

	LineIdx uint32
	Counts  uint32
}

type PathSegment struct {
	_ structs.HostLayout

	Point0 [2]float32
	Point1 [2]float32
	YEdge  float32
	_      uint32 // padding
}

type Path struct {
	_ structs.HostLayout

	Bbox  [4]uint32
	Tiles uint32
	_     [3]uint32
}

type PathBbox struct {
	_ structs.HostLayout

	/// Minimum x value.
	X0 int32
	/// Minimum y value.
	Y0 int32
	/// Maximum x value.
	X1 int32
	/// Maximum y value.
	Y1 int32
	/// Style flags
	Draw_flags uint32
	/// Index into the transform stream.
	Trans_ix uint32
}
