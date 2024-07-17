package renderer

import "structs"

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
