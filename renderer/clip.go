package renderer

import "structs"

type Clip struct {
	_ structs.HostLayout

	// Index of the draw object.
	Idx uint32
	/// This is a packed encoding of an enum with the sign bit as the tag. If positive,
	/// this entry is a `BeginClip` and contains the associated path index. If negative,
	/// it is an `EndClip` and contains the bitwise-not of the `EndClip` draw object index.
	PathIdx int32
}

type ClipBbox struct {
	_ structs.HostLayout

	Bbox [4]float32
}

type ClipBic struct {
	_ structs.HostLayout

	/// When interpreted as a stack operation, the number of pop operations.
	A uint32
	/// When interpreted as a stack operation, the number of push operations.
	B uint32
}

func (cb ClipBic) Combine(other ClipBic) ClipBic {
	m := min(cb.B, other.A)
	return ClipBic{A: cb.A + other.A - m, B: cb.B + other.B - m}
}

type ClipElement struct {
	_ structs.HostLayout

	ParentIdx uint32
	_         [12]uint8
	Bbox      [4]float32
}

type BinHeader struct {
	_ structs.HostLayout

	ElementCount uint32
	ChunkOffset  uint32
}
