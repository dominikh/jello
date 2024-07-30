package renderer

import (
	"structs"

	"honnef.co/go/jello/encoding"
)

type DrawMonoid struct {
	_ structs.HostLayout

	// The number of paths preceding this draw object.
	PathIdx uint32
	// The number of clip operations preceding this draw object.
	ClipIdx uint32
	// The offset of the encoded draw object in the scene (u32s).
	SceneOffset uint32
	// The offset of the associated info.
	InfoOffset uint32
}

func NewDrawMonoid(tag encoding.DrawTag) DrawMonoid {
	var pathIdx uint32
	if tag != encoding.DrawTagNop {
		pathIdx = 1
	}
	return DrawMonoid{
		PathIdx:     pathIdx,
		ClipIdx:     uint32(tag) & 1,
		SceneOffset: (uint32(tag) >> 2) & 0x7,
		InfoOffset:  (uint32(tag) >> 6) & 0xf,
	}
}

func (m DrawMonoid) Combine(other DrawMonoid) DrawMonoid {
	return DrawMonoid{
		PathIdx:     m.PathIdx + other.PathIdx,
		ClipIdx:     m.ClipIdx + other.ClipIdx,
		SceneOffset: m.SceneOffset + other.SceneOffset,
		InfoOffset:  m.InfoOffset + other.InfoOffset,
	}
}

type DrawBbox struct {
	_ structs.HostLayout

	Bbox [4]float32
}
