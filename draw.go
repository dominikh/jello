package jello

import "structs"

type DrawTag uint32

const (
	/// No operation.
	NOP DrawTag = 0

	/// Color fill.
	COLOR DrawTag = 0x44

	/// Linear gradient fill.
	LINEAR_GRADIENT DrawTag = 0x114

	/// Radial gradient fill.
	RADIAL_GRADIENT DrawTag = 0x29c

	/// Sweep gradient fill.
	SWEEP_GRADIENT DrawTag = 0x254

	/// Image fill.
	IMAGE DrawTag = 0x248

	/// Begin layer/clip.
	BEGIN_CLIP DrawTag = 0x9

	/// End layer/clip.
	END_CLIP DrawTag = 0x21
)

func (tag DrawTag) InfoSize() uint32 {
	return uint32((tag >> 6) & 0xf)
}

type DrawColor struct {
	_ structs.HostLayout

	RGBA uint32
}

type DrawMonoid struct {
	_ structs.HostLayout

	// The number of paths preceding this draw object.
	Path_ix uint32
	// The number of clip operations preceding this draw object.
	Clip_ix uint32
	// The offset of the encoded draw object in the scene (u32s).
	Scene_offset uint32
	// The offset of the associated info.
	Info_offset uint32
}

type DrawBbox struct {
	_ structs.HostLayout

	Bbox [4]float32
}
