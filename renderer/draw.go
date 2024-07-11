package renderer

import "structs"

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

type DrawBbox struct {
	_ structs.HostLayout

	Bbox [4]float32
}
