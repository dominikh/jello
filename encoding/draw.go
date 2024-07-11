package encoding

import "structs"

type DrawTag uint32

const (
	/// No operation.
	DrawTagNop DrawTag = 0

	/// Color fill.
	DrawTagColor DrawTag = 0x44

	/// Linear gradient fill.
	DrawTagLinearGradient DrawTag = 0x114

	/// Radial gradient fill.
	DrawTagRadialGradient DrawTag = 0x29c

	/// Sweep gradient fill.
	DrawTagSweepGradient DrawTag = 0x254

	/// Image fill.
	DrawTagImage DrawTag = 0x248

	/// Begin layer/clip.
	DrawTagBeginClip DrawTag = 0x9

	/// End layer/clip.
	DrawTagEndClip DrawTag = 0x21
)

func (tag DrawTag) InfoSize() uint32 {
	return uint32((tag >> 6) & 0xf)
}

type drawColor struct {
	_ structs.HostLayout

	RGBA uint32
}
