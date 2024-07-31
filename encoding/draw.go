// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package encoding

import (
	"structs"

	"honnef.co/go/jello/gfx"
)

type DrawTag uint32

const (
	// No operation.
	DrawTagNop DrawTag = 0

	// Color fill.
	DrawTagColor DrawTag = 0x44

	// Linear gradient fill.
	DrawTagLinearGradient DrawTag = 0x114

	// Radial gradient fill.
	DrawTagRadialGradient DrawTag = 0x29c

	// Sweep gradient fill.
	DrawTagSweepGradient DrawTag = 0x254

	// Image fill.
	DrawTagImage DrawTag = 0x248

	// Begin layer/clip.
	DrawTagBeginClip DrawTag = 0x9

	// End layer/clip.
	DrawTagEndClip DrawTag = 0x21
)

func (tag DrawTag) InfoSize() uint32 {
	return uint32((tag >> 6) & 0xf)
}

type drawColor struct {
	_ structs.HostLayout

	RGBA uint32
}

func newDrawColor(color gfx.Color) drawColor {
	if color == nil {
		return drawColor{}
	}
	return drawColor{RGBA: color.LinearSRGB().PremulUint32()}
}

type drawLinearGradient struct {
	_ structs.HostLayout

	Index uint32
	P0    [2]float32
	P1    [2]float32
}

type drawRadialGradient struct {
	_ structs.HostLayout

	Index uint32
	P0    [2]float32
	P1    [2]float32
	R0    float32
	R1    float32
}

type drawSweepGradient struct {
	_ structs.HostLayout

	Index uint32
	P0    [2]float32
	T0    float32
	T1    float32
}

type drawImage struct {
	_ structs.HostLayout

	index       uint32
	widthHeight uint32
}
