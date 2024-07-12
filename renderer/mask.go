package renderer

import "encoding/binary"

//! Create a lookup table of half-plane sample masks.

// Width is number of discrete translations
const maskWidth = 32

// Height is the number of discrete slopes
const maskHeight = 32

var maskPattern = [...]uint8{0, 5, 3, 7, 1, 4, 6, 2}

func oneMask(slope float64, translation float64, isPos bool) uint8 {
	if isPos {
		translation = 1. - translation
	}
	var result uint8
	for i, item := range maskPattern {
		y := (float64(i) + 0.5) * 0.125
		x := (float64(item) + 0.5) * 0.125
		if !isPos {
			y = 1. - y
		}
		if (x-(1.0-translation))*(1.-slope)-(y-translation)*slope >= 0. {
			result |= 1 << i
		}
	}
	return result
}

var maskLUT8 = makeMaskLUT()
var maskLUT16 = makeMaskLUT16()

// / Make a lookup table of half-plane masks.
// /
// / The table is organized into two blocks each with `maskHeight/2` slopes.
// / The first block is negative slopes (x decreases as y increates),
// / the second as positive.
func makeMaskLUT() []uint8 {
	var out []uint8
	for i := range maskWidth * maskHeight {
		const halfHeight = maskHeight / 2
		u := i % maskWidth
		v := i / maskWidth
		isPos := v >= halfHeight
		y := (float64(v%halfHeight) + 0.5) * (1.0 / float64(halfHeight))
		x := (float64(u) + 0.5) * (1.0 / float64(maskWidth))
		out = append(out, oneMask(y, x, isPos))
	}
	return out
}

// Width is number of discrete translations
const mask16Width = 64

// Height is the number of discrete slopes
const mask16Height = 64

// This is based on the [D3D11 standard sample pattern].
//
// [D3D11 standard sample pattern]: https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_standard_multisample_quality_levels
var maskPattern16 = [...]uint8{1, 8, 4, 11, 15, 7, 3, 12, 0, 9, 5, 13, 2, 10, 6, 14}

func oneMask16(slope float64, translation float64, isPos bool) uint16 {
	if isPos {
		translation = 1. - translation
	}
	var result uint16
	for i, item := range maskPattern16 {
		y := (float64(i) + 0.5) * 0.0625
		x := (float64(item) + 0.5) * 0.0625
		if !isPos {
			y = 1. - y
		}
		if (x-(1.0-translation))*(1.-slope)-(y-translation)*slope >= 0. {
			result |= 1 << i
		}
	}
	return result
}

// / Make a lookup table of half-plane masks.
// /
// / The table is organized into two blocks each with `mask16Height/2` slopes.
// / The first block is negative slopes (x decreases as y increates),
// / the second as positive.
func makeMaskLUT16() []uint8 {
	var out []uint8
	for i := range mask16Width * mask16Height {
		const halfHeight = mask16Height / 2
		u := i % mask16Width
		v := i / mask16Width
		isPos := v >= halfHeight
		y := (float64(v%halfHeight) + 0.5) * (1.0 / float64(halfHeight))
		x := (float64(u) + 0.5) * (1.0 / float64(mask16Width))

		v16 := oneMask16(y, x, isPos)
		out = binary.LittleEndian.AppendUint16(out, v16)
	}
	return out
}
