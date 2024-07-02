package jello

import "encoding/binary"

//! Create a lookup table of half-plane sample masks.

// Width is number of discrete translations
const MASK_WIDTH = 32

// Height is the number of discrete slopes
const MASK_HEIGHT = 32

var PATTERN = [...]uint8{0, 5, 3, 7, 1, 4, 6, 2}

func one_mask(slope float64, translation float64, is_pos bool) uint8 {
	if is_pos {
		translation = 1. - translation
	}
	var result uint8
	for i, item := range PATTERN {
		y := (float64(i) + 0.5) * 0.125
		x := (float64(item) + 0.5) * 0.125
		if !is_pos {
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
// / The table is organized into two blocks each with `MASK_HEIGHT/2` slopes.
// / The first block is negative slopes (x decreases as y increates),
// / the second as positive.
func make_mask_lut() []uint8 {
	var out []uint8
	for i := range MASK_WIDTH * MASK_HEIGHT {
		const HALF_HEIGHT = MASK_HEIGHT / 2
		u := i % MASK_WIDTH
		v := i / MASK_WIDTH
		is_pos := v >= HALF_HEIGHT
		y := (float64(v%HALF_HEIGHT) + 0.5) * (1.0 / float64(HALF_HEIGHT))
		x := (float64(u) + 0.5) * (1.0 / float64(MASK_WIDTH))
		out = append(out, one_mask(y, x, is_pos))
	}
	return out
}

// Width is number of discrete translations
const MASK16_WIDTH = 64

// Height is the number of discrete slopes
const MASK16_HEIGHT = 64

// This is based on the [D3D11 standard sample pattern].
//
// [D3D11 standard sample pattern]: https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_standard_multisample_quality_levels
var PATTERN_16 = [...]uint8{1, 8, 4, 11, 15, 7, 3, 12, 0, 9, 5, 13, 2, 10, 6, 14}

func one_mask_16(slope float64, translation float64, is_pos bool) uint16 {
	if is_pos {
		translation = 1. - translation
	}
	var result uint16
	for i, item := range PATTERN_16 {
		y := (float64(i) + 0.5) * 0.0625
		x := (float64(item) + 0.5) * 0.0625
		if !is_pos {
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
// / The table is organized into two blocks each with `MASK16_HEIGHT/2` slopes.
// / The first block is negative slopes (x decreases as y increates),
// / the second as positive.
func make_mask_lut_16() []uint8 {
	var out []uint8
	for i := range MASK16_WIDTH * MASK16_HEIGHT {
		const HALF_HEIGHT = MASK16_HEIGHT / 2
		u := i % MASK16_WIDTH
		v := i / MASK16_WIDTH
		is_pos := v >= HALF_HEIGHT
		y := (float64(v%HALF_HEIGHT) + 0.5) * (1.0 / float64(HALF_HEIGHT))
		x := (float64(u) + 0.5) * (1.0 / float64(MASK16_WIDTH))

		v16 := one_mask_16(y, x, is_pos)
		out = binary.LittleEndian.AppendUint16(out, v16)
	}
	return out
}
