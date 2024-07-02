package jello

import (
	"math"
	"structs"

	"honnef.co/go/curve"
)

type Transform struct {
	_ structs.HostLayout

	Matrix      [4]float32
	Translation [2]float32
}

var Identity = Transform{
	Matrix: [4]float32{1, 0, 0, 1},
}

func (self Transform) Mul(other Transform) Transform {
	return Transform{
		Matrix: [4]float32{
			self.Matrix[0]*other.Matrix[0] + self.Matrix[2]*other.Matrix[1],
			self.Matrix[1]*other.Matrix[0] + self.Matrix[3]*other.Matrix[1],
			self.Matrix[0]*other.Matrix[2] + self.Matrix[2]*other.Matrix[3],
			self.Matrix[1]*other.Matrix[2] + self.Matrix[3]*other.Matrix[3],
		},
		Translation: [2]float32{
			self.Matrix[0]*other.Translation[0] +
				self.Matrix[2]*other.Translation[1] +
				self.Translation[0],
			self.Matrix[1]*other.Translation[0] +
				self.Matrix[3]*other.Translation[1] +
				self.Translation[1],
		},
	}
}

// / Converts an f32 to IEEE-754 binary16 format represented as the bits of a u16.
// / This implementation was adapted from Fabian Giesen's `float_to_half_fast3`()
// / function which can be found at <https://gist.github.com/rygorous/2156668#file-gistfile1-cpp-L285>
// /
// / TODO: We should consider adopting <https://crates.io/crates/half> as a dependency since it nicely
// / wraps native ARM and x86 instructions for floating-point conversion.
func f32_to_f16(val float32) uint16 {
	const INF_32 uint32 = 255 << 23
	const INF_16 uint32 = 31 << 23
	const MAGIC uint32 = 15 << 23
	const SIGN_MASK uint32 = 0x8000_0000
	const ROUND_MASK uint32 = 0xF000

	u := math.Float32bits(val)
	sign := u & SIGN_MASK
	u = u ^ sign

	// NOTE all the integer compares in this function can be safely
	// compiled into signed compares since all operands are below
	// 0x80000000. Important if you want fast straight SSE2 code
	// (since there's no unsigned PCMPGTD).

	// Inf or NaN (all exponent bits set)
	var output uint16
	if u >= INF_32 {
		// NaN -> qNaN and Inf->Inf
		if u > INF_32 {
			output = 0x7E00
		} else {
			output = 0x7C00
		}
	} else {
		// (De)normalized number or zero
		u := u & ROUND_MASK
		u = math.Float32bits(math.Float32frombits(u) * math.Float32frombits(MAGIC))
		u = u - ROUND_MASK

		// Clamp to signed infinity if exponent overflowed
		if u > INF_16 {
			u = INF_16
		}
		output = uint16(u >> 13) // Take the bits!
	}
	return output | uint16(sign>>16)
}

func transformFromKurbo(transform curve.Affine) Transform {
	c := transform.Coefficients()
	return Transform{
		Matrix:      [4]float32{float32(c[0]), float32(c[1]), float32(c[2]), float32(c[3])},
		Translation: [2]float32{float32(c[4]), float32(c[5])},
	}
}
