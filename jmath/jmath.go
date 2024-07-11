package jmath

import (
	"math"
	"structs"

	"honnef.co/go/curve"
)

const Epsilon = 1e-12

func Abs32(f float32) float32 {
	return float32(math.Abs(float64(f)))
}

type Transform struct {
	_ structs.HostLayout

	Matrix      [4]float32
	Translation [2]float32
}

var Identity = Transform{
	Matrix: [4]float32{1, 0, 0, 1},
}

func (t Transform) Mul(other Transform) Transform {
	return Transform{
		Matrix: [4]float32{
			t.Matrix[0]*other.Matrix[0] + t.Matrix[2]*other.Matrix[1],
			t.Matrix[1]*other.Matrix[0] + t.Matrix[3]*other.Matrix[1],
			t.Matrix[0]*other.Matrix[2] + t.Matrix[2]*other.Matrix[3],
			t.Matrix[1]*other.Matrix[2] + t.Matrix[3]*other.Matrix[3],
		},
		Translation: [2]float32{
			t.Matrix[0]*other.Translation[0] +
				t.Matrix[2]*other.Translation[1] +
				t.Translation[0],
			t.Matrix[1]*other.Translation[0] +
				t.Matrix[3]*other.Translation[1] +
				t.Translation[1],
		},
	}
}

// / Converts an f32 to IEEE-754 binary16 format represented as the bits of a u16.
// / This implementation was adapted from Fabian Giesen's `float_to_half_fast3`()
// / function which can be found at <https://gist.github.com/rygorous/2156668#file-gistfile1-cpp-L285>
// /
// / TODO: We should consider adopting <https://crates.io/crates/half> as a dependency since it nicely
// / wraps native ARM and x86 instructions for floating-point conversion.
func Float16(val float32) uint16 {
	const inf32 uint32 = 255 << 23
	const inf16 uint32 = 31 << 23
	const magic uint32 = 15 << 23
	const signMask uint32 = 0x8000_0000
	const roundMask uint32 = 0xF000

	u := math.Float32bits(val)
	sign := u & signMask
	u = u ^ sign

	// NOTE all the integer compares in this function can be safely
	// compiled into signed compares since all operands are below
	// 0x80000000. Important if you want fast straight SSE2 code
	// (since there's no unsigned PCMPGTD).

	// Inf or NaN (all exponent bits set)
	var output uint16
	if u >= inf32 {
		// NaN -> qNaN and Inf->Inf
		if u > inf32 {
			output = 0x7E00
		} else {
			output = 0x7C00
		}
	} else {
		// (De)normalized number or zero
		u := u & roundMask
		u = math.Float32bits(math.Float32frombits(u) * math.Float32frombits(magic))
		u = u - roundMask

		// Clamp to signed infinity if exponent overflowed
		if u > inf16 {
			u = inf16
		}
		output = uint16(u >> 13) // Take the bits!
	}
	return output | uint16(sign>>16)
}

func TransformFromKurbo(transform curve.Affine) Transform {
	c := transform.Coefficients()
	return Transform{
		Matrix:      [4]float32{float32(c[0]), float32(c[1]), float32(c[2]), float32(c[3])},
		Translation: [2]float32{float32(c[4]), float32(c[5])},
	}
}

func AlignUp(len int, alignment int) int {
	return (len + alignment - 1) & -alignment
}

// TODO(dh): make alignUp generic and remove alignUpU32
func AlignUp32(len uint32, alignment uint32) uint32 {
	return (len + alignment - 1) & -alignment
}
