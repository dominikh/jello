package jmath

import (
	"math"
	"structs"

	"golang.org/x/exp/constraints"
	"honnef.co/go/curve"
)

const Epsilon = 1e-12

func AbsInt32(x int32) int32 {
	if x < 0 {
		return -x
	}
	return x
}

func Abs32(f float32) float32 {
	return float32(math.Abs(float64(f)))
}

func Floor32(f float32) float32 {
	return float32(math.Floor(float64(f)))
}

func Ceil32(f float32) float32 {
	return float32(math.Ceil(float64(f)))
}

func Round32(f float32) float32 {
	return float32(math.Round(float64(f)))
}

func Copysign32(f, sign float32) float32 {
	return float32(math.Copysign(float64(f), float64(sign)))
}

func Clamp[T constraints.Integer | constraints.Float](x, minv, maxv T) T {
	return min(max(x, minv), maxv)
}

func Sqrt32(f float32) float32 {
	return float32(math.Sqrt(float64(f)))
}

func Hypot32(a, b float32) float32 {
	return float32(math.Hypot(float64(a), float64(b)))
}

func Acos32(x float32) float32 {
	return float32(math.Acos(float64(x)))
}

func Sincos32(x float32) (float32, float32) {
	a, b := math.Sincos(float64(x))
	return float32(a), float32(b)
}

func Atan232(y, x float32) float32 {
	return float32(math.Atan2(float64(y), float64(x)))
}

func Cos32(x float32) float32 {
	return float32(math.Cos(float64(x)))
}

func Sin32(x float32) float32 {
	return float32(math.Sin(float64(x)))
}

func Pow32(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func Asin32(x float32) float32 {
	return float32(math.Asin(float64(x)))
}

func Cbrt32(x float32) float32 {
	return float32(math.Cbrt(float64(x)))
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

// Float16 converts a float32 to IEEE-754 binary16 format represented as the bits of a u16.
// This implementation was adapted from Fabian Giesen's `float_to_half_fast3`()
// function which can be found at <https://gist.github.com/rygorous/2156668#file-gistfile1-cpp-L285>
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

// Float32 converts a 16-bit precision IEEE-754 binary16 float to float32.
//
// This implementation was adapted from Fabian Giesen's `half_to_float()`
// function which can be found at <https://gist.github.com/rygorous/2156668#file-gistfile1-cpp-L574>
func Float32(val uint16) float32 {
	bits := uint32(val)
	const magic uint32 = 113 << 23
	const shiftedExp uint32 = 0x7c00 << 13 // exponent mask after shift

	o := (bits & 0x7fff) << 13 // exponent/mantissa bits
	exp := shiftedExp & o      // just the exponent
	o += (127 - 15) << 23      // exponent adjust

	// handle exponent special cases
	switch exp {
	case shiftedExp:
		// Inf/NaN?
		o += (128 - 16) << 23 // extra exp adjust
	case 0:
		// Zero/Denormal?
		o += 1 << 23                                                                // extra exp adjust
		o = math.Float32bits(math.Float32frombits(o) - math.Float32frombits(magic)) // normalize
	}

	return math.Float32frombits(o | ((bits & 0x8000) << 16)) // sign bit
}

func TransformFromKurbo(transform curve.Affine) Transform {
	c := transform.Coefficients()
	return Transform{
		Matrix:      [4]float32{float32(c[0]), float32(c[1]), float32(c[2]), float32(c[3])},
		Translation: [2]float32{float32(c[4]), float32(c[5])},
	}
}

func AlignUp[Int constraints.Integer](len Int, alignment Int) Int {
	return (len + alignment - 1) & -alignment
}

func PointToF32(p curve.Point) [2]float32 {
	return [2]float32{
		float32(p.X),
		float32(p.Y),
	}
}
