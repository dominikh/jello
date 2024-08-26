// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package gfx

import (
	"math"

	"honnef.co/go/jello/jmath"
)

var _ Color = LinearSRGB{}
var _ Color = SRGB{}
var _ Color = Oklab{}
var _ Color = Oklch{}

type Color interface {
	LinearSRGB() LinearSRGB
	Lerp(to Color, t float64) Color
	WithAlphaFactor(alpha float32) Color
}

type LinearSRGB struct {
	R, G, B, A float32
}

func (c LinearSRGB) LinearSRGB() LinearSRGB { return c }

func (c LinearSRGB) SRGB() SRGB {
	compress := func(c float32) uint8 {
		if c < 0 {
			c = 0
		}
		if c > 1 {
			c = 1
		}

		if c <= 0.0031308 {
			return uint8(math.Round(float64(c*12.92) * 255))
		} else {
			return uint8(math.Round((1.055*math.Pow(float64(c), 1.0/2.4) - 0.055) * 255))
		}
	}

	out := SRGB{
		R: compress(c.R),
		G: compress(c.G),
		B: compress(c.B),
		A: uint8(math.Round(float64(max(0, min(1, c.A)) * 255))),
	}
	return out
}

func (c LinearSRGB) Lerp(to Color, t float64) Color {
	tol := to.LinearSRGB()
	return LinearSRGB{
		R: lerpFloat(c.R, tol.R, t),
		G: lerpFloat(c.G, tol.G, t),
		B: lerpFloat(c.B, tol.B, t),
		A: lerpFloat(c.A, tol.A, t),
	}
}

func (c LinearSRGB) WithAlphaFactor(alpha float32) Color {
	c.A *= alpha
	return c
}

func (c LinearSRGB) Premul16() [4]jmath.Float16 {
	return [4]jmath.Float16{
		jmath.Float16bits(c.R * c.A),
		jmath.Float16bits(c.G * c.A),
		jmath.Float16bits(c.B * c.A),
		jmath.Float16bits(c.A),
	}
}

func (c LinearSRGB) Premul32() [4]float32 {
	return [4]float32{
		c.R * c.A,
		c.G * c.A,
		c.B * c.A,
		c.A,
	}
}

func (c LinearSRGB) Oklab() Oklab {
	r := float64(c.R)
	g := float64(c.G)
	b := float64(c.B)

	l := 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
	m := 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
	s := 0.0883024619*r + 0.2817188376*g + 0.6299787005*b

	l_ := math.Cbrt(l)
	m_ := math.Cbrt(m)
	s_ := math.Cbrt(s)

	return Oklab{
		L:     float32(0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_),
		A:     float32(1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_),
		B:     float32(0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_),
		Alpha: float32(c.A),
	}
}

func (c SRGB) WithAlphaFactor(alpha float32) Color {
	c.A = uint8(math.Round(float64(float32(c.A) * alpha)))
	return c
}

type SRGB struct {
	R, G, B, A uint8
}

func (c SRGB) LinearSRGB() LinearSRGB {
	decompress := func(c float32) float32 {
		if c <= 0.04045 {
			return c / 12.92
		} else {
			return float32(math.Pow(float64((c+0.055)/1.055), 2.4))
		}
	}

	fr := float32(c.R) / 255.0
	fg := float32(c.G) / 255.0
	fb := float32(c.B) / 255.0
	fa := float32(c.A) / 255.0

	return LinearSRGB{
		R: decompress(fr),
		G: decompress(fg),
		B: decompress(fb),
		A: fa,
	}
}

func (c SRGB) Lerp(to Color, t float64) Color {
	return c.LinearSRGB().Lerp(to.LinearSRGB(), t)
}

type Oklab struct {
	L, A, B, Alpha float32
}

func (c Oklab) Oklch() Oklch {
	return Oklch(lab(c).LCh())
}

func (c Oklab) Lerp(to Color, t float64) Color {
	var toOklab Oklab
	switch to := to.(type) {
	case Oklab:
		toOklab = to
	case interface{ Oklab() Oklab }:
		toOklab = to.Oklab()
	default:
		toOklab = to.LinearSRGB().Oklab()
	}
	return Oklab{
		L:     lerpFloat(c.L, toOklab.L, t),
		A:     lerpFloat(c.A, toOklab.A, t),
		B:     lerpFloat(c.B, toOklab.B, t),
		Alpha: lerpFloat(c.Alpha, toOklab.Alpha, t),
	}
}

func (c Oklab) WithAlphaFactor(alpha float32) Color {
	c.Alpha *= alpha
	return c
}

func (c Oklab) LinearSRGB() LinearSRGB {
	c = c.Oklch().MapToSRGBGamut().Oklab()
	return c.unmappedLinearSRGB()
}

func (c Oklab) unmappedLinearSRGB() LinearSRGB {
	l_ := c.L + 0.3963377774*c.A + 0.2158037573*c.B
	m_ := c.L - 0.1055613458*c.A - 0.0638541728*c.B
	s_ := c.L - 0.0894841775*c.A - 1.2914855480*c.B

	l := l_ * l_ * l_
	m := m_ * m_ * m_
	s := s_ * s_ * s_

	return LinearSRGB{
		+4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
		-1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
		-0.0041960863*l - 0.7034186147*m + 1.7076147010*s,
		c.Alpha,
	}
}

type Oklch struct {
	L, C, H, A float32
}

func (c Oklch) Lerp(to Color, t float64) Color {
	var toOklch Oklch
	switch to := to.(type) {
	case Oklch:
		toOklch = to
	case interface{ Oklch() Oklch }:
		toOklch = to.Oklch()
	case interface{ Oklab() Oklab }:
		toOklch = to.Oklab().Oklch()
	default:
		toOklch = to.LinearSRGB().Oklab().Oklch()
	}
	return Oklch{
		L: lerpFloat(c.L, toOklch.L, t),
		C: lerpFloat(c.C, toOklch.C, t),
		H: lerpFloat(c.H, toOklch.H, t),
		A: lerpFloat(c.A, toOklch.A, t),
	}
}

func (c Oklch) LinearSRGB() LinearSRGB {
	return c.MapToSRGBGamut().Oklab().LinearSRGB()
}

func (c Oklch) WithAlphaFactor(alpha float32) Color {
	c.A *= alpha
	return c
}

func (c Oklch) Oklab() Oklab {
	h := float64(c.H * (math.Pi / 180))
	return Oklab{
		L:     c.L,
		A:     c.C * float32(math.Cos(h)),
		B:     c.C * float32(math.Sin(h)),
		Alpha: c.A,
	}
}

// MapToSRGBGamut maps colors that fall outside the sRGB gamut to the sRGB
// gamut. It uses the same algorithm as [CSS Color Module Level 4]. Note that
// the mapping implements a relative colorimetric intent. That is, colors that
// are already inside the gamut are unchanged. This is intended for mapping
// individual colors, not for mapping images.
//
// [CSS Color Module Level 4]: https://www.w3.org/TR/css-color-4/#css-gamut-mapping
func (c Oklch) MapToSRGBGamut() LinearSRGB {
	// The just noticeable difference between two colors in Oklch
	const jnd = 0.02
	const epsilon = 0.0001

	if c.L >= 1 {
		return LinearSRGB{1, 1, 1, c.A}
	}
	if c.L <= 0 {
		return LinearSRGB{0, 0, 0, c.A}
	}

	inGamut := func(color Oklch) (LinearSRGB, bool) {
		// OPT(dh): is there an easier way to check if the color is in gamut than to try and convert it?
		s := color.Oklab().unmappedLinearSRGB()
		if s.R >= 0 && s.R <= 1 &&
			s.G >= 0 && s.G <= 1 &&
			s.B >= 0 && s.B <= 1 {
			return s, true
		} else {
			return LinearSRGB{}, false
		}
	}
	inGamut1 := func(color Oklch) bool {
		_, ok := inGamut(color)
		return ok
	}

	if m, ok := inGamut(c); ok {
		return m
	}

	clip := func(color Oklch) LinearSRGB {
		m := color.Oklab().unmappedLinearSRGB()
		fmin := func(a, b float32) float32 {
			if a <= b {
				return a
			} else {
				return b
			}
		}
		fmax := func(a, b float32) float32 {
			if a >= b {
				return a
			} else {
				return b
			}
		}

		m.R = fmin(fmax(m.R, 0), 1)
		m.G = fmin(fmax(m.G, 0), 1)
		m.B = fmin(fmax(m.B, 0), 1)
		return m
	}

	min := float32(0)
	max := c.C
	min_inGamut := true
	current := c
	clipped := clip(c)

	E := okLabDifference(clipped.Oklab(), current.Oklab())
	if E < jnd {
		return clipped
	}

	for max-min > epsilon {
		chroma := (min + max) / 2
		current.C = chroma

		if min_inGamut && inGamut1(current) {
			min = chroma
		} else {
			clipped = clip(current)
			E = okLabDifference(clipped.Oklab(), current.Oklab())
			if E < jnd {
				if jnd-E < epsilon {
					return clipped
				} else {
					min_inGamut = false
					min = chroma
				}
			} else {
				max = chroma
			}
		}
	}
	return current.Oklab().unmappedLinearSRGB()
}

func okLabDifference(reference, sample Oklab) (deltaEOK float32) {
	L1, a1, b1 := reference.L, reference.A, reference.B
	L2, a2, b2 := sample.L, sample.A, sample.B
	deltaL := float64(L1 - L2)
	deltaa := float64(a1 - a2)
	deltab := float64(b1 - b2)
	return float32(math.Hypot(math.Hypot(deltaL, deltaa), deltab))
}

func lerpFloat(x, y float32, a float64) float32 {
	return x*float32(1.0-a) + y*float32(a)
}

type lab struct {
	L     float32
	A     float32
	B     float32
	Alpha float32
}

type lch struct {
	L float32
	C float32
	H float32
	A float32
}

func (c lab) LCh() lch {
	hue := float32(math.Atan2(float64(c.B), float64(c.A))) * (180 / math.Pi)
	if hue < 0 {
		hue += 360
	}
	return lch{
		c.L,
		float32(math.Hypot(float64(c.A), float64(c.B))),
		hue,
		c.Alpha,
	}
}

func (c lch) Lab() lab {
	h := float64(c.H * (math.Pi / 180))
	return lab{
		L:     c.L,
		A:     c.C * float32(math.Cos(h)),
		B:     c.C * float32(math.Sin(h)),
		Alpha: c.A,
	}
}
