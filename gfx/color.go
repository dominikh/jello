package gfx

import "math"

type Color struct {
	R, G, B, A float32
}

func FromSRGB(r, g, b, a uint8) Color {
	decompress := func(c float32) float32 {
		if c <= 0.04045 {
			return c / 12.92
		} else {
			return float32(math.Pow(float64((c+0.055)/1.055), 2.4))
		}
	}

	fr := float32(r) / 255.0
	fg := float32(g) / 255.0
	fb := float32(b) / 255.0
	fa := float32(a) / 255.0

	return Color{
		R: decompress(fr),
		G: decompress(fg),
		B: decompress(fb),
		A: fa,
	}
}

func (c Color) WithAlphaFactor(alpha float32) Color {
	c.A *= alpha
	return c
}

func (c Color) PremulUint32() uint32 {
	clamp := func(v, low, high float32) float32 {
		if v < low {
			return low
		}
		if v > high {
			return high
		}
		return v
	}
	a := clamp(c.A, 0, 1)
	r := uint32(clamp(c.R*a, 0, 1) * 255)
	g := uint32(clamp(c.G*a, 0, 1) * 255)
	b := uint32(clamp(c.B*a, 0, 1) * 255)
	ua := uint32(a * 255.0)
	return (r << 24) | (g << 16) | (b << 8) | ua
}
