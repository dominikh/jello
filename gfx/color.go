package gfx

import "math"

type Color struct {
	R, G, B, A uint8
}

func (c Color) WithAlphaFactor(alpha float32) Color {
	c.A = uint8(math.Round(float64(float32(c.A) * alpha)))
	return c
}

func (c Color) PremulUint32() uint32 {
	a := float64(c.A) * (1.0 / 255.0)
	r := uint32(math.Round(float64(c.R) * a))
	g := uint32(math.Round(float64(c.G) * a))
	b := uint32(math.Round(float64(c.B) * a))
	return (r << 24) | (g << 16) | (b << 8) | uint32(c.A)
}
