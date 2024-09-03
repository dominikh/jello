// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package gfx

import (
	"honnef.co/go/color"
	"honnef.co/go/jello/jmath"
)

func Premul16(c *color.Color) [4]jmath.Float16 {
	cc := c.Convert(color.LinearSRGB)
	r := cc.Values[0]
	g := cc.Values[1]
	b := cc.Values[2]
	a := cc.Alpha

	return [4]jmath.Float16{
		jmath.Float16bits(float32(r * a)),
		jmath.Float16bits(float32(g * a)),
		jmath.Float16bits(float32(b * a)),
		jmath.Float16bits(float32(a)),
	}
}

func Premul32(c *color.Color) [4]float32 {
	cc := c.Convert(color.LinearSRGB)
	r := cc.Values[0]
	g := cc.Values[1]
	b := cc.Values[2]
	a := cc.Alpha

	return [4]float32{
		float32(r * a),
		float32(g * a),
		float32(b * a),
		float32(a),
	}
}
