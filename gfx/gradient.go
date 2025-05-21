// Copyright 2022 the Peniko Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package gfx

import (
	"honnef.co/go/color"
	"honnef.co/go/curve"
)

type ColorStop struct {
	Offset float32
	Color  color.Color
}

func (cs *ColorStop) WithAlphaFactor(alpha float32) ColorStop {
	c := cs.Color
	c.Values[3] = float64(alpha)
	return ColorStop{
		Offset: cs.Offset,
		Color:  c,
	}
}

type Gradient interface {
	isGradient()
}

type LinearGradient struct {
	Start  curve.Point
	End    curve.Point
	Stops  []ColorStop
	Extend Extend
}

func (LinearGradient) isGradient() {}

type RadialGradient struct {
	StartCenter curve.Point
	StartRadius float32
	EndCenter   curve.Point
	EndRadius   float32
	Stops       []ColorStop
	Extend      Extend
}

func (RadialGradient) isGradient() {}

type SweepGradient struct {
	Center     curve.Point
	StartAngle float32
	EndAngle   float32
	Stops      []ColorStop
	Extend     Extend
}

func (SweepGradient) isGradient() {}
