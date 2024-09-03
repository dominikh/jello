// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package gfx

import "honnef.co/go/color"

type Brush interface {
	isBrush()
}

type SolidBrush struct {
	Color color.Color
}

type GradientBrush struct {
	Gradient Gradient
}

type ImageBrush struct {
	Image Image
}

func (SolidBrush) isBrush()    {}
func (GradientBrush) isBrush() {}
func (ImageBrush) isBrush()    {}

type Extend int

const (
	Pad Extend = iota
	Repeat
	Reflect
)
