package gfx

import "honnef.co/go/curve"

type ColorStop struct {
	Offset float32
	Color  Color
}

func (cs ColorStop) WithAlphaFactor(alpha float32) ColorStop {
	return ColorStop{
		Offset: cs.Offset,
		Color:  cs.Color.WithAlphaFactor(alpha),
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
