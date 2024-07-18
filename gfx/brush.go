package gfx

type Brush interface {
	isBrush()
}

type SolidBrush struct {
	Color Color
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
