package jello

import (
	"encoding/binary"
	"iter"
	"math"
	"slices"

	"honnef.co/go/brush"
	"honnef.co/go/curve"
)

type Encoding struct {
	PathTags   []PathTag
	PathData   []byte
	DrawTags   []DrawTag
	DrawData   []byte
	Transforms []Transform
	Styles     []Style
	// Resources Resources
	NumPaths        uint32
	NumPathSegments uint32
	NumClips        uint32
	NumOpenClips    uint32
	Flags           uint32
}

const (
	FORCE_NEXT_TRANSFORM uint32 = 1
	FORCE_NEXT_STYLE     uint32 = 2
)

func (enc *Encoding) IsEmpty() bool {
	return len(enc.PathTags) == 0
}

func (enc *Encoding) Reset() {
	enc.PathTags = enc.PathTags[:0]
	enc.PathData = enc.PathData[:0]
	enc.DrawTags = enc.DrawTags[:0]
	enc.DrawData = enc.DrawData[:0]
	enc.Transforms = enc.Transforms[:0]
	enc.Styles = enc.Styles[:0]
	enc.NumPaths = 0
	enc.NumPathSegments = 0
	enc.NumClips = 0
	enc.NumOpenClips = 0
	enc.Flags = 0
}

func (enc *Encoding) Append(other *Encoding, transform Transform) {
	enc.PathTags = append(enc.PathTags, other.PathTags...)
	enc.PathData = append(enc.PathData, other.PathData...)
	enc.DrawTags = append(enc.DrawTags, other.DrawTags...)
	enc.DrawData = append(enc.DrawData, other.DrawData...)
	enc.NumPaths += other.NumPaths
	enc.NumPathSegments += other.NumPathSegments
	enc.NumClips += other.NumClips
	enc.NumOpenClips += other.NumOpenClips
	// XXX(dh): is this assignment to enc.flags correct?
	enc.Flags = other.Flags
	if transform != Identity {
		enc.Transforms = slices.Grow(enc.Transforms, len(other.Transforms))
		for _, t := range other.Transforms {
			enc.Transforms = append(enc.Transforms, transform.Mul(t))
		}
	} else {
		enc.Transforms = append(enc.Transforms, other.Transforms...)
	}
	enc.Styles = append(enc.Styles, other.Styles...)
}

func (enc *Encoding) StreamOffsets() StreamOffsets {
	return StreamOffsets{
		PathTags:   len(enc.PathTags),
		PathData:   len(enc.PathData),
		DrawTags:   len(enc.DrawTags),
		DrawData:   len(enc.DrawData),
		Transforms: len(enc.Transforms),
		Styles:     len(enc.Styles),
	}
}

func (enc *Encoding) EncodeFillStyle(fill brush.Fill) {
	enc.EncodeStyle(StyleFromFill(fill))
}

func (enc *Encoding) EncodeStrokeStyle(stroke curve.Stroke) {
	enc.EncodeStyle(StyleFromStroke(stroke))
}

func (enc *Encoding) EncodeStyle(style Style) {
	if enc.Flags&FORCE_NEXT_STYLE != 0 || len(enc.Styles) == 0 || enc.Styles[len(enc.Styles)-1] != style {
		enc.PathTags = append(enc.PathTags, STYLE)
		enc.Styles = append(enc.Styles, style)
		enc.Flags &^= FORCE_NEXT_STYLE
	}
}

func (enc *Encoding) EncodeTransform(transform Transform) bool {
	if enc.Flags&FORCE_NEXT_TRANSFORM != 0 || len(enc.Transforms) == 0 || enc.Transforms[len(enc.Transforms)-1] != transform {
		enc.PathTags = append(enc.PathTags, TRANSFORM)
		enc.Transforms = append(enc.Transforms, transform)
		enc.Flags &^= FORCE_NEXT_TRANSFORM
		return true
	} else {
		return false
	}
}

func (enc *Encoding) EncodePath(isFill bool) *PathEncoder {
	return &PathEncoder{
		tags:        &enc.PathTags,
		data:        &enc.PathData,
		numSegments: &enc.NumPathSegments,
		numPaths:    &enc.NumPaths,
		isFill:      isFill,
	}
}

func (enc *Encoding) EncodeShape(shape curve.Shape, isFill bool) bool {
	pe := enc.EncodePath(isFill)
	pe.Shape(shape)
	return pe.Finish(true) != 0
}

func (enc *Encoding) EncodePathElements(path iter.Seq[curve.PathElement], isFill bool) bool {
	pe := enc.EncodePath(isFill)
	pe.PathElements(path)
	return pe.Finish(true) != 0
}

func (enc *Encoding) EncodeBrush(b brush.Brush, alpha float32) {
	switch b := b.(type) {
	case brush.SolidBrush:
		var color brush.Color
		if alpha == 1.0 {
			color = b.Color
		} else {
			color = b.Color.WithAlphaFactor(alpha)
		}
		enc.EncodeColor(DrawColor{RGBA: color.PremulUint32()})
	case brush.GradientBrush:
		panic("unsupported")
	case brush.ImageBrush:
		panic("unsupported")
	}
}

func (enc *Encoding) EncodeColor(color DrawColor) {
	enc.DrawTags = append(enc.DrawTags, COLOR)
	enc.DrawData = binary.LittleEndian.AppendUint32(enc.DrawData, color.RGBA)
}

// XXX EncodeLinearGradient
// XXX EncodeRadialGradient
// XXX EncodeSweepGradient
// XXX EncodeImage

func (enc *Encoding) EncodeBeginClip(blendMode brush.BlendMode, alpha float32) {
	enc.DrawTags = append(enc.DrawTags, BEGIN_CLIP)
	d1 := (uint32(blendMode.Mix) << 8) | uint32(blendMode.Compose)
	d2 := alpha
	var d [8]byte
	binary.LittleEndian.PutUint32(d[0:4], d1)
	binary.LittleEndian.PutUint32(d[4:8], math.Float32bits(d2))
	enc.DrawData = append(enc.DrawData, d[:]...)
	enc.NumClips++
	enc.NumOpenClips++
}

func (enc *Encoding) EncodeEndClip() {
	if enc.NumOpenClips == 0 {
		return
	}
	enc.DrawTags = append(enc.DrawTags, END_CLIP)
	// This is a dummy path, and will go away with the new clip impl.
	enc.PathTags = append(enc.PathTags, PATH)
	enc.NumPaths++
	enc.NumClips++
	enc.NumOpenClips--
}

func (enc *Encoding) ForceNextTransformAndStyle() {
	enc.Flags |= FORCE_NEXT_TRANSFORM | FORCE_NEXT_STYLE
}

func (enc *Encoding) SwapLastPathTags() {
	n := len(enc.PathTags)
	enc.PathTags[n-2], enc.PathTags[n-1] = enc.PathTags[n-1], enc.PathTags[n-2]
}

// XXX AddRamp

type StreamOffsets struct {
	// Current length of path tag stream.
	PathTags int
	// Current length of path data stream.
	PathData int
	// Current length of draw tag stream.
	DrawTags int
	// Current length of draw data stream.
	DrawData int
	// Current length of transform stream.
	Transforms int
	// Current length of style stream.
	Styles int
}

func (so StreamOffsets) Add(oso StreamOffsets) StreamOffsets {
	return StreamOffsets{
		so.PathTags + oso.PathTags,
		so.PathData + oso.PathData,
		so.DrawTags + oso.DrawTags,
		so.DrawData + oso.DrawData,
		so.Transforms + oso.Transforms,
		so.Styles + oso.Styles,
	}
}
