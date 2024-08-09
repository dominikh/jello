// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package encoding

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"honnef.co/go/curve"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/safeish"
)

type Encoding struct {
	PathTags        []PathTag
	PathData        []byte
	DrawTags        []DrawTag
	DrawData        []byte
	Transforms      []jmath.Transform
	Styles          []Style
	Resources       Resources
	NumPaths        uint32
	NumPathSegments uint32
	NumClips        uint32
	NumOpenClips    uint32
	Flags           uint32
}

const (
	forceNextTransform uint32 = 1
	forceNextStyle     uint32 = 2
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
	enc.Resources.Reset()
	enc.NumPaths = 0
	enc.NumPathSegments = 0
	enc.NumClips = 0
	enc.NumOpenClips = 0
	enc.Flags = 0
}

func (enc *Encoding) Append(other *Encoding, transform jmath.Transform) {
	offsets := enc.StreamOffsets()
	stopsBase := len(enc.Resources.ColorStops)
	// XXX glyph stuff
	for _, patch := range other.Resources.Patches {
		// XXX glyph stuff
		switch patch := patch.(type) {
		case RampPatch:
			stops := [2]int{
				patch.Stops[0] + stopsBase,
				patch.Stops[1] + stopsBase,
			}
			enc.Resources.Patches = append(enc.Resources.Patches, RampPatch{
				DrawDataOffset: patch.DrawDataOffset + offsets.DrawData,
				Stops:          stops,
				Extend:         patch.Extend,
			})
		case ImagePatch:
			enc.Resources.Patches = append(enc.Resources.Patches, ImagePatch{
				Image:          patch.Image,
				DrawDataOffset: patch.DrawDataOffset + offsets.DrawData,
			})
		default:
			panic(fmt.Sprintf("unhandled type %T", patch))
		}
	}
	enc.Resources.ColorStops = append(enc.Resources.ColorStops, other.Resources.ColorStops...)

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
	if transform != jmath.Identity {
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

func (enc *Encoding) ApplyTransform(transform jmath.Transform) {
	for i := range enc.Transforms {
		enc.Transforms[i] = transform.Mul(enc.Transforms[i])
	}
}

func (enc *Encoding) EncodeFillStyle(fill gfx.Fill) {
	enc.EncodeStyle(styleFromFill(fill))
}

func (enc *Encoding) EncodeStrokeStyle(stroke curve.Stroke) {
	enc.EncodeStyle(styleFromStroke(stroke))
}

func (enc *Encoding) EncodeStyle(style Style) {
	if enc.Flags&forceNextStyle != 0 || len(enc.Styles) == 0 || enc.Styles[len(enc.Styles)-1] != style {
		enc.PathTags = append(enc.PathTags, PathTagStyle)
		enc.Styles = append(enc.Styles, style)
		enc.Flags &^= forceNextStyle
	}
}

func (enc *Encoding) EncodeTransform(transform jmath.Transform) bool {
	if enc.Flags&forceNextTransform != 0 || len(enc.Transforms) == 0 || enc.Transforms[len(enc.Transforms)-1] != transform {
		enc.PathTags = append(enc.PathTags, PathTagTransform)
		enc.Transforms = append(enc.Transforms, transform)
		enc.Flags &^= forceNextTransform
		return true
	} else {
		return false
	}
}

func (enc *Encoding) EncodePath(path curve.BezPath, isFill bool) bool {
	pe := &pathEncoder{
		tags:        &enc.PathTags,
		data:        &enc.PathData,
		numSegments: &enc.NumPathSegments,
		numPaths:    &enc.NumPaths,
		isFill:      isFill,
	}
	pe.Path(path)
	return pe.Finish(true) != 0
}

func (enc *Encoding) EncodeBrush(b gfx.Brush, alpha float32) {
	switch b := b.(type) {
	case gfx.SolidBrush:
		var color gfx.Color
		if alpha == 1.0 {
			color = b.Color
		} else {
			color = b.Color.WithAlphaFactor(alpha)
		}
		enc.EncodeColor(newDrawColor(color))
	case gfx.GradientBrush:
		switch g := b.Gradient.(type) {
		case gfx.LinearGradient:
			enc.EncodeLinearGradient(
				drawLinearGradient{
					Index: 0,
					P0:    jmath.PointToF32(g.Start),
					P1:    jmath.PointToF32(g.End),
				},
				g.Stops,
				alpha,
				g.Extend,
			)
		case gfx.RadialGradient:
			enc.EncodeRadialGradient(
				drawRadialGradient{
					Index: 0,
					P0:    jmath.PointToF32(g.StartCenter),
					P1:    jmath.PointToF32(g.EndCenter),
					R0:    g.StartRadius,
					R1:    g.EndRadius,
				},
				g.Stops,
				alpha,
				g.Extend,
			)
		case gfx.SweepGradient:
			enc.EncodeSweepGradient(
				drawSweepGradient{
					Index: 0,
					P0:    jmath.PointToF32(g.Center),
					T0:    g.StartAngle / (2 * math.Pi),
					T1:    g.EndAngle / (2 * math.Pi),
				},
				g.Stops,
				alpha,
				g.Extend,
			)
		default:
			panic(fmt.Sprintf("unsupported gradient %T", g))
		}
	case gfx.ImageBrush:
		enc.EncodeImage(b.Image, 1)
	default:
		panic(fmt.Sprintf("unhandled type %T", b))
	}
}

func (enc *Encoding) EncodeColor(color drawColor) {
	enc.DrawTags = append(enc.DrawTags, DrawTagColor)
	enc.DrawData = binary.LittleEndian.AppendUint32(enc.DrawData, color.RGBA)
}

func (enc *Encoding) addRamp(colorStops []gfx.ColorStop, alpha float32, extend gfx.Extend) {
	if len(colorStops) < 2 {
		panic("addRamp called with less than 2 color stops")
	}

	offset := len(enc.DrawData)
	stopsStart := len(enc.Resources.ColorStops)
	if alpha != 1.0 {
		cp := make([]gfx.ColorStop, len(colorStops))
		copy(cp, colorStops)
		for i := range cp {
			cp[i] = cp[i].WithAlphaFactor(alpha)
		}
		colorStops = cp
	}
	enc.Resources.ColorStops = append(enc.Resources.ColorStops, colorStops...)
	stopsEnd := len(enc.Resources.ColorStops)
	enc.Resources.Patches = append(enc.Resources.Patches, RampPatch{
		DrawDataOffset: offset,
		Stops:          [2]int{stopsStart, stopsEnd},
		Extend:         extend,
	})
}

func (enc *Encoding) EncodeLinearGradient(
	gradient drawLinearGradient,
	colorStops []gfx.ColorStop,
	alpha float32,
	extend gfx.Extend,
) {
	switch len(colorStops) {
	case 0:
		enc.EncodeColor(drawColor{})
	case 1:
		enc.EncodeColor(newDrawColor(colorStops[0].Color.WithAlphaFactor(alpha)))
	default:
		enc.addRamp(colorStops, alpha, extend)
		enc.DrawTags = append(enc.DrawTags, DrawTagLinearGradient)
		enc.DrawData = append(enc.DrawData, safeish.AsBytes(&gradient)...)
	}
}

func (enc *Encoding) EncodeRadialGradient(
	gradient drawRadialGradient,
	colorStops []gfx.ColorStop,
	alpha float32,
	extend gfx.Extend,
) {
	// Match Skia's epsilon for radii comparison
	const skiaEpsilon = 1.0 / (1 << 12)
	if gradient.P0 == gradient.P1 && jmath.Abs32(gradient.R0-gradient.R1) < skiaEpsilon {
		enc.EncodeColor(drawColor{})
		return
	}

	switch len(colorStops) {
	case 0:
		enc.EncodeColor(drawColor{})
	case 1:
		enc.EncodeColor(newDrawColor(colorStops[0].Color.WithAlphaFactor(alpha)))
	default:
		enc.addRamp(colorStops, alpha, extend)
		enc.DrawTags = append(enc.DrawTags, DrawTagRadialGradient)
		enc.DrawData = append(enc.DrawData, safeish.AsBytes(&gradient)...)
	}
}

func (enc *Encoding) EncodeSweepGradient(
	gradient drawSweepGradient,
	colorStops []gfx.ColorStop,
	alpha float32,
	extend gfx.Extend,
) {
	const skiaDegenerateThreshold = 1.0 / (1 << 15)
	if jmath.Abs32(gradient.T0-gradient.T1) < skiaDegenerateThreshold {
		enc.EncodeColor(drawColor{})
		return
	}
	switch len(colorStops) {
	case 0:
		enc.EncodeColor(drawColor{})
	case 1:
		enc.EncodeColor(newDrawColor(colorStops[0].Color.WithAlphaFactor(alpha)))
	default:
		enc.addRamp(colorStops, alpha, extend)
		enc.DrawTags = append(enc.DrawTags, DrawTagSweepGradient)
		enc.DrawData = append(enc.DrawData, safeish.AsBytes(&gradient)...)
	}
}

func (enc *Encoding) EncodeImage(img gfx.Image, alpha float32) {
	enc.Resources.Patches = append(enc.Resources.Patches, ImagePatch{
		Image:          img,
		DrawDataOffset: len(enc.DrawData),
	})
	enc.DrawTags = append(enc.DrawTags, DrawTagImage)
	b := img.Image.Bounds().Canon()
	drawImg := drawImage{
		widthHeight: uint32(b.Dx()<<16 | b.Dy()&0xFFFF),
	}
	enc.DrawData = append(enc.DrawData, safeish.AsBytes(&drawImg)...)
}

func (enc *Encoding) EncodeBeginClip(blendMode gfx.BlendMode, alpha float32) {
	enc.DrawTags = append(enc.DrawTags, DrawTagBeginClip)
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
	enc.DrawTags = append(enc.DrawTags, DrawTagEndClip)
	// This is a dummy path, and will go away with the new clip impl.
	enc.PathTags = append(enc.PathTags, PathTagPath)
	enc.NumPaths++
	enc.NumClips++
	enc.NumOpenClips--
}

func (enc *Encoding) ForceNextTransformAndStyle() {
	enc.Flags |= forceNextTransform | forceNextStyle
}

func (enc *Encoding) SwapLastPathTags() {
	n := len(enc.PathTags)
	enc.PathTags[n-2], enc.PathTags[n-1] = enc.PathTags[n-1], enc.PathTags[n-2]
}

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

type Resources struct {
	Patches    []Patch
	ColorStops []gfx.ColorStop
	// XXX glyph stuff
}

func (r *Resources) Reset() {
	r.Patches = r.Patches[:0]
	r.ColorStops = r.ColorStops[:0]
}

// XXX the following types are in encoding/resolver.rs, but we put the resolver in renderer instead

type Patch interface {
	isPatch()
}

type RampPatch struct {
	DrawDataOffset int
	Stops          [2]int
	Extend         gfx.Extend
}

func (RampPatch) isPatch() {}

type ImagePatch struct {
	DrawDataOffset int
	Image          gfx.Image
}

func (ImagePatch) isPatch() {}
