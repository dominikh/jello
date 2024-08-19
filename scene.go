// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package jello

import (
	"fmt"
	"slices"

	"honnef.co/go/curve"
	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/jello/renderer"
)

const debugTrace = false

type Scene struct {
	encoding  encoding.Encoding
	estimator renderer.BumpEstimator
}

func (s *Scene) Reset() {
	s.encoding.Reset()
	s.estimator.Reset()
}

// Encoding returns the scene encoding. The scene mustn't be manipulated
// concurrently with using the encoding.
func (s *Scene) Encoding() *encoding.Encoding {
	return &s.encoding
}

func (s *Scene) bumpEstimate(affine *curve.Affine) renderer.BumpAllocatorMemory {
	var trans *jmath.Transform
	if affine != nil {
		ret := jmath.TransformFromKurbo(*affine)
		trans = &ret
	}
	return s.estimator.Tally(trans)
}

func (s *Scene) PushLayer(
	blend gfx.BlendMode,
	alpha float32,
	clipTransform curve.Affine,
	clip curve.BezPath,
) {
	if debugTrace {
		fmt.Println("{")
		fmt.Println("\tvar clip curve.BezPath")
		for _, el := range clip {
			fmt.Printf("\tclip.Push(%#v)\n", el)
		}
		fmt.Printf("\ts.PushLayer(%#v, %g, %#v, clip)\n", blend, alpha, clipTransform)
		fmt.Println("}")
	}

	t := jmath.TransformFromKurbo(clipTransform)
	s.encoding.EncodeTransform(t)
	s.encoding.EncodeFillStyle(gfx.NonZero)
	if !s.encoding.EncodePath(clip, true) {
		// If the layer shape is invalid, encode a valid empty path. This suppresses
		// all drawing until the layer is popped.
		s.encoding.EncodePath(slices.Collect(curve.Rect{}.PathElements(0.1)), true)
		s.encoding.EncodeEmptyShape()
		path := curve.BezPath{
			curve.MoveTo(curve.Pt(0, 0)),
			curve.LineTo(curve.Pt(0, 0)),
		}
		s.estimator.CountPath(path, t, nil)
	} else {
		s.estimator.CountPath(clip, t, nil)
	}
	s.encoding.EncodeBeginClip(blend, min(max(alpha, 0), 1))
}

func (s *Scene) PopLayer() {
	if debugTrace {
		fmt.Println("s.PopLayer()")
	}

	s.encoding.EncodeEndClip()
}

func (s *Scene) Fill(
	style gfx.Fill,
	transform curve.Affine,
	brush gfx.Brush,
	brushTransform curve.Affine,
	path curve.BezPath,
) {
	if debugTrace {
		fmt.Println("{")
		fmt.Println("\tvar clip curve.BezPath")
		for _, el := range path {
			fmt.Printf("\tclip.Push(%#v)\n", el)
		}
		fmt.Printf("\ts.Fill(%d, %#v, %#v, %#v, clip)\n", style, transform, brush, brushTransform)
		fmt.Println("}")
	}

	t := jmath.TransformFromKurbo(transform)
	s.encoding.EncodeTransform(t)
	s.encoding.EncodeFillStyle(style)
	if s.encoding.EncodePath(path, true) {
		if brushTransform != curve.Identity {
			if s.encoding.EncodeTransform(jmath.TransformFromKurbo(transform.Mul(brushTransform))) {
				s.encoding.SwapLastPathTags()
			}
		}
		s.encoding.EncodeBrush(brush, 1.0)
		s.estimator.CountPath(path, t, nil)
	}
}

func (s *Scene) Stroke(
	style curve.Stroke,
	transform curve.Affine,
	b gfx.Brush,
	brushTransform curve.Affine,
	shape curve.BezPath,
) {
	// The setting for tolerance are a compromise. For most applications,
	// shape tolerance doesn't matter, as the input is likely Bézier paths,
	// which is exact. Note that shape tolerance is hard-coded as 0.1 in
	// the encoding crate.
	//
	// Stroke tolerance is a different matter. Generally, the cost scales
	// with inverse O(n^6), so there is moderate rendering cost to setting
	// too fine a value. On the other hand, error scales with the transform
	// applied post-stroking, so may exceed visible threshold. When we do
	// GPU-side stroking, the transform will be known. In the meantime,
	// this is a compromise.
	const shapeTolerance = 0.01
	const strokeTolerance = shapeTolerance

	if debugTrace {
		fmt.Println("{")
		fmt.Println("\tvar clip curve.BezPath")
		for _, el := range shape {
			fmt.Printf("\tclip.Push(%#v)\n", el)
		}
		fmt.Printf("\ts.Stroke(%#v, %#v, %#v, %#v, clip)\n", style, transform, b, brushTransform)
		fmt.Println("}")
	}

	const gpuStrokes = true // Set this to `true` to enable GPU-side stroking
	if gpuStrokes {
		t := jmath.TransformFromKurbo(transform)
		s.encoding.EncodeTransform(t)
		s.encoding.EncodeStrokeStyle(style)

		// We currently don't support dashing on the GPU. If the style has a dash pattern, then
		// we convert it into stroked paths on the CPU and encode those as individual draw
		// objects.
		var encodeResult bool
		if len(style.DashPattern) == 0 {
			s.estimator.CountPath(shape, t, &style)
			encodeResult = s.encoding.EncodePath(shape, false)
		} else {
			// TODO: We currently collect the output of the dash iterator because
			// `encode_path_elements` wants to consume the iterator. We want to avoid calling
			// `dash` twice when `bump_estimate` is enabled because it internally allocates.
			// Bump estimation will move to resolve time rather than scene construction time,
			// so we can revert this back to not collecting when that happens.
			dashed := slices.Collect(curve.Dash(
				slices.Values(shape),
				style.DashOffset,
				style.DashPattern,
			))
			// We turn the iterator into a slice and then turn it into an
			// iterator again to avoid doing the curve.Dash work twice.
			s.estimator.CountPath(dashed, t, &style)
			encodeResult = s.encoding.EncodePath(dashed, false)
		}
		if encodeResult {
			if brushTransform != curve.Identity {
				if s.encoding.EncodeTransform(jmath.TransformFromKurbo(transform.Mul(brushTransform))) {
					s.encoding.SwapLastPathTags()
				}
			}
			s.encoding.EncodeBrush(b, 1.0)
		}
	} else {
		stroked := curve.StrokePath(
			slices.Values(shape),
			style,
			curve.StrokeOpts{},
			strokeTolerance,
		)
		s.Fill(gfx.NonZero, transform, b, brushTransform, stroked)
	}
}

func (s *Scene) Append(other *Scene, transform curve.Affine) {
	// OPT(dh): we'd like to combine multiple scenes without having to copy data around

	t := jmath.TransformFromKurbo(transform)
	s.encoding.Append(&other.encoding, t)
	s.estimator.Append(&other.estimator, &t)
}

// ApplyTransform applies an affine transformation to everything that has
// already been drawn in the scene. It does not affect future drawing
// operations.
//
// This can be used, for example, to apply HiDPI scaling.
func (s *Scene) ApplyTransform(transform curve.Affine) {
	t := jmath.TransformFromKurbo(transform)
	s.encoding.ApplyTransform(t)
}
