package jello

import (
	"slices"

	"honnef.co/go/brush"
	"honnef.co/go/curve"
	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/jello/renderer"
)

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
	blend brush.BlendMode,
	alpha float32,
	transform curve.Affine,
	clip curve.Shape,
) {
	t := jmath.TransformFromKurbo(transform)
	s.encoding.EncodeTransform(t)
	s.encoding.EncodeFillStyle(brush.NonZero)
	if !s.encoding.EncodeShape(clip, true) {
		// If the layer shape is invalid, encode a valid empty path. This suppresses
		// all drawing until the layer is popped.
		s.encoding.EncodeShape(curve.Rect{}, true)
	} else {
		s.estimator.CountPath(clip.PathElements(0.1), t, nil)
	}
	s.encoding.EncodeBeginClip(blend, min(max(alpha, 0), 1))
}

func (s *Scene) PopLayer() {
	s.encoding.EncodeEndClip()
}

func (s *Scene) Fill(
	style brush.Fill,
	transform curve.Affine,
	brush brush.Brush,
	brushTransform *curve.Affine,
	shape curve.Shape,
) {
	t := jmath.TransformFromKurbo(transform)
	s.encoding.EncodeTransform(t)
	s.encoding.EncodeFillStyle(style)
	if s.encoding.EncodeShape(shape, true) {
		if brushTransform != nil {
			if s.encoding.EncodeTransform(jmath.TransformFromKurbo(transform.Mul(*brushTransform))) {
				s.encoding.SwapLastPathTags()
			}
		}
		s.encoding.EncodeBrush(brush, 1.0)
		s.estimator.CountPath(shape.PathElements(0.1), t, nil)
	}
}

func (s *Scene) Stroke(
	style curve.Stroke,
	transform curve.Affine,
	b brush.Brush,
	brushTransform *curve.Affine,
	shape curve.Shape,
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
			s.estimator.CountPath(shape.PathElements(shapeTolerance), t, &style)
			encodeResult = s.encoding.EncodeShape(shape, false)
		} else {
			// TODO: We currently collect the output of the dash iterator because
			// `encode_path_elements` wants to consume the iterator. We want to avoid calling
			// `dash` twice when `bump_estimate` is enabled because it internally allocates.
			// Bump estimation will move to resolve time rather than scene construction time,
			// so we can revert this back to not collecting when that happens.
			dashed := slices.Collect(curve.Dash(
				shape.PathElements(shapeTolerance),
				style.DashOffset,
				style.DashPattern,
			))
			// We turn the iterator into a slice and then turn it into an
			// iterator again to avoid doing the curve.Dash work twice.
			s.estimator.CountPath(slices.Values(dashed), t, &style)
			encodeResult = s.encoding.EncodePathElements(slices.Values(dashed), false)
		}
		if encodeResult {
			if brushTransform != nil {
				if s.encoding.EncodeTransform(jmath.TransformFromKurbo(transform.Mul(*brushTransform))) {
					s.encoding.SwapLastPathTags()
				}
			}
			s.encoding.EncodeBrush(b, 1.0)
		}
	} else {
		stroked := curve.StrokePath(
			shape.PathElements(shapeTolerance),
			style,
			curve.StrokeOpts{},
			strokeTolerance,
		)
		s.Fill(brush.NonZero, transform, b, brushTransform, &stroked)
	}
}

func (s *Scene) Append(other *Scene, transform *curve.Affine) {
	t := jmath.Identity
	if transform != nil {
		t = jmath.TransformFromKurbo(*transform)
	}
	s.encoding.Append(&other.encoding, t)
	s.estimator.Append(&other.estimator, &t)
}
