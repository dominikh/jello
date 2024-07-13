package jello

import (
	"iter"
	"slices"

	"honnef.co/go/jello/gfx"
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
	blend gfx.BlendMode,
	alpha float32,
	transform curve.Affine,
	clip iter.Seq[curve.PathElement],
) {
	t := jmath.TransformFromKurbo(transform)
	s.encoding.EncodeTransform(t)
	s.encoding.EncodeFillStyle(gfx.NonZero)
	if !s.encoding.EncodePathElements(clip, true) {
		// If the layer shape is invalid, encode a valid empty path. This suppresses
		// all drawing until the layer is popped.
		s.encoding.EncodePathElements(curve.Rect{}.PathElements(0.1), true)
	} else {
		s.estimator.CountPath(clip, t, nil)
	}
	s.encoding.EncodeBeginClip(blend, min(max(alpha, 0), 1))
}

func (s *Scene) PopLayer() {
	s.encoding.EncodeEndClip()
}

func (s *Scene) Fill(
	style gfx.Fill,
	transform curve.Affine,
	brush gfx.Brush,
	brushTransform *curve.Affine,
	path iter.Seq[curve.PathElement],
) {
	t := jmath.TransformFromKurbo(transform)
	s.encoding.EncodeTransform(t)
	s.encoding.EncodeFillStyle(style)
	if s.encoding.EncodePathElements(path, true) {
		if brushTransform != nil {
			if s.encoding.EncodeTransform(jmath.TransformFromKurbo(transform.Mul(*brushTransform))) {
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
	brushTransform *curve.Affine,
	shape iter.Seq[curve.PathElement],
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
			s.estimator.CountPath(shape, t, &style)
			encodeResult = s.encoding.EncodePathElements(shape, false)
		} else {
			// TODO: We currently collect the output of the dash iterator because
			// `encode_path_elements` wants to consume the iterator. We want to avoid calling
			// `dash` twice when `bump_estimate` is enabled because it internally allocates.
			// Bump estimation will move to resolve time rather than scene construction time,
			// so we can revert this back to not collecting when that happens.
			dashed := slices.Collect(curve.Dash(
				shape,
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
			shape,
			style,
			curve.StrokeOpts{},
			strokeTolerance,
		)
		s.Fill(gfx.NonZero, transform, b, brushTransform, stroked.Elements())
	}
}

func (s *Scene) Append(other *Scene, transform curve.Affine) {
	// OPT(dh): we'd like to combine multiple scenes without having to copy data around

	t := jmath.TransformFromKurbo(transform)
	s.encoding.Append(&other.encoding, t)
	s.estimator.Append(&other.estimator, &t)
}
