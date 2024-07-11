package jello

import (
	"slices"

	"honnef.co/go/brush"
	"honnef.co/go/curve"
)

type Scene struct {
	Encoding  Encoding
	Estimator BumpEstimator
}

func (s *Scene) Reset() {
	s.Encoding.Reset()
	s.Estimator.Reset()
}

func (s *Scene) BumpEstimate(affine option[curve.Affine]) BumpAllocatorMemory {
	var trans option[Transform]
	if affine.isSet {
		trans.set(transformFromKurbo(affine.value))
	}
	return s.Estimator.Tally(trans)
}

func (sc *Scene) PushLayer(
	blend brush.BlendMode,
	alpha float32,
	transform curve.Affine,
	clip curve.Shape,
) {
	t := transformFromKurbo(transform)
	sc.Encoding.EncodeTransform(t)
	sc.Encoding.EncodeFillStyle(brush.NonZero)
	if !sc.Encoding.EncodeShape(clip, true) {
		// If the layer shape is invalid, encode a valid empty path. This suppresses
		// all drawing until the layer is popped.
		sc.Encoding.EncodeShape(curve.Rect{}, true)
	} else {
		sc.Estimator.CountPath(clip.PathElements(0.1), t, option[curve.Stroke]{})
	}
	sc.Encoding.EncodeBeginClip(blend, min(max(alpha, 0), 1))
}

func (sc *Scene) PopLayer() {
	sc.Encoding.EncodeEndClip()
}

func (sc *Scene) Fill(
	style brush.Fill,
	transform curve.Affine,
	brush brush.Brush,
	brushTransform *curve.Affine,
	shape curve.Shape,
) {
	t := transformFromKurbo(transform)
	sc.Encoding.EncodeTransform(t)
	sc.Encoding.EncodeFillStyle(style)
	if sc.Encoding.EncodeShape(shape, true) {
		if brushTransform != nil {
			if sc.Encoding.EncodeTransform(transformFromKurbo(transform.Mul(*brushTransform))) {
				sc.Encoding.SwapLastPathTags()
			}
		}
		sc.Encoding.EncodeBrush(brush, 1.0)
		sc.Estimator.CountPath(shape.PathElements(0.1), t, option[curve.Stroke]{})
	}
}

func (sc *Scene) Stroke(
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
		t := transformFromKurbo(transform)
		sc.Encoding.EncodeTransform(t)
		sc.Encoding.EncodeStrokeStyle(style)

		// We currently don't support dashing on the GPU. If the style has a dash pattern, then
		// we convert it into stroked paths on the CPU and encode those as individual draw
		// objects.
		var encodeResult bool
		if len(style.DashPattern) == 0 {
			sc.Estimator.CountPath(shape.PathElements(shapeTolerance), t, some(style))
			encodeResult = sc.Encoding.EncodeShape(shape, false)
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
			sc.Estimator.CountPath(slices.Values(dashed), t, some(style))
			encodeResult = sc.Encoding.EncodePathElements(slices.Values(dashed), false)
		}
		if encodeResult {
			if brushTransform != nil {
				if sc.Encoding.EncodeTransform(transformFromKurbo(transform.Mul(*brushTransform))) {
					sc.Encoding.SwapLastPathTags()
				}
			}
			sc.Encoding.EncodeBrush(b, 1.0)
		}
	} else {
		stroked := curve.StrokePath(
			shape.PathElements(shapeTolerance),
			style,
			curve.StrokeOpts{},
			strokeTolerance,
		)
		sc.Fill(brush.NonZero, transform, b, brushTransform, &stroked)
	}
}

func (sc *Scene) Append(other *Scene, transform option[curve.Affine]) {
	t := Identity
	if transform.isSet {
		t = transformFromKurbo(transform.value)
	}
	sc.Encoding.Append(&other.Encoding, t)
	sc.Estimator.Append(&other.Estimator, some(t))
}
