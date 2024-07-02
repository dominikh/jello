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

func (self *Scene) PushLayer(
	blend brush.BlendMode,
	alpha float32,
	transform curve.Affine,
	clip curve.Shape,
) {
	t := transformFromKurbo(transform)
	self.Encoding.EncodeTransform(t)
	self.Encoding.EncodeFillStyle(brush.NonZero)
	if !self.Encoding.EncodeShape(clip, true) {
		// If the layer shape is invalid, encode a valid empty path. This suppresses
		// all drawing until the layer is popped.
		self.Encoding.EncodeShape(curve.Rect{}, true)
	} else {
		self.Estimator.CountPath(clip.PathElements(0.1), t, option[curve.Stroke]{})
	}
	self.Encoding.EncodeBeginClip(blend, min(max(alpha, 0), 1))
}

func (self *Scene) PopLayer() {
	self.Encoding.EncodeEndClip()
}

func (self *Scene) Fill(
	style brush.Fill,
	transform curve.Affine,
	brush brush.Brush,
	brush_transform *curve.Affine,
	shape curve.Shape,
) {
	t := transformFromKurbo(transform)
	self.Encoding.EncodeTransform(t)
	self.Encoding.EncodeFillStyle(style)
	if self.Encoding.EncodeShape(shape, true) {
		if brush_transform != nil {
			if self.Encoding.EncodeTransform(transformFromKurbo(transform.Mul(*brush_transform))) {
				self.Encoding.SwapLastPathTags()
			}
		}
		self.Encoding.EncodeBrush(brush, 1.0)
		self.Estimator.CountPath(shape.PathElements(0.1), t, option[curve.Stroke]{})
	}
}

func (self *Scene) Stroke(
	style curve.Stroke,
	transform curve.Affine,
	b brush.Brush,
	brush_transform *curve.Affine,
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
	const SHAPE_TOLERANCE = 0.01
	const STROKE_TOLERANCE = SHAPE_TOLERANCE

	const GPU_STROKES = true // Set this to `true` to enable GPU-side stroking
	if GPU_STROKES {
		t := transformFromKurbo(transform)
		self.Encoding.EncodeTransform(t)
		self.Encoding.EncodeStrokeStyle(style)

		// We currently don't support dashing on the GPU. If the style has a dash pattern, then
		// we convert it into stroked paths on the CPU and encode those as individual draw
		// objects.
		var encode_result bool
		if len(style.DashPattern) == 0 {
			self.Estimator.CountPath(shape.PathElements(SHAPE_TOLERANCE), t, some(style))
			encode_result = self.Encoding.EncodeShape(shape, false)
		} else {
			// TODO: We currently collect the output of the dash iterator because
			// `encode_path_elements` wants to consume the iterator. We want to avoid calling
			// `dash` twice when `bump_estimate` is enabled because it internally allocates.
			// Bump estimation will move to resolve time rather than scene construction time,
			// so we can revert this back to not collecting when that happens.
			dashed := slices.Collect(curve.Dash(
				shape.PathElements(SHAPE_TOLERANCE),
				style.DashOffset,
				style.DashPattern,
			))
			// We turn the iterator into a slice and then turn it into an
			// iterator again to avoid doing the curve.Dash work twice.
			self.Estimator.CountPath(slices.Values(dashed), t, some(style))
			encode_result = self.Encoding.EncodePathElements(slices.Values(dashed), false)
		}
		if encode_result {
			if brush_transform != nil {
				if self.Encoding.EncodeTransform(transformFromKurbo(transform.Mul(*brush_transform))) {
					self.Encoding.SwapLastPathTags()
				}
			}
			self.Encoding.EncodeBrush(b, 1.0)
		}
	} else {
		stroked := curve.StrokePath(
			shape.PathElements(SHAPE_TOLERANCE),
			style,
			curve.StrokeOpts{},
			STROKE_TOLERANCE,
		)
		self.Fill(brush.NonZero, transform, b, brush_transform, &stroked)
	}
}

func (self *Scene) Append(other *Scene, transform option[curve.Affine]) {
	// let t = transform.as_ref().map(Transform::from_kurbo);
	t := Identity
	if transform.isSet {
		t = transformFromKurbo(transform.value)
	}
	self.Encoding.Append(&other.Encoding, t)
	self.Estimator.Append(&other.Estimator, some(t))
}
