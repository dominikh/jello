package jello

import (
	"iter"
	"math"

	"honnef.co/go/curve"
)

//! This utility provides conservative size estimation for buffer allocations backing
//! GPU bump memory. This estimate relies on heuristics and naturally overestimates.

// use super::{BumpAllocatorMemory, BumpAllocators, Transform};
// use peniko::kurbo::{Cap, Join, PathEl, Point, Stroke, Vec2};

const RSQRT_OF_TOL = 2.2360679775 // tol = 0.2

type BumpEstimator struct {
	// TODO: support binning
	// TODO: support ptcl
	// TODO: support tile

	// NOTE: The segment count estimation could use further refinement, particularly to handle
	// viewport clipping and rotation applied to fragments during append. We can produce a more
	// optimal result under scale and rotation if we track more data for each shape during insertion
	// and defer the final tally to resolve-time (in which we could evaluate the estimates using
	// precisely transformed coordinates). For now we apply a fudge factor of sqrt(2) and inflate
	// the number of tile crossing (a near~diagonal line orientation would result in worst case for
	// the number of intersected tiles) to account for this.
	//
	// Accounting for viewport clipping (for the right and bottom edges of the viewport) is simply
	// impossible at insertion time as the render target dimensions are unknown. We could
	// potentially account for clipping (including clip shapes/layers) by tracking bounding boxes
	// during insertion and resolving all clips at tally time (e.g. one could come up with a
	// heuristic for scaling the counts based on the proportions of a clipped bbox area).
	//
	// Since we currently don't account for clipping, this will always overshoot when clips are
	// present and when the bounding box of a shape is partially or wholly outside the viewport.
	segments uint32
	lines    estimateLineSoup
}

func (be *BumpEstimator) Reset() {
	*be = BumpEstimator{}
}

// impl BumpEstimator {
// / Combine the counts of this estimator with `other` after applying an optional `transform`.
func (be *BumpEstimator) Append(other *BumpEstimator, transform option[Transform]) {
	scale := transform_scale(transform)
	be.segments += uint32(math.Ceil(float64(other.segments) * scale))
	be.lines.add(&other.lines, scale)
}

func (self *BumpEstimator) CountPath(path iter.Seq[curve.PathElement], t Transform, stroke option[curve.Stroke]) {
	caps := uint32(1)
	fill_close_lines := uint32(1)
	var joins, lineto_lines, curve_lines, curve_count, segments uint32

	// Track the path state to correctly count empty paths and close joins.
	var first_pt, last_pt option[curve.Point]
	scale := transform_scale(some(t))
	var scaled_width float64
	if stroke.isSet {
		scaled_width = stroke.value.Width * scale
	}
	offset_fudge := max(1, math.Sqrt(scaled_width))
	for el := range path {
		switch el.Kind {
		case curve.MoveToKind:
			first_pt.set(el.P0)
			if !last_pt.isSet {
				continue
			}
			caps += 1
			if joins > 0 {
				joins--
			}
			fill_close_lines += 1
			segments += count_segments_for_line(first_pt.unwrap(), last_pt.unwrap(), t)
			last_pt.clear()
		case curve.ClosePathKind:
			if last_pt.isSet {
				joins += 1
				lineto_lines += 1
				segments += count_segments_for_line(first_pt.unwrap(), last_pt.unwrap(), t)
			}
			last_pt = first_pt
		case curve.LineToKind:
			last_pt.set(el.P0)
			joins += 1
			lineto_lines += 1
			segments += count_segments_for_line(first_pt.unwrap(), last_pt.unwrap(), t)
		case curve.QuadToKind:
			var p0 curve.Vec2
			if last_pt.isSet {
				p0 = curve.Vec2(last_pt.value)
			} else if first_pt.isSet {
				p0 = curve.Vec2(first_pt.value)
			} else {
				continue
			}
			last_pt.set(el.P1)

			p1 := curve.Vec2(el.P0)
			p2 := curve.Vec2(el.P1)
			lines := offset_fudge * wangQuadratic(RSQRT_OF_TOL, p0, p1, p2, t)

			curve_lines += uint32(math.Ceil(lines))
			curve_count++
			joins++

			segs := offset_fudge * count_segments_for_quadratic(p0, p1, p2, t)
			segments += uint32(max(math.Ceil(segs), math.Ceil(lines)))
		case curve.CubicToKind:
			var p0 curve.Vec2
			if last_pt.isSet {
				p0 = curve.Vec2(last_pt.value)
			} else if first_pt.isSet {
				p0 = curve.Vec2(first_pt.value)
			} else {
				continue
			}
			last_pt.set(el.P2)

			p1 := curve.Vec2(el.P0)
			p2 := curve.Vec2(el.P1)
			p3 := curve.Vec2(el.P2)
			lines := offset_fudge * wangCubic(RSQRT_OF_TOL, p0, p1, p2, p3, t)

			curve_lines += uint32(math.Ceil(lines))
			curve_count += 1
			joins += 1
			segs := count_segments_for_cubic(p0, p1, p2, p3, t)
			segments += uint32(max(math.Ceil(segs), math.Ceil(lines)))
		}
	}

	if !stroke.isSet {
		self.lines.linetos += lineto_lines + fill_close_lines
		self.lines.curves += curve_lines
		self.lines.curve_count += curve_count
		self.segments += segments

		// Account for the implicit close
		if first_pt.isSet && last_pt.isSet {
			self.segments += count_segments_for_line(first_pt.value, last_pt.value, t)
		}
		return
	}
	style := stroke.value

	// For strokes, double-count the lines to estimate offset curves.
	self.lines.linetos += 2 * lineto_lines
	self.lines.curves += 2 * curve_lines
	self.lines.curve_count += 2 * curve_count
	self.segments += 2 * segments

	self.count_stroke_caps(style.StartCap, scaled_width, caps)
	self.count_stroke_caps(style.EndCap, scaled_width, caps)
	self.count_stroke_joins(style.Join, scaled_width, style.MiterLimit, joins)
}

// / Produce the final total, applying an optional transform to all content.
func (self *BumpEstimator) Tally(transform option[Transform]) BumpAllocatorMemory {
	scale := transform_scale(transform)

	// The post-flatten line estimate.
	lines := self.lines.tally(scale)

	// The estimate for tile crossings for lines. Here we ensure that there are at least as many
	// segments as there are lines, in case `segments` was underestimated at small scales.
	n_segments := max(lines, uint32(math.Ceil(float64(self.segments)*scale)), lines)

	bump := BumpAllocators{
		Failed: 0,
		// TODO: we can provide a tighter bound here but for now we
		// assume that binning must be bounded by the segment count.
		Binning:   n_segments,
		Ptcl:      0,
		Tile:      0,
		Blend:     0,
		SegCounts: n_segments,
		Segments:  n_segments,
		Lines:     lines,
	}
	return bump.Memory()
}

func (self *BumpEstimator) count_stroke_caps(style curve.Cap, scaled_width float64, count uint32) {
	switch style {
	case curve.ButtCap:
		self.lines.linetos += count
		self.segments += count_segments_for_line_length(scaled_width) * count
	case curve.SquareCap:
		self.lines.linetos += 3 * count
		self.segments += count_segments_for_line_length(scaled_width) * count
		self.segments += 2 * count_segments_for_line_length(0.5*scaled_width) * count
	case curve.RoundCap:
		arc_lines, line_len := estimate_arc_lines(scaled_width)
		self.lines.curves += count * arc_lines
		self.lines.curve_count += 1
		self.segments += count * arc_lines * count_segments_for_line_length(line_len)
	}
}

func (self *BumpEstimator) count_stroke_joins(style curve.Join, scaled_width float64, miter_limit float64, count uint32) {
	switch style {
	case curve.BevelJoin:
		self.lines.linetos += count
		self.segments += count_segments_for_line_length(scaled_width) * count
	case curve.MiterJoin:
		max_miter_len := scaled_width * miter_limit
		self.lines.linetos += 2 * count
		self.segments += 2 * count * count_segments_for_line_length(max_miter_len)
	case curve.RoundJoin:
		arc_lines, line_len := estimate_arc_lines(scaled_width)
		self.lines.curves += count * arc_lines
		self.lines.curve_count += 1
		self.segments += count * arc_lines * count_segments_for_line_length(line_len)
	}

	// Count inner join lines
	self.lines.linetos += count
	self.segments += count_segments_for_line_length(scaled_width) * count
}

func estimate_arc_lines(scaledStrokeWidth float64) (uint32, float64) {
	// These constants need to be kept consistent with the definitions in `flatten_arc` in
	// flatten.wgsl.
	// TODO: It would be better if these definitions were shared/configurable. For example an
	// option is for all tolerances to be parameters to the estimator as well as the GPU pipelines
	// (the latter could be in the form of a config uniform) which would help to keep them in
	// sync.
	const MIN_THETA = 1e-6
	const TOL = 0.25
	radius := max(TOL, scaledStrokeWidth*0.5)
	theta := max(2.0*math.Acos(1.-TOL/radius), MIN_THETA)
	arc_lines := max(2, uint32(math.Ceil(math.Pi/2/theta)))
	return arc_lines, 2.0 * math.Sin(theta) * radius
}

type estimateLineSoup struct {
	// Explicit lines (such as linetos and non-round stroke caps/joins) and Bezier curves
	// get tracked separately to ensure that explicit lines remain scale invariant.
	linetos uint32
	curves  uint32

	// Curve count is simply used to ensure a minimum number of lines get counted for each curve
	// at very small scales to reduce the chances of an under-estimate.
	curve_count uint32
}

func (ls *estimateLineSoup) tally(scale float64) uint32 {
	curves := max(ls.scaled_curve_line_count(scale), 5*ls.curve_count)
	return ls.linetos + curves
}

func (ls *estimateLineSoup) scaled_curve_line_count(scale float64) uint32 {
	return uint32(math.Ceil(float64(ls.curves) * math.Sqrt(scale)))
}

func (ls *estimateLineSoup) add(other *estimateLineSoup, scale float64) {
	ls.linetos += other.linetos
	ls.curves += other.scaled_curve_line_count(scale)
	ls.curve_count += other.curve_count
}

// TODO: The 32-bit Vec2 definition from cpu_shaders/util.rs could come in handy here.
func transform(t Transform, v curve.Vec2) curve.Vec2 {
	return curve.Vec(
		float64(t.Matrix[0])*v.X+float64(t.Matrix[2])*v.Y,
		float64(t.Matrix[1])*v.X+float64(t.Matrix[3])*v.Y,
	)
}

func transform_scale(t option[Transform]) float64 {
	if t.isSet {
		m := t.value.Matrix
		v1x := float64(m[0]) + float64(m[3])
		v2x := float64(m[0]) - float64(m[3])
		v1y := float64(m[1]) - float64(m[2])
		v2y := float64(m[1]) + float64(m[2])
		return math.Sqrt(v1x*v1x+v1y*v1y) + math.Sqrt(v2x*v2x+v2y*v2y)
	} else {
		return 1.0
	}
}

func approx_arc_length_cubic(p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, p3 curve.Vec2) float64 {
	chord_len := (p3.Sub(p0)).Hypot()
	// Length of the control polygon
	poly_len := (p1.Sub(p0)).Hypot() + (p2.Sub(p1)).Hypot() + (p3.Sub(p2)).Hypot()
	return 0.5 * (chord_len + poly_len)
}

func count_segments_for_cubic(p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, p3 curve.Vec2, t Transform) float64 {
	p0 = transform(t, p0)
	p1 = transform(t, p1)
	p2 = transform(t, p2)
	p3 = transform(t, p3)
	return math.Ceil(approx_arc_length_cubic(p0, p1, p2, p3) * 0.0625 * math.Sqrt2)
}

func count_segments_for_quadratic(p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, t Transform) float64 {
	return count_segments_for_cubic(p0, p1.Lerp(p0, 0.333333), p1.Lerp(p2, 0.333333), p2, t)
}

// Estimate tile crossings for a line with known endpoints.
func count_segments_for_line(p0 curve.Point, p1 curve.Point, t Transform) uint32 {
	dxdy := p0.Sub(p1)
	dxdy = transform(t, dxdy)
	segments := math.Ceil(math.Ceil(math.Abs(dxdy.X))*0.0625) + math.Ceil(math.Ceil(math.Abs(dxdy.Y))*0.0625)
	return max(1, uint32(segments))
}

// Estimate tile crossings for a line with a known length.
func count_segments_for_line_length(scaled_width float64) uint32 {
	// scale the tile count by sqrt(2) to allow some slack for diagonal lines.
	// TODO: Would "2" be a better factor?
	return max(1, uint32(math.Ceil(scaled_width*0.0625*math.Sqrt2)))
}

/// Wang's Formula (as described in Pyramid Algorithms by Ron Goldman, 2003, Chapter 5, Section
/// 5.6.3 on Bezier Approximation) is a fast method for computing a lower bound on the number of
/// recursive subdivisions required to approximate a Bezier curve within a certain tolerance. The
/// formula for a Bezier curve of degree `n`, control points `p[0]...p[n]`, and number of levels of
/// subdivision `l`, and flattening tolerance `tol` is defined as follows:
///
/// ```ignore
///     m = max([length(p[k+2] - 2 * p[k+1] + p[k]) for (0 <= k <= n-2)])
///     l >= log_4((n * (n - 1) * m) / (8 * tol))
/// ```
///
/// For recursive subdivisions that split a curve into 2 segments at each level, the minimum number
/// of segments is given by 2^l. From the formula above it follows that:
///
/// ```ignore
///       segments >= 2^l >= 2^log_4(x)                      (1)
///     segments^2 >= 2^(2*log_4(x)) >= 4^log_4(x)           (2)
///     segments^2 >= x
///       segments >= sqrt((n * (n - 1) * m) / (8 * tol))    (3)
/// ```
///
/// Wang's formula computes an error bound on recursive subdivision based on the second derivative
/// which tends to result in a suboptimal estimate when the curvature within the curve has a lot of
/// variation. This is expected to frequently overshoot the flattening formula used in vello, which
/// is closer to optimal (vello uses a method based on a numerical approximation of the integral
/// over the continuous change in the number of flattened segments, with an error expressed in terms
/// of curvature and infinitesimal arclength).

// The curve degree term sqrt(n * (n - 1) / 8) specialized for cubics:
//
//	sqrt(3 * (3 - 1) / 8)
const SQRT_OF_DEGREE_TERM_CUBIC = 0.86602540378

// The curve degree term sqrt(n * (n - 1) / 8) specialized for quadratics:
//
//	sqrt(2 * (2 - 1) / 8)
const SQRT_OF_DEGREE_TERM_QUAD = 0.5

func wangQuadratic(rsqrt_of_tol float64, p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, t Transform) float64 {
	v := p1.Mul(-2).Add(p0).Add(p2)
	v = transform(t, v) // transform is distributive
	m := v.Hypot()
	return math.Ceil(SQRT_OF_DEGREE_TERM_QUAD * math.Sqrt(m) * rsqrt_of_tol)
}

func wangCubic(rsqrt_of_tol float64, p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, p3 curve.Vec2, t Transform) float64 {
	v1 := p1.Mul(-2).Add(p0).Add(p2)
	v2 := p2.Mul(-2).Add(p1).Add(p3)
	v1 = transform(t, v1)
	v2 = transform(t, v2)
	m := max(v1.Hypot(), v2.Hypot())
	return math.Ceil(SQRT_OF_DEGREE_TERM_CUBIC * math.Sqrt(m) * rsqrt_of_tol)
}
