package renderer

import (
	"iter"
	"math"

	"honnef.co/go/curve"
	"honnef.co/go/jello/jmath"
)

//! This utility provides conservative size estimation for buffer allocations backing
//! GPU bump memory. This estimate relies on heuristics and naturally overestimates.

// use super::{BumpAllocatorMemory, BumpAllocators, Transform};
// use peniko::kurbo::{Cap, Join, PathEl, Point, Stroke, Vec2};

const rsqrtOfTol = 2.2360679775 // tol = 0.2

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
func (be *BumpEstimator) Append(other *BumpEstimator, transform *jmath.Transform) {
	scale := transformScale(transform)
	be.segments += uint32(math.Ceil(float64(other.segments) * scale))
	be.lines.add(&other.lines, scale)
}

func (est *BumpEstimator) CountPath(path iter.Seq[curve.PathElement], t jmath.Transform, stroke *curve.Stroke) {
	caps := uint32(1)
	fillCloseLines := uint32(1)
	var joins, lineToLines, curveLines, curveCount, segments uint32

	// Track the path state to correctly count empty paths and close joins.
	var firstPt, lastPt option[curve.Point]
	scale := transformScale(&t)
	var scaledWidth float64
	if stroke != nil {
		scaledWidth = stroke.Width * scale
	}
	offsetFudge := max(1, math.Sqrt(scaledWidth))
	for el := range path {
		switch el.Kind {
		case curve.MoveToKind:
			firstPt.set(el.P0)
			if !lastPt.isSet {
				continue
			}
			caps += 1
			if joins > 0 {
				joins--
			}
			fillCloseLines += 1
			segments += countSegmentsForLine(firstPt.unwrap(), lastPt.unwrap(), t)
			lastPt.clear()
		case curve.ClosePathKind:
			if lastPt.isSet {
				joins += 1
				lineToLines += 1
				segments += countSegmentsForLine(firstPt.unwrap(), lastPt.unwrap(), t)
			}
			lastPt = firstPt
		case curve.LineToKind:
			lastPt.set(el.P0)
			joins += 1
			lineToLines += 1
			segments += countSegmentsForLine(firstPt.unwrap(), lastPt.unwrap(), t)
		case curve.QuadToKind:
			var p0 curve.Vec2
			if lastPt.isSet {
				p0 = curve.Vec2(lastPt.value)
			} else if firstPt.isSet {
				p0 = curve.Vec2(firstPt.value)
			} else {
				continue
			}
			lastPt.set(el.P1)

			p1 := curve.Vec2(el.P0)
			p2 := curve.Vec2(el.P1)
			lines := offsetFudge * wangQuadratic(rsqrtOfTol, p0, p1, p2, t)

			curveLines += uint32(math.Ceil(lines))
			curveCount++
			joins++

			segs := offsetFudge * countSegmentsForQuadratic(p0, p1, p2, t)
			segments += uint32(max(math.Ceil(segs), math.Ceil(lines)))
		case curve.CubicToKind:
			var p0 curve.Vec2
			if lastPt.isSet {
				p0 = curve.Vec2(lastPt.value)
			} else if firstPt.isSet {
				p0 = curve.Vec2(firstPt.value)
			} else {
				continue
			}
			lastPt.set(el.P2)

			p1 := curve.Vec2(el.P0)
			p2 := curve.Vec2(el.P1)
			p3 := curve.Vec2(el.P2)
			lines := offsetFudge * wangCubic(rsqrtOfTol, p0, p1, p2, p3, t)

			curveLines += uint32(math.Ceil(lines))
			curveCount += 1
			joins += 1
			segs := countSegmentsForCubic(p0, p1, p2, p3, t)
			segments += uint32(max(math.Ceil(segs), math.Ceil(lines)))
		}
	}

	if stroke == nil {
		est.lines.linetos += lineToLines + fillCloseLines
		est.lines.curves += curveLines
		est.lines.curveCount += curveCount
		est.segments += segments

		// Account for the implicit close
		if firstPt.isSet && lastPt.isSet {
			est.segments += countSegmentsForLine(firstPt.value, lastPt.value, t)
		}
		return
	}
	style := *stroke

	// For strokes, double-count the lines to estimate offset curves.
	est.lines.linetos += 2 * lineToLines
	est.lines.curves += 2 * curveLines
	est.lines.curveCount += 2 * curveCount
	est.segments += 2 * segments

	est.countStrokeCaps(style.StartCap, scaledWidth, caps)
	est.countStrokeCaps(style.EndCap, scaledWidth, caps)
	est.countStrokeJoins(style.Join, scaledWidth, style.MiterLimit, joins)
}

// / Produce the final total, applying an optional transform to all content.
func (est *BumpEstimator) Tally(transform *jmath.Transform) BumpAllocatorMemory {
	scale := transformScale(transform)

	// The post-flatten line estimate.
	lines := est.lines.tally(scale)

	// The estimate for tile crossings for lines. Here we ensure that there are at least as many
	// segments as there are lines, in case `segments` was underestimated at small scales.
	numSegments := max(lines, uint32(math.Ceil(float64(est.segments)*scale)), lines)

	bump := BumpAllocators{
		Failed: 0,
		// TODO: we can provide a tighter bound here but for now we
		// assume that binning must be bounded by the segment count.
		Binning:   numSegments,
		Ptcl:      0,
		Tile:      0,
		Blend:     0,
		SegCounts: numSegments,
		Segments:  numSegments,
		Lines:     lines,
	}
	return bump.Memory()
}

func (est *BumpEstimator) countStrokeCaps(style curve.Cap, scaledWidth float64, count uint32) {
	switch style {
	case curve.ButtCap:
		est.lines.linetos += count
		est.segments += countSegmentsForLineLength(scaledWidth) * count
	case curve.SquareCap:
		est.lines.linetos += 3 * count
		est.segments += countSegmentsForLineLength(scaledWidth) * count
		est.segments += 2 * countSegmentsForLineLength(0.5*scaledWidth) * count
	case curve.RoundCap:
		arcLines, lineLen := estimateArcLines(scaledWidth)
		est.lines.curves += count * arcLines
		est.lines.curveCount += 1
		est.segments += count * arcLines * countSegmentsForLineLength(lineLen)
	}
}

func (est *BumpEstimator) countStrokeJoins(style curve.Join, scaledWidth float64, miterLimit float64, count uint32) {
	switch style {
	case curve.BevelJoin:
		est.lines.linetos += count
		est.segments += countSegmentsForLineLength(scaledWidth) * count
	case curve.MiterJoin:
		maxMiterLen := scaledWidth * miterLimit
		est.lines.linetos += 2 * count
		est.segments += 2 * count * countSegmentsForLineLength(maxMiterLen)
	case curve.RoundJoin:
		arcLines, lineLen := estimateArcLines(scaledWidth)
		est.lines.curves += count * arcLines
		est.lines.curveCount += 1
		est.segments += count * arcLines * countSegmentsForLineLength(lineLen)
	}

	// Count inner join lines
	est.lines.linetos += count
	est.segments += countSegmentsForLineLength(scaledWidth) * count
}

func estimateArcLines(scaledStrokeWidth float64) (uint32, float64) {
	// These constants need to be kept consistent with the definitions in `flatten_arc` in
	// flatten.wgsl.
	// TODO: It would be better if these definitions were shared/configurable. For example an
	// option is for all tolerances to be parameters to the estimator as well as the GPU pipelines
	// (the latter could be in the form of a config uniform) which would help to keep them in
	// sync.
	const minTheta = 1e-6
	const tol = 0.25
	radius := max(tol, scaledStrokeWidth*0.5)
	theta := max(2.0*math.Acos(1.-tol/radius), minTheta)
	arcLines := max(2, uint32(math.Ceil(math.Pi/2/theta)))
	return arcLines, 2.0 * math.Sin(theta) * radius
}

type estimateLineSoup struct {
	// Explicit lines (such as linetos and non-round stroke caps/joins) and Bezier curves
	// get tracked separately to ensure that explicit lines remain scale invariant.
	linetos uint32
	curves  uint32

	// Curve count is simply used to ensure a minimum number of lines get counted for each curve
	// at very small scales to reduce the chances of an under-estimate.
	curveCount uint32
}

func (ls *estimateLineSoup) tally(scale float64) uint32 {
	curves := max(ls.scaledCurveLineCount(scale), 5*ls.curveCount)
	return ls.linetos + curves
}

func (ls *estimateLineSoup) scaledCurveLineCount(scale float64) uint32 {
	return uint32(math.Ceil(float64(ls.curves) * math.Sqrt(scale)))
}

func (ls *estimateLineSoup) add(other *estimateLineSoup, scale float64) {
	ls.linetos += other.linetos
	ls.curves += other.scaledCurveLineCount(scale)
	ls.curveCount += other.curveCount
}

// TODO: The 32-bit Vec2 definition from cpu_shaders/util.rs could come in handy here.
func transform(t jmath.Transform, v curve.Vec2) curve.Vec2 {
	return curve.Vec(
		float64(t.Matrix[0])*v.X+float64(t.Matrix[2])*v.Y,
		float64(t.Matrix[1])*v.X+float64(t.Matrix[3])*v.Y,
	)
}

func transformScale(t *jmath.Transform) float64 {
	if t != nil {
		m := t.Matrix
		v1x := float64(m[0]) + float64(m[3])
		v2x := float64(m[0]) - float64(m[3])
		v1y := float64(m[1]) - float64(m[2])
		v2y := float64(m[1]) + float64(m[2])
		return math.Sqrt(v1x*v1x+v1y*v1y) + math.Sqrt(v2x*v2x+v2y*v2y)
	} else {
		return 1.0
	}
}

func approxArcLengthCubic(p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, p3 curve.Vec2) float64 {
	chordLen := (p3.Sub(p0)).Hypot()
	// Length of the control polygon
	polyLen := (p1.Sub(p0)).Hypot() + (p2.Sub(p1)).Hypot() + (p3.Sub(p2)).Hypot()
	return 0.5 * (chordLen + polyLen)
}

func countSegmentsForCubic(p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, p3 curve.Vec2, t jmath.Transform) float64 {
	p0 = transform(t, p0)
	p1 = transform(t, p1)
	p2 = transform(t, p2)
	p3 = transform(t, p3)
	return math.Ceil(approxArcLengthCubic(p0, p1, p2, p3) * 0.0625 * math.Sqrt2)
}

func countSegmentsForQuadratic(p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, t jmath.Transform) float64 {
	return countSegmentsForCubic(p0, p1.Lerp(p0, 0.333333), p1.Lerp(p2, 0.333333), p2, t)
}

// Estimate tile crossings for a line with known endpoints.
func countSegmentsForLine(p0 curve.Point, p1 curve.Point, t jmath.Transform) uint32 {
	dxdy := p0.Sub(p1)
	dxdy = transform(t, dxdy)
	segments := math.Ceil(math.Ceil(math.Abs(dxdy.X))*0.0625) + math.Ceil(math.Ceil(math.Abs(dxdy.Y))*0.0625)
	return max(1, uint32(segments))
}

// Estimate tile crossings for a line with a known length.
func countSegmentsForLineLength(scaledWidth float64) uint32 {
	// scale the tile count by sqrt(2) to allow some slack for diagonal lines.
	// TODO: Would "2" be a better factor?
	return max(1, uint32(math.Ceil(scaledWidth*0.0625*math.Sqrt2)))
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
const sqrtOfDegreeTermCubic = 0.86602540378

// The curve degree term sqrt(n * (n - 1) / 8) specialized for quadratics:
//
//	sqrt(2 * (2 - 1) / 8)
const sqrtOfDegreeTermQuad = 0.5

func wangQuadratic(rsqrtOfTol float64, p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, t jmath.Transform) float64 {
	v := p1.Mul(-2).Add(p0).Add(p2)
	v = transform(t, v) // transform is distributive
	m := v.Hypot()
	return math.Ceil(sqrtOfDegreeTermQuad * math.Sqrt(m) * rsqrtOfTol)
}

func wangCubic(rsqrtOfTol float64, p0 curve.Vec2, p1 curve.Vec2, p2 curve.Vec2, p3 curve.Vec2, t jmath.Transform) float64 {
	v1 := p1.Mul(-2).Add(p0).Add(p2)
	v2 := p2.Mul(-2).Add(p1).Add(p3)
	v1 = transform(t, v1)
	v2 = transform(t, v2)
	m := max(v1.Hypot(), v2.Hypot())
	return math.Ceil(sqrtOfDegreeTermCubic * math.Sqrt(m) * rsqrtOfTol)
}

type option[T any] struct {
	isSet bool
	value T
}

func (opt *option[T]) set(v T) {
	opt.isSet = true
	opt.value = v
}

func (opt *option[T]) clear() {
	opt.isSet = false
	opt.value = *new(T)
}

func (opt option[T]) unwrap() T {
	if !opt.isSet {
		panic("option isn't set")
	}
	return opt.value
}
