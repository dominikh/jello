// Copyright 2023 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

package cpu

import (
	"math"
	"math/bits"
	"unsafe"

	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/jello/mem"
	"honnef.co/go/jello/renderer"
	"honnef.co/go/safeish"
)

// Note to readers: this file contains sophisticated techniques for expanding stroke
// outlines to flattened filled outlines, based on Euler spirals as an intermediate
// curve representation.
// A paper is in the works explaining the techniques in more detail.

// / Threshold below which a derivative is considered too small.
const derivThresh = 1e-6

// / Amount to nudge t when derivative is near-zero.
const derivEps = 1e-6

// Limit for subdivision of cubic Béziers.
const subdivLimit = 1.0 / 65536.0

// Evaluate both the point and derivative of a cubic bezier.
func evalCubicAndDeriv(
	p0 vec2,
	p1 vec2,
	p2 vec2,
	p3 vec2,
	t float32,
) (vec2, vec2) {
	m := 1.0 - t
	mm := m * m
	mt := m * t
	tt := t * t
	p := (p0.mul(mm * m)).add(((p1.mul(3.0 * mm)).add(p2.mul(3.0 * mt)).add(p3.mul(tt))).mul(t))
	q := ((p1.sub(p0)).mul(mm)).add((p2.sub(p1)).mul(2.0 * mt)).add((p3.sub(p2)).mul(tt))
	return p, q
}

func cubicStartTangent(p0, p1, p2, p3 vec2) vec2 {
	d01 := p1.sub(p0)
	d02 := p2.sub(p0)
	d03 := p3.sub(p0)
	if d01.lengthSquared() > robustEpsilon {
		return d01
	} else if d02.lengthSquared() > robustEpsilon {
		return d02
	} else {
		return d03
	}
}

func cubicEndTangent(p0, p1, p2, p3 vec2) vec2 {
	d23 := p3.sub(p2)
	d13 := p3.sub(p1)
	d03 := p3.sub(p0)
	if d23.lengthSquared() > robustEpsilon {
		return d23
	} else if d13.lengthSquared() > robustEpsilon {
		return d13
	} else {
		return d03
	}
}

func readFloat32Point(idx uint32, pathdata []uint32) vec2 {
	x := math.Float32frombits(pathdata[idx])
	y := math.Float32frombits(pathdata[idx+1])
	return vec2{x, y}
}

func readInt16Point(idx uint32, pathdata []uint32) vec2 {
	raw := pathdata[idx]
	x := float32((int32(raw << 16)) >> 16)
	y := float32(int32(raw) >> 16)
	return vec2{x, y}
}

type intBbox struct {
	x0 int32
	y0 int32
	x1 int32
	y1 int32
}

func newIntBbox() intBbox {
	return intBbox{
		x0: 0x7fff_ffff,
		y0: 0x7fff_ffff,
		x1: -0x8000_0000,
		y1: -0x8000_0000,
	}
}

func (bbox *intBbox) addPoint(pt vec2) {
	bbox.x0 = min(bbox.x0, int32(jmath.Floor32(pt.x)))
	bbox.y0 = min(bbox.y0, int32(jmath.Floor32(pt.y)))
	bbox.x1 = max(bbox.x1, int32(jmath.Ceil32(pt.x)))
	bbox.y1 = max(bbox.y1, int32(jmath.Ceil32(pt.y)))
}

type pathTagData struct {
	tagByte uint8
	monoid  renderer.PathMonoid
}

func computeTagMonoid(idx int, pathtags []uint32, tagMonoids []renderer.PathMonoid) pathTagData {
	tagWord := pathtags[idx>>2]
	shift := (idx & 3) * 8
	tm := renderer.NewPathMonoid(tagWord & ((1 << shift) - 1))
	tagByte := uint8((tagWord >> shift) & 0xff)
	if tagByte != 0 {
		tm = tagMonoids[idx>>2].Combine(tm)
	}
	// We no longer encode an initial transform and style so these
	// are off by one.
	// We wrap here because these values will return to positive values later
	// (when we add StyleBase)
	tm.TransIdx = tm.TransIdx - 1

	tm.StyleIdx = tm.StyleIdx - uint32(unsafe.Sizeof(encoding.Style{})/4)
	return pathTagData{
		tagByte: tagByte,
		monoid:  tm,
	}
}

type cubicPoints struct {
	p0 vec2
	p1 vec2
	p2 vec2
	p3 vec2
}

func readPathSegment(tag pathTagData, isStroke bool, pathdata []uint32) cubicPoints {
	var p0, p1, p2, p3 vec2

	segType := tag.tagByte & pathTagSegType
	pathSegOffset := tag.monoid.PathSegOffset
	isStrokeCapMarker := isStroke && (encoding.PathTag(tag.tagByte)&encoding.PathTagSubpathEndBit) != 0
	isOpen := segType == pathTagQuadTo

	if (tag.tagByte & pathTagF32) != 0 {
		p0 = readFloat32Point(pathSegOffset, pathdata)
		p1 = readFloat32Point(pathSegOffset+2, pathdata)
		if segType >= pathTagQuadTo {
			p2 = readFloat32Point(pathSegOffset+4, pathdata)
			if segType == pathTagCubicTo {
				p3 = readFloat32Point(pathSegOffset+6, pathdata)
			}
		}
	} else {
		p0 = readInt16Point(pathSegOffset, pathdata)
		p1 = readInt16Point(pathSegOffset+1, pathdata)
		if segType >= pathTagQuadTo {
			p2 = readInt16Point(pathSegOffset+2, pathdata)
			if segType == pathTagCubicTo {
				p3 = readInt16Point(pathSegOffset+3, pathdata)
			}
		}
	}

	if isStrokeCapMarker && isOpen {
		p0 = p1
		p1 = p2
		segType = pathTagLineTo
	}

	// Degree-raise
	switch segType {
	case pathTagLineTo:
		p3 = p1
		p2 = p3.mix(p0, 1.0/3.0)
		p1 = p0.mix(p3, 1.0/3.0)
	case pathTagQuadTo:
		p3 = p2
		p2 = p1.mix(p2, 1.0/3.0)
		p1 = p1.mix(p0, 1.0/3.0)
	}

	return cubicPoints{p0, p1, p2, p3}
}

type neighboringSegment struct {
	doJoin  bool
	tangent vec2
}

func readNeighboringSegment(
	idx int,
	pathTags []uint32,
	pathData []uint32,
	tagMonoids []renderer.PathMonoid,
) neighboringSegment {
	tag := computeTagMonoid(idx, pathTags, tagMonoids)
	pts := readPathSegment(tag, true, pathData)

	isClosed := (tag.tagByte & pathTagSegType) == pathTagLineTo
	isStrokeCapMarker := (encoding.PathTag(tag.tagByte) & encoding.PathTagSubpathEndBit) != 0
	doJoin := !isStrokeCapMarker || isClosed
	tangent := cubicStartTangent(pts.p0, pts.p1, pts.p2, pts.p3)
	return neighboringSegment{doJoin, tangent}
}

// A robustness strategy for the ESPC integral
type espcRobust int

const (
	// Both k1 and dist are large enough to divide by robustly.
	espcRobustNormal = iota
	// k1 is low, so model curve as a circular arc.
	espcRobustLowK1
	// dist is low, so model curve as just an Euler spiral.
	espcRobustLowDist
)

func writeLine(
	lineIdx int,
	pathIdx uint32,
	p0 vec2,
	p1 vec2,
	bbox *intBbox,
	lines []renderer.LineSoup,
) {
	assert(!p0.nan() && !p1.nan())
	bbox.addPoint(p0)
	bbox.addPoint(p1)
	lines[lineIdx] = renderer.LineSoup{
		PathIdx: pathIdx,
		P0:      [2]float32{p0.x, p0.y},
		P1:      [2]float32{p1.x, p1.y},
	}
}

func writeLineWithTransform(
	lineIdx int,
	pathIdx uint32,
	p0 vec2,
	p1 vec2,
	transform transform,
	bbox *intBbox,
	lines []renderer.LineSoup,
) {
	writeLine(
		lineIdx,
		pathIdx,
		transform.apply(p0),
		transform.apply(p1),
		bbox,
		lines,
	)
}

func outputLine(
	pathIdx uint32,
	p0 vec2,
	p1 vec2,
	lineIdx *int,
	bbox *intBbox,
	lines []renderer.LineSoup,
) {
	writeLine(*lineIdx, pathIdx, p0, p1, bbox, lines)
	*lineIdx += 1
}

func outputLineWithTransform(
	pathIdx uint32,
	p0 vec2,
	p1 vec2,
	transform transform,
	lineIdx *int,
	lines []renderer.LineSoup,
	bbox *intBbox,
) {
	writeLineWithTransform(*lineIdx, pathIdx, p0, p1, transform, bbox, lines)
	*lineIdx += 1
}

func outputTwoLinesWithTransform(
	pathIdx uint32,
	p00 vec2,
	p01 vec2,
	p10 vec2,
	p11 vec2,
	transform transform,
	lineIdx *int,
	lines []renderer.LineSoup,
	bbox *intBbox,
) {
	writeLineWithTransform(*lineIdx, pathIdx, p00, p01, transform, bbox, lines)
	writeLineWithTransform(*lineIdx+1, pathIdx, p10, p11, transform, bbox, lines)
	*lineIdx += 2
}

func flattenArc(
	pathIdx uint32,
	begin vec2,
	end vec2,
	center vec2,
	angle float32,
	trans transform,
	lineIdx *int,
	lines []renderer.LineSoup,
	bbox *intBbox,
) {
	const minTheta = 0.0001

	p0 := trans.apply(begin)
	r := begin.sub(center)
	const tol = 0.25
	radius := max(tol, ((p0.sub(trans.apply(center))).length()))
	theta := max((2.0 * jmath.Acos32(1.0-tol/radius)), minTheta)

	// Always output at least one line so that we always draw the chord.
	numLines := max(uint32(jmath.Ceil32(angle/theta)), 1)

	s, c := jmath.Sincos32(theta)
	rot := transform{c, -s, s, c, 0, 0}

	for range numLines - 1 {
		r = rot.apply(r)
		p1 := trans.apply(center.add(r))
		outputLine(pathIdx, p0, p1, lineIdx, bbox, lines)
		p0 = p1
	}
	p1 := trans.apply(end)
	outputLine(pathIdx, p0, p1, lineIdx, bbox, lines)
}

func flattenEuler(
	cubic cubicPoints,
	pathIdx uint32,
	localToDevice transform,
	offset float32,
	startP vec2,
	endP vec2,
	lineIdx *int,
	lines []renderer.LineSoup,
	bbox *intBbox,
) {
	// Flatten in local coordinates if this is a stroke. Flatten in device space otherwise.
	var (
		p0, p1, p2, p3 vec2
		scale          float32
		transform      transform
	)
	if offset == 0 {
		p0 = localToDevice.apply(cubic.p0)
		p1 = localToDevice.apply(cubic.p1)
		p2 = localToDevice.apply(cubic.p2)
		p3 = localToDevice.apply(cubic.p3)
		scale = 1
		transform = identity
	} else {
		t := localToDevice
		scale = 0.5*vec2{t[0] + t[3], t[1] - t[2]}.length() + vec2{t[0] - t[3], t[1] + t[2]}.length()
		p0 = cubic.p0
		p1 = cubic.p1
		p2 = cubic.p2
		p3 = cubic.p3
		transform = localToDevice
	}
	var tStart, tEnd vec2
	if offset == 0.0 {
		tStart, tEnd = p0, p3
	} else {
		tStart, tEnd = startP, endP
	}

	// Drop zero length lines. This is an exact equality test because dropping very short
	// line segments may result in loss of watertightness. The parallel curves of zero
	// length lines add nothing to stroke outlines, but we still may need to draw caps.
	if p0 == p1 && p0 == p2 && p0 == p3 {
		return
	}

	const tol = 0.25
	var t0u uint32
	dt := float32(1)
	lastP := p0
	lastQ := p1.sub(p0)
	// We want to avoid near zero derivatives, so the general technique is to
	// detect, then sample a nearby t value if it fails to meet the threshold.
	if lastQ.lengthSquared() < derivThresh*derivThresh {
		_, lastQ = evalCubicAndDeriv(p0, p1, p2, p3, derivEps)
	}
	var lastT float32
	lp0 := tStart

	for {
		t0 := (float32(t0u)) * dt
		if t0 == 1. {
			break
		}
		t1 := t0 + dt
		thisP0 := lastP
		thisQ0 := lastQ
		thisP1, thisQ1 := evalCubicAndDeriv(p0, p1, p2, p3, t1)
		if thisQ1.lengthSquared() < derivThresh*derivThresh {
			newP1, newQ1 := evalCubicAndDeriv(p0, p1, p2, p3, t1-derivEps)
			thisQ1 = newQ1
			// Change just the derivative at the endpoint, but also move the point so it
			// matches the derivative exactly if in the interior.
			if t1 < 1. {
				thisP1 = newP1
				t1 -= derivEps
			}
		}
		actualDt := t1 - lastT
		cubicParams := cubicParamsFromPointsDerivs(thisP0, thisP1, thisQ0, thisQ1, actualDt)
		if cubicParams.err*scale <= tol || dt <= subdivLimit {
			eulerParams := eulerParamsFromAngles(cubicParams.th0, cubicParams.th1)
			es := eulerSegFromParams(thisP0, thisP1, eulerParams)

			k0, k1 := es.params.k0-0.5*es.params.k1, es.params.k1

			// compute forward integral to determine number of subdivisions
			normalizedOffset := offset / cubicParams.chordLen
			distScaled := normalizedOffset * es.params.ch
			// The number of subdivisions for curvature = 1
			scaleMultiplier := 0.5 * 1.0 / (math.Sqrt2) * jmath.Sqrt32(scale*cubicParams.chordLen/(es.params.ch*tol))
			// TODO: tune these thresholds
			const k1Thresh = 1e-3
			const distThresh = 1e-3
			var a, b, integral, int0 float32
			var nFrac float32
			var robust espcRobust
			if jmath.Abs32(k1) < k1Thresh {
				k := k0 + 0.5*k1
				nFrac = jmath.Sqrt32(jmath.Abs32((k * (k*distScaled + 1.0))))
				robust = espcRobustLowK1
			} else if jmath.Abs32(distScaled) < distThresh {
				f := func(x float32) float32 { return x * jmath.Sqrt32(jmath.Abs32(x)) }
				a = k1
				b = k0
				int0 = f(b)
				int1 := f(a + b)
				integral = int1 - int0
				nFrac = (2. / 3.) * integral / a
				robust = espcRobustLowDist
			} else {
				a = -2.0 * distScaled * k1
				b = -1.0 - 2.0*distScaled*k0
				int0 = espcIntApprox(b)
				int1 := espcIntApprox(a + b)
				integral = int1 - int0
				kPeak := k0 - k1*b/a
				integrandPeak := jmath.Sqrt32(jmath.Abs32((kPeak * (kPeak*distScaled + 1.0))))
				scaledInt := integral * integrandPeak / a
				nFrac = scaledInt
				robust = espcRobustNormal
			}
			n := jmath.Clamp(jmath.Ceil32(nFrac*scaleMultiplier), 1.0, 100.0)

			// Flatten line segments
			assert(!math.IsNaN(float64(n)))
			for i := range int(n) {
				var lp1 vec2
				if i == int(n)-1 && t1 == 1.0 {
					lp1 = tEnd
				} else {
					t := float32(i+1) / n
					var s float32
					switch robust {
					case espcRobustLowK1:
						s = t
					// Note opportunities to minimize divergence
					case espcRobustLowDist:
						c := jmath.Cbrt32(integral*t + int0)
						inv := c * jmath.Abs32(c)
						s = (inv - b) / a
					case espcRobustNormal:
						inv := espcIntInvApprox(integral*t + int0)
						s = (inv - b) / a
					}
					lp1 = es.evalWithOffset(s, normalizedOffset)
				}
				var l0, l1 vec2
				if offset >= 0. {
					l0 = lp0
				} else {
					l0 = lp1
				}
				if offset >= 0. {
					l1 = lp1
				} else {
					l1 = lp0
				}
				outputLineWithTransform(pathIdx, l0, l1, transform, lineIdx, lines, bbox)
				lp0 = lp1
			}
			lastP = thisP1
			lastQ = thisQ1
			lastT = t1
			// Advance segment to next range. Beginning of segment is the end of
			// this one. The number of trailing zeros represents the number of stack
			// frames to pop in the recursive version of adaptive subdivision, and
			// each stack pop represents doubling of the size of the range.
			t0u += 1
			shift := bits.TrailingZeros32(t0u)
			t0u >>= shift
			dt *= float32(int(1) << shift)
		} else {
			// Subdivide; halve the size of the range while retaining its start.
			t0u2 := uint64(t0u) * 2
			if t0u2 > math.MaxUint32 {
				t0u2 = math.MaxUint32
			}
			t0u = uint32(t0u2)
			dt *= 0.5
		}
	}
}

func drawCap(
	pathIdx uint32,
	capStyle uint32,
	point vec2,
	cap0 vec2,
	cap1 vec2,
	offsetTangent vec2,
	transform transform,
	lineIdx *int,
	lines []renderer.LineSoup,
	bbox *intBbox,
) {
	if capStyle == encoding.FlagsCapBitsRound {
		flattenArc(
			pathIdx,
			cap0,
			cap1,
			point,
			math.Pi,
			transform,
			lineIdx,
			lines,
			bbox,
		)
		return
	}

	start := cap0
	end := cap1
	if capStyle == encoding.FlagsCapBitsSquare {
		v := offsetTangent
		p0 := start.add(v)
		p1 := end.add(v)
		outputLineWithTransform(pathIdx, start, p0, transform, lineIdx, lines, bbox)
		outputLineWithTransform(pathIdx, p1, end, transform, lineIdx, lines, bbox)
		start = p0
		end = p1
	}
	outputLineWithTransform(pathIdx, start, end, transform, lineIdx, lines, bbox)
}

func drawJoin(
	pathIdx uint32,
	styleFlags uint32,
	p0 vec2,
	tanPrev vec2,
	tanNext vec2,
	nPrev vec2,
	nNext vec2,
	transform transform,
	lineIdx *int,
	lines []renderer.LineSoup,
	bbox *intBbox,
) {
	front0 := p0.add(nPrev)
	front1 := p0.add(nNext)
	back0 := p0.sub(nNext)
	back1 := p0.sub(nPrev)

	cr := tanPrev.x*tanNext.y - tanPrev.y*tanNext.x
	d := tanPrev.dot(tanNext)

	switch styleFlags & encoding.FlagsJoinMask {
	case encoding.FlagsJoinBitsBevel:
		if front0 != front1 && back0 != back1 {
			outputTwoLinesWithTransform(
				pathIdx, front0, front1, back0, back1, transform, lineIdx, lines, bbox,
			)
		}
	case encoding.FlagsJoinBitsMiter:
		hypot := jmath.Hypot32(cr, d)
		miterLimit := jmath.Float16frombits(uint16(styleFlags & encoding.MiterLimitMask))

		if 2.*hypot < (hypot+d)*miterLimit*miterLimit && cr != 0. {
			isBackside := cr > 0.
			var fpLast, fpThis, p vec2
			if isBackside {
				fpLast = back1
			} else {
				fpLast = front0
			}
			if isBackside {
				fpThis = back0
			} else {
				fpThis = front1
			}
			if isBackside {
				p = back0
			} else {
				p = front0
			}

			v := fpThis.sub(fpLast)
			h := (tanPrev.x*v.y - tanPrev.y*v.x) / cr
			miterPt := fpThis.sub(tanNext.mul(h))

			outputLineWithTransform(pathIdx, p, miterPt, transform, lineIdx, lines, bbox)

			if isBackside {
				back0 = miterPt
			} else {
				front0 = miterPt
			}
		}
		outputTwoLinesWithTransform(
			pathIdx, front0, front1, back0, back1, transform, lineIdx, lines, bbox,
		)
	case encoding.FlagsJoinBitsRound:
		var arc0, arc1, other0, other1 vec2
		if cr > 0. {
			arc0, arc1, other0, other1 = back0, back1, front0, front1
		} else {
			arc0, arc1, other0, other1 = front0, front1, back0, back1
		}
		flattenArc(
			pathIdx,
			arc0,
			arc1,
			p0,
			jmath.Abs32(jmath.Atan232(cr, d)),
			transform,
			lineIdx,
			lines,
			bbox,
		)
		outputLineWithTransform(pathIdx, other0, other1, transform, lineIdx, lines, bbox)
	default:
		panic("unreachable")
	}
}

func Flatten(_ *mem.Arena, numWgs uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	tagMonoids := safeish.SliceCast[[]renderer.PathMonoid](resources[2].(CPUBuffer))
	pathBboxes := safeish.SliceCast[[]renderer.PathBbox](resources[3].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[4].(CPUBuffer))
	lines := safeish.SliceCast[[]renderer.LineSoup](resources[5].(CPUBuffer))

	lineIdx := 0
	pathtags := scene[config.Layout.PathTagBase:]
	pathdata := scene[config.Layout.PathDataBase:]

	for idx := range int(numWgs) * wgSize {
		bbox := newIntBbox()
		tag := computeTagMonoid(idx, pathtags, tagMonoids)
		pathIdx := tag.monoid.PathIdx
		styleIdx := tag.monoid.StyleIdx
		transIdx := tag.monoid.TransIdx
		styleFlags := scene[(config.Layout.StyleBase + styleIdx)]
		if (tag.tagByte & pathTagPath) != 0 {
			out := &pathBboxes[pathIdx]
			if (styleFlags & encoding.FlagsFillBit) == 0 {
				out.DrawFlags = 0
			} else {
				out.DrawFlags = drawInfoFlagsFillRuleBit
			}
			out.TransIdx = transIdx
		}

		segType := tag.tagByte & pathTagSegType
		if segType != 0 {
			isStroke := (styleFlags & encoding.FlagsStyleBit) != 0
			transform := transformRead(config.Layout.TransformBase, transIdx, scene)
			pts := readPathSegment(tag, isStroke, pathdata)

			if isStroke {
				linewidth := math.Float32frombits(scene[(config.Layout.StyleBase + styleIdx + 1)])
				offset := 0.5 * linewidth

				isOpen := segType != pathTagLineTo
				isStrokeCapMarker := (encoding.PathTag(tag.tagByte) & encoding.PathTagSubpathEndBit) != 0
				if isStrokeCapMarker {
					if isOpen {
						// Draw start cap
						tangent := cubicStartTangent(pts.p0, pts.p1, pts.p2, pts.p3)
						offsetTangent := tangent.normalize().mul(offset)
						n := vec2{-offsetTangent.y, offsetTangent.x}
						drawCap(
							pathIdx,
							(styleFlags&encoding.FlagsStartCapMask)>>2,
							pts.p0,
							pts.p0.sub(n),
							pts.p0.add(n),
							offsetTangent.mul(-1),
							transform,
							&lineIdx,
							lines,
							&bbox,
						)
					} else {
						// Don't draw anything if the path is closed.
					}
				} else {
					// Read the neighboring segment.
					neighbor := readNeighboringSegment(idx+1, pathtags, pathdata, tagMonoids)
					tanPrev := cubicEndTangent(pts.p0, pts.p1, pts.p2, pts.p3)
					tanNext := neighbor.tangent
					tanStart := cubicStartTangent(pts.p0, pts.p1, pts.p2, pts.p3)
					// TODO: be consistent w/ robustness here

					// TODO: add NaN assertions to CPU shaders PR (when writing lines)
					// TODO: not all zero-length segments are getting filtered out
					// TODO: this is a hack. How to handle caps on degenerate stroke?
					// TODO: debug tricky stroke by isolation
					if tanStart.lengthSquared() < (tangentThresh * tangentThresh) {
						tanStart = vec2{tangentThresh, 0.}
					}
					if tanPrev.lengthSquared() < (tangentThresh * tangentThresh) {
						tanPrev = vec2{tangentThresh, 0.}
					}
					if tanNext.lengthSquared() < (tangentThresh * tangentThresh) {
						tanNext = vec2{tangentThresh, 0.}
					}

					nStart := vec2{-tanStart.y, tanStart.x}.normalize().mul(offset)
					offsetTangent := tanPrev.normalize().mul(offset)
					nPrev := vec2{-offsetTangent.y, offsetTangent.x}
					tanNextNorm := tanNext.normalize()
					nNext := vec2{-tanNextNorm.y, tanNextNorm.x}.mul(offset)

					// Render offset curves
					flattenEuler(
						pts,
						pathIdx,
						transform,
						offset,
						pts.p0.add(nStart),
						pts.p3.add(nPrev),
						&lineIdx,
						lines,
						&bbox,
					)
					flattenEuler(
						pts,
						pathIdx,
						transform,
						-offset,
						pts.p0.sub(nStart),
						pts.p3.sub(nPrev),
						&lineIdx,
						lines,
						&bbox,
					)

					if neighbor.doJoin {
						drawJoin(
							pathIdx,
							styleFlags,
							pts.p3,
							tanPrev,
							tanNext,
							nPrev,
							nNext,
							transform,
							&lineIdx,
							lines,
							&bbox,
						)
					} else {
						// Draw end cap.
						drawCap(
							pathIdx,
							styleFlags&encoding.FlagsEndCapMask,
							pts.p3,
							pts.p3.add(nPrev),
							pts.p3.sub(nPrev),
							offsetTangent,
							transform,
							&lineIdx,
							lines,
							&bbox,
						)
					}
				}
			} else {
				flattenEuler(
					pts,
					pathIdx,
					transform,
					/*offset*/ 0.,
					pts.p0,
					pts.p3,
					&lineIdx,
					lines,
					&bbox,
				)
			}
		}

		if int(pathIdx) < len(pathBboxes) && (bbox.x1 > bbox.x0 || bbox.y1 > bbox.y0) {
			out := &pathBboxes[pathIdx]
			out.X0 = min(out.X0, bbox.x0)
			out.Y0 = min(out.Y0, bbox.y0)
			out.X1 = max(out.X1, bbox.x1)
			out.Y1 = max(out.Y1, bbox.y1)
		}
	}
	bump.Lines = uint32(lineIdx)
}
