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
const DERIV_THRESH = 1e-6

// / Amount to nudge t when derivative is near-zero.
const DERIV_EPS = 1e-6

// Limit for subdivision of cubic Béziers.
const SUBDIV_LIMIT = 1.0 / 65536.0

// Evaluate both the point and derivative of a cubic bezier.
func eval_cubic_and_deriv(
	p0 Vec2,
	p1 Vec2,
	p2 Vec2,
	p3 Vec2,
	t float32,
) (Vec2, Vec2) {
	m := 1.0 - t
	mm := m * m
	mt := m * t
	tt := t * t
	p := (p0.Mul(mm * m)).Add(((p1.Mul(3.0 * mm)).Add(p2.Mul(3.0 * mt)).Add(p3.Mul(tt))).Mul(t))
	q := ((p1.Sub(p0)).Mul(mm)).Add((p2.Sub(p1)).Mul(2.0 * mt)).Add((p3.Sub(p2)).Mul(tt))
	return p, q
}

func cubic_start_tangent(p0, p1, p2, p3 Vec2) Vec2 {
	d01 := p1.Sub(p0)
	d02 := p2.Sub(p0)
	d03 := p3.Sub(p0)
	if d01.lengthSquared() > ROBUST_EPSILON {
		return d01
	} else if d02.lengthSquared() > ROBUST_EPSILON {
		return d02
	} else {
		return d03
	}
}

func cubic_end_tangent(p0, p1, p2, p3 Vec2) Vec2 {
	d23 := p3.Sub(p2)
	d13 := p3.Sub(p1)
	d03 := p3.Sub(p0)
	if d23.lengthSquared() > ROBUST_EPSILON {
		return d23
	} else if d13.lengthSquared() > ROBUST_EPSILON {
		return d13
	} else {
		return d03
	}
}

func read_f32_point(ix uint32, pathdata []uint32) Vec2 {
	x := math.Float32frombits(pathdata[ix])
	y := math.Float32frombits(pathdata[ix+1])
	return Vec2{x, y}
}

func read_i16_point(ix uint32, pathdata []uint32) Vec2 {
	raw := pathdata[ix]
	x := float32((int32(raw << 16)) >> 16)
	y := float32(int32(raw) >> 16)
	return Vec2{x, y}
}

type IntBbox struct {
	x0 int32
	y0 int32
	x1 int32
	y1 int32
}

func newIntBbox() IntBbox {
	return IntBbox{
		x0: 0x7fff_ffff,
		y0: 0x7fff_ffff,
		x1: -0x8000_0000,
		y1: -0x8000_0000,
	}
}

func (self *IntBbox) add_pt(pt Vec2) {
	self.x0 = min(self.x0, int32(jmath.Floor32(pt.x)))
	self.y0 = min(self.y0, int32(jmath.Floor32(pt.y)))
	self.x1 = max(self.x1, int32(jmath.Ceil32(pt.x)))
	self.y1 = max(self.y1, int32(jmath.Ceil32(pt.y)))
}

type PathTagData struct {
	tag_byte uint8
	monoid   renderer.PathMonoid
}

func compute_tag_monoid(ix int, pathtags []uint32, tag_monoids []renderer.PathMonoid) PathTagData {
	tag_word := pathtags[ix>>2]
	shift := (ix & 3) * 8
	tm := renderer.NewPathMonoid(tag_word & ((1 << shift) - 1))
	tag_byte := uint8((tag_word >> shift) & 0xff)
	if tag_byte != 0 {
		tm = tag_monoids[ix>>2].Combine(tm)
	}
	// We no longer encode an initial transform and style so these
	// are off by one.
	// We wrap here because these values will return to positive values later
	// (when we add style_base)
	tm.TransIdx = tm.TransIdx - 1

	tm.StyleIdx = tm.StyleIdx - uint32(unsafe.Sizeof(encoding.Style{})/4)
	return PathTagData{
		tag_byte: tag_byte,
		monoid:   tm,
	}
}

type CubicPoints struct {
	p0 Vec2
	p1 Vec2
	p2 Vec2
	p3 Vec2
}

func read_path_segment(tag PathTagData, is_stroke bool, pathdata []uint32) CubicPoints {
	var p0, p1, p2, p3 Vec2

	seg_type := tag.tag_byte & PATH_TAG_SEG_TYPE
	pathseg_offset := tag.monoid.PathSegOffset
	is_stroke_cap_marker := is_stroke && (encoding.PathTag(tag.tag_byte)&encoding.PathTagSubpathEndBit) != 0
	is_open := seg_type == PATH_TAG_QUADTO

	if (tag.tag_byte & PATH_TAG_F32) != 0 {
		p0 = read_f32_point(pathseg_offset, pathdata)
		p1 = read_f32_point(pathseg_offset+2, pathdata)
		if seg_type >= PATH_TAG_QUADTO {
			p2 = read_f32_point(pathseg_offset+4, pathdata)
			if seg_type == PATH_TAG_CUBICTO {
				p3 = read_f32_point(pathseg_offset+6, pathdata)
			}
		}
	} else {
		p0 = read_i16_point(pathseg_offset, pathdata)
		p1 = read_i16_point(pathseg_offset+1, pathdata)
		if seg_type >= PATH_TAG_QUADTO {
			p2 = read_i16_point(pathseg_offset+2, pathdata)
			if seg_type == PATH_TAG_CUBICTO {
				p3 = read_i16_point(pathseg_offset+3, pathdata)
			}
		}
	}

	if is_stroke_cap_marker && is_open {
		p0 = p1
		p1 = p2
		seg_type = PATH_TAG_LINETO
	}

	// Degree-raise
	switch seg_type {
	case PATH_TAG_LINETO:
		p3 = p1
		p2 = p3.mix(p0, 1.0/3.0)
		p1 = p0.mix(p3, 1.0/3.0)
	case PATH_TAG_QUADTO:
		p3 = p2
		p2 = p1.mix(p2, 1.0/3.0)
		p1 = p1.mix(p0, 1.0/3.0)
	}

	return CubicPoints{p0, p1, p2, p3}
}

type NeighboringSegment struct {
	do_join bool
	tangent Vec2
}

func read_neighboring_segment(
	ix int,
	pathtags []uint32,
	pathdata []uint32,
	tag_monoids []renderer.PathMonoid,
) NeighboringSegment {
	tag := compute_tag_monoid(ix, pathtags, tag_monoids)
	pts := read_path_segment(tag, true, pathdata)

	is_closed := (tag.tag_byte & PATH_TAG_SEG_TYPE) == PATH_TAG_LINETO
	is_stroke_cap_marker := (encoding.PathTag(tag.tag_byte) & encoding.PathTagSubpathEndBit) != 0
	do_join := !is_stroke_cap_marker || is_closed
	tangent := cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3)
	return NeighboringSegment{do_join, tangent}
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

func write_line(
	line_ix int,
	path_ix uint32,
	p0 Vec2,
	p1 Vec2,
	bbox *IntBbox,
	lines []renderer.LineSoup,
) {
	assert(!p0.NaN() && !p1.NaN())
	bbox.add_pt(p0)
	bbox.add_pt(p1)
	lines[line_ix] = renderer.LineSoup{
		PathIdx: path_ix,
		P0:      p0.to_array(),
		P1:      p1.to_array(),
	}
}

func write_line_with_transform(
	line_ix int,
	path_ix uint32,
	p0 Vec2,
	p1 Vec2,
	transform Transform,
	bbox *IntBbox,
	lines []renderer.LineSoup,
) {
	write_line(
		line_ix,
		path_ix,
		transform.apply(p0),
		transform.apply(p1),
		bbox,
		lines,
	)
}

func output_line(
	path_ix uint32,
	p0 Vec2,
	p1 Vec2,
	line_ix *int,
	bbox *IntBbox,
	lines []renderer.LineSoup,
) {
	write_line(*line_ix, path_ix, p0, p1, bbox, lines)
	*line_ix += 1
}

func output_line_with_transform(
	path_ix uint32,
	p0 Vec2,
	p1 Vec2,
	transform Transform,
	line_ix *int,
	lines []renderer.LineSoup,
	bbox *IntBbox,
) {
	write_line_with_transform(*line_ix, path_ix, p0, p1, transform, bbox, lines)
	*line_ix += 1
}

func output_two_lines_with_transform(
	path_ix uint32,
	p00 Vec2,
	p01 Vec2,
	p10 Vec2,
	p11 Vec2,
	transform Transform,
	line_ix *int,
	lines []renderer.LineSoup,
	bbox *IntBbox,
) {
	write_line_with_transform(*line_ix, path_ix, p00, p01, transform, bbox, lines)
	write_line_with_transform(*line_ix+1, path_ix, p10, p11, transform, bbox, lines)
	*line_ix += 2
}

func flatten_arc(
	path_ix uint32,
	begin Vec2,
	end Vec2,
	center Vec2,
	angle float32,
	transform Transform,
	line_ix *int,
	lines []renderer.LineSoup,
	bbox *IntBbox,
) {
	const MIN_THETA = 0.0001

	p0 := transform.apply(begin)
	r := begin.Sub(center)
	const tol = 0.25
	radius := max(tol, ((p0.Sub(transform.apply(center))).length()))
	theta := max((2.0 * jmath.Acos32(1.0-tol/radius)), MIN_THETA)

	// Always output at least one line so that we always draw the chord.
	n_lines := max(uint32(jmath.Ceil32(angle/theta)), 1)

	s, c := jmath.Sincos32(theta)
	rot := Transform{c, -s, s, c, 0, 0}

	for range n_lines - 1 {
		r = rot.apply(r)
		p1 := transform.apply(center.Add(r))
		output_line(path_ix, p0, p1, line_ix, bbox, lines)
		p0 = p1
	}
	p1 := transform.apply(end)
	output_line(path_ix, p0, p1, line_ix, bbox, lines)
}

func flatten_euler(
	cubic CubicPoints,
	path_ix uint32,
	local_to_device Transform,
	offset float32,
	start_p Vec2,
	end_p Vec2,
	line_ix *int,
	lines []renderer.LineSoup,
	bbox *IntBbox,
) {
	// Flatten in local coordinates if this is a stroke. Flatten in device space otherwise.
	var (
		p0, p1, p2, p3 Vec2
		scale          float32
		transform      Transform
	)
	if offset == 0 {
		p0 = local_to_device.apply(cubic.p0)
		p1 = local_to_device.apply(cubic.p1)
		p2 = local_to_device.apply(cubic.p2)
		p3 = local_to_device.apply(cubic.p3)
		scale = 1
		transform = identity
	} else {
		t := local_to_device
		scale = 0.5*Vec2{t[0] + t[3], t[1] - t[2]}.length() + Vec2{t[0] - t[3], t[1] + t[2]}.length()
		p0 = cubic.p0
		p1 = cubic.p1
		p2 = cubic.p2
		p3 = cubic.p3
		transform = local_to_device
	}
	var t_start, t_end Vec2
	if offset == 0.0 {
		t_start, t_end = p0, p3
	} else {
		t_start, t_end = start_p, end_p
	}

	// Drop zero length lines. This is an exact equality test because dropping very short
	// line segments may result in loss of watertightness. The parallel curves of zero
	// length lines add nothing to stroke outlines, but we still may need to draw caps.
	if p0 == p1 && p0 == p2 && p0 == p3 {
		return
	}

	const tol = 0.25
	var t0_u uint32
	dt := float32(1)
	last_p := p0
	last_q := p1.Sub(p0)
	// We want to avoid near zero derivatives, so the general technique is to
	// detect, then sample a nearby t value if it fails to meet the threshold.
	if last_q.lengthSquared() < DERIV_THRESH*DERIV_THRESH {
		_, last_q = eval_cubic_and_deriv(p0, p1, p2, p3, DERIV_EPS)
	}
	var last_t float32
	lp0 := t_start

	for {
		t0 := (float32(t0_u)) * dt
		if t0 == 1. {
			break
		}
		t1 := t0 + dt
		this_p0 := last_p
		this_q0 := last_q
		this_p1, this_q1 := eval_cubic_and_deriv(p0, p1, p2, p3, t1)
		if this_q1.lengthSquared() < DERIV_THRESH*DERIV_THRESH {
			new_p1, new_q1 := eval_cubic_and_deriv(p0, p1, p2, p3, t1-DERIV_EPS)
			this_q1 = new_q1
			// Change just the derivative at the endpoint, but also move the point so it
			// matches the derivative exactly if in the interior.
			if t1 < 1. {
				this_p1 = new_p1
				t1 -= DERIV_EPS
			}
		}
		actual_dt := t1 - last_t
		cubic_params := cubicParamsFromPointsDerivs(this_p0, this_p1, this_q0, this_q1, actual_dt)
		if cubic_params.err*scale <= tol || dt <= SUBDIV_LIMIT {
			euler_params := eulerParamsFromAngles(cubic_params.th0, cubic_params.th1)
			es := eulerSegFromParams(this_p0, this_p1, euler_params)

			k0, k1 := es.params.k0-0.5*es.params.k1, es.params.k1

			// compute forward integral to determine number of subdivisions
			normalized_offset := offset / cubic_params.chord_len
			dist_scaled := normalized_offset * es.params.ch
			// The number of subdivisions for curvature = 1
			scale_multiplier := 0.5 * 1.0 / (math.Sqrt2) * jmath.Sqrt32(scale*cubic_params.chord_len/(es.params.ch*tol))
			// TODO: tune these thresholds
			const K1_THRESH = 1e-3
			const DIST_THRESH = 1e-3
			var a, b, integral, int0 float32
			var n_frac float32
			var robust espcRobust
			if jmath.Abs32(k1) < K1_THRESH {
				k := k0 + 0.5*k1
				n_frac = jmath.Sqrt32(jmath.Abs32((k * (k*dist_scaled + 1.0))))
				robust = espcRobustLowK1
			} else if jmath.Abs32(dist_scaled) < DIST_THRESH {
				f := func(x float32) float32 { return x * jmath.Sqrt32(jmath.Abs32(x)) }
				a = k1
				b = k0
				int0 = f(b)
				int1 := f(a + b)
				integral = int1 - int0
				n_frac = (2. / 3.) * integral / a
				robust = espcRobustLowDist
			} else {
				a = -2.0 * dist_scaled * k1
				b = -1.0 - 2.0*dist_scaled*k0
				int0 = espc_int_approx(b)
				int1 := espc_int_approx(a + b)
				integral = int1 - int0
				k_peak := k0 - k1*b/a
				integrand_peak := jmath.Sqrt32(jmath.Abs32((k_peak * (k_peak*dist_scaled + 1.0))))
				scaled_int := integral * integrand_peak / a
				n_frac = scaled_int
				robust = espcRobustNormal
			}
			n := jmath.Clamp(jmath.Ceil32(n_frac*scale_multiplier), 1.0, 100.0)

			// Flatten line segments
			assert(!math.IsNaN(float64(n)))
			for i := range int(n) {
				var lp1 Vec2
				if i == int(n)-1 && t1 == 1.0 {
					lp1 = t_end
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
						inv := espc_int_inv_approx(integral*t + int0)
						s = (inv - b) / a
					}
					lp1 = es.eval_with_offset(s, normalized_offset)
				}
				var l0, l1 Vec2
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
				output_line_with_transform(path_ix, l0, l1, transform, line_ix, lines, bbox)
				lp0 = lp1
			}
			last_p = this_p1
			last_q = this_q1
			last_t = t1
			// Advance segment to next range. Beginning of segment is the end of
			// this one. The number of trailing zeros represents the number of stack
			// frames to pop in the recursive version of adaptive subdivision, and
			// each stack pop represents doubling of the size of the range.
			t0_u += 1
			shift := bits.TrailingZeros32(t0_u)
			t0_u >>= shift
			dt *= float32(int(1) << shift)
		} else {
			// Subdivide; halve the size of the range while retaining its start.
			t0_u2 := uint64(t0_u) * 2
			if t0_u2 > math.MaxUint32 {
				t0_u2 = math.MaxUint32
			}
			t0_u = uint32(t0_u2)
			dt *= 0.5
		}
	}
}

func draw_cap(
	path_ix uint32,
	cap_style uint32,
	point Vec2,
	cap0 Vec2,
	cap1 Vec2,
	offset_tangent Vec2,
	transform Transform,
	line_ix *int,
	lines []renderer.LineSoup,
	bbox *IntBbox,
) {
	if cap_style == encoding.FlagsCapBitsRound {
		flatten_arc(
			path_ix,
			cap0,
			cap1,
			point,
			math.Pi,
			transform,
			line_ix,
			lines,
			bbox,
		)
		return
	}

	start := cap0
	end := cap1
	if cap_style == encoding.FlagsCapBitsSquare {
		v := offset_tangent
		p0 := start.Add(v)
		p1 := end.Add(v)
		output_line_with_transform(path_ix, start, p0, transform, line_ix, lines, bbox)
		output_line_with_transform(path_ix, p1, end, transform, line_ix, lines, bbox)
		start = p0
		end = p1
	}
	output_line_with_transform(path_ix, start, end, transform, line_ix, lines, bbox)
}

func draw_join(
	path_ix uint32,
	style_flags uint32,
	p0 Vec2,
	tan_prev Vec2,
	tan_next Vec2,
	n_prev Vec2,
	n_next Vec2,
	transform Transform,
	line_ix *int,
	lines []renderer.LineSoup,
	bbox *IntBbox,
) {
	front0 := p0.Add(n_prev)
	front1 := p0.Add(n_next)
	back0 := p0.Sub(n_next)
	back1 := p0.Sub(n_prev)

	cr := tan_prev.x*tan_next.y - tan_prev.y*tan_next.x
	d := tan_prev.dot(tan_next)

	switch style_flags & encoding.FlagsJoinMask {
	case encoding.FlagsJoinBitsBevel:
		if front0 != front1 && back0 != back1 {
			output_two_lines_with_transform(
				path_ix, front0, front1, back0, back1, transform, line_ix, lines, bbox,
			)
		}
	case encoding.FlagsJoinBitsMiter:
		hypot := jmath.Hypot32(cr, d)
		miter_limit := jmath.Float32(uint16(style_flags & encoding.MiterLimitMask))

		if 2.*hypot < (hypot+d)*miter_limit*miter_limit && cr != 0. {
			is_backside := cr > 0.
			var fp_last, fp_this, p Vec2
			if is_backside {
				fp_last = back1
			} else {
				fp_last = front0
			}
			if is_backside {
				fp_this = back0
			} else {
				fp_this = front1
			}
			if is_backside {
				p = back0
			} else {
				p = front0
			}

			v := fp_this.Sub(fp_last)
			h := (tan_prev.x*v.y - tan_prev.y*v.x) / cr
			miter_pt := fp_this.Sub(tan_next.Mul(h))

			output_line_with_transform(path_ix, p, miter_pt, transform, line_ix, lines, bbox)

			if is_backside {
				back0 = miter_pt
			} else {
				front0 = miter_pt
			}
		}
		output_two_lines_with_transform(
			path_ix, front0, front1, back0, back1, transform, line_ix, lines, bbox,
		)
	case encoding.FlagsJoinBitsRound:
		var arc0, arc1, other0, other1 Vec2
		if cr > 0. {
			arc0, arc1, other0, other1 = back0, back1, front0, front1
		} else {
			arc0, arc1, other0, other1 = front0, front1, back0, back1
		}
		flatten_arc(
			path_ix,
			arc0,
			arc1,
			p0,
			jmath.Abs32(jmath.Atan232(cr, d)),
			transform,
			line_ix,
			lines,
			bbox,
		)
		output_line_with_transform(path_ix, other0, other1, transform, line_ix, lines, bbox)
	default:
		panic("unreachable")
	}
}

func Flatten(_ *mem.Arena, n_wg uint32, resources []CPUBinding) {
	config := fromBytes[renderer.ConfigUniform](resources[0].(CPUBuffer))
	scene := safeish.SliceCast[[]uint32](resources[1].(CPUBuffer))
	tag_monoids := safeish.SliceCast[[]renderer.PathMonoid](resources[2].(CPUBuffer))
	path_bboxes := safeish.SliceCast[[]renderer.PathBbox](resources[3].(CPUBuffer))
	bump := fromBytes[renderer.BumpAllocators](resources[4].(CPUBuffer))
	lines := safeish.SliceCast[[]renderer.LineSoup](resources[5].(CPUBuffer))

	line_ix := 0
	pathtags := scene[config.Layout.PathTagBase:]
	pathdata := scene[config.Layout.PathDataBase:]

	for ix := range int(n_wg) * WG_SIZE {
		bbox := newIntBbox()
		tag := compute_tag_monoid(ix, pathtags, tag_monoids)
		path_ix := tag.monoid.PathIdx
		style_ix := tag.monoid.StyleIdx
		trans_ix := tag.monoid.TransIdx
		style_flags := scene[(config.Layout.StyleBase + style_ix)]
		if (tag.tag_byte & PATH_TAG_PATH) != 0 {
			out := &path_bboxes[path_ix]
			if (style_flags & encoding.FlagsFillBit) == 0 {
				out.DrawFlags = 0
			} else {
				out.DrawFlags = DRAW_INFO_FLAGS_FILL_RULE_BIT
			}
			out.TransIdx = trans_ix
		}

		seg_type := tag.tag_byte & PATH_TAG_SEG_TYPE
		if seg_type != 0 {
			is_stroke := (style_flags & encoding.FlagsStyleBit) != 0
			transform := transformRead(config.Layout.TransformBase, trans_ix, scene)
			pts := read_path_segment(tag, is_stroke, pathdata)

			if is_stroke {
				linewidth := math.Float32frombits(scene[(config.Layout.StyleBase + style_ix + 1)])
				offset := 0.5 * linewidth

				is_open := seg_type != PATH_TAG_LINETO
				is_stroke_cap_marker := (encoding.PathTag(tag.tag_byte) & encoding.PathTagSubpathEndBit) != 0
				if is_stroke_cap_marker {
					if is_open {
						// Draw start cap
						tangent := cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3)
						offset_tangent := tangent.normalize().Mul(offset)
						n := Vec2{-offset_tangent.y, offset_tangent.x}
						draw_cap(
							path_ix,
							(style_flags&encoding.FlagsStartCapMask)>>2,
							pts.p0,
							pts.p0.Sub(n),
							pts.p0.Add(n),
							offset_tangent.Mul(-1),
							transform,
							&line_ix,
							lines,
							&bbox,
						)
					} else {
						// Don't draw anything if the path is closed.
					}
				} else {
					// Read the neighboring segment.
					neighbor := read_neighboring_segment(ix+1, pathtags, pathdata, tag_monoids)
					tan_prev := cubic_end_tangent(pts.p0, pts.p1, pts.p2, pts.p3)
					tan_next := neighbor.tangent
					tan_start := cubic_start_tangent(pts.p0, pts.p1, pts.p2, pts.p3)
					// TODO: be consistent w/ robustness here

					// TODO: add NaN assertions to CPU shaders PR (when writing lines)
					// TODO: not all zero-length segments are getting filtered out
					// TODO: this is a hack. How to handle caps on degenerate stroke?
					// TODO: debug tricky stroke by isolation
					if tan_start.lengthSquared() < (TANGENT_THRESH * TANGENT_THRESH) {
						tan_start = Vec2{TANGENT_THRESH, 0.}
					}
					if tan_prev.lengthSquared() < (TANGENT_THRESH * TANGENT_THRESH) {
						tan_prev = Vec2{TANGENT_THRESH, 0.}
					}
					if tan_next.lengthSquared() < (TANGENT_THRESH * TANGENT_THRESH) {
						tan_next = Vec2{TANGENT_THRESH, 0.}
					}

					n_start := Vec2{-tan_start.y, tan_start.x}.normalize().Mul(offset)
					offset_tangent := tan_prev.normalize().Mul(offset)
					n_prev := Vec2{-offset_tangent.y, offset_tangent.x}
					tan_next_norm := tan_next.normalize()
					n_next := Vec2{-tan_next_norm.y, tan_next_norm.x}.Mul(offset)

					// Render offset curves
					flatten_euler(
						pts,
						path_ix,
						transform,
						offset,
						pts.p0.Add(n_start),
						pts.p3.Add(n_prev),
						&line_ix,
						lines,
						&bbox,
					)
					flatten_euler(
						pts,
						path_ix,
						transform,
						-offset,
						pts.p0.Sub(n_start),
						pts.p3.Sub(n_prev),
						&line_ix,
						lines,
						&bbox,
					)

					if neighbor.do_join {
						draw_join(
							path_ix,
							style_flags,
							pts.p3,
							tan_prev,
							tan_next,
							n_prev,
							n_next,
							transform,
							&line_ix,
							lines,
							&bbox,
						)
					} else {
						// Draw end cap.
						draw_cap(
							path_ix,
							style_flags&encoding.FlagsEndCapMask,
							pts.p3,
							pts.p3.Add(n_prev),
							pts.p3.Sub(n_prev),
							offset_tangent,
							transform,
							&line_ix,
							lines,
							&bbox,
						)
					}
				}
			} else {
				flatten_euler(
					pts,
					path_ix,
					transform,
					/*offset*/ 0.,
					pts.p0,
					pts.p3,
					&line_ix,
					lines,
					&bbox,
				)
			}
		}

		if int(path_ix) < len(path_bboxes) && (bbox.x1 > bbox.x0 || bbox.y1 > bbox.y0) {
			out := &path_bboxes[path_ix]
			out.X0 = min(out.X0, bbox.x0)
			out.Y0 = min(out.Y0, bbox.y0)
			out.X1 = max(out.X1, bbox.x1)
			out.Y1 = max(out.Y1, bbox.y1)
		}
	}
	bump.Lines = uint32(line_ix)
}
