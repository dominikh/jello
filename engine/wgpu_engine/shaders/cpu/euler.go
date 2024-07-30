// Copyright 2023 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

package cpu

import (
	"math"

	"honnef.co/go/jello/jmath"
)

// Utility functions for Euler Spiral based stroke expansion.

// Threshold for tangents to be considered near zero length
const tangentThresh = 1e-6

// This struct contains parameters derived from a cubic Bézier for the
// purpose of fitting a G1 continuous Euler spiral segment and estimating
// the Fréchet distance.
//
// The tangent angles represent deviation from the chord, so that when they
// are equal, the corresponding Euler spiral is a circular arc.
type cubicParams struct {
	// Tangent angle relative to chord at start.
	th0 float32
	// Tangent angle relative to chord at end.
	th1 float32
	// The effective chord length, always a robustly nonzero value.
	chordLen float32
	// The estimated error between the source cubic and the proposed Euler spiral.
	err float32
}

type eulerParams struct {
	th0 float32
	th1 float32
	k0  float32
	k1  float32
	ch  float32
}

type eulerSeg struct {
	p0     vec2
	p1     vec2
	params eulerParams
}

// Compute parameters from endpoints and derivatives.
//
// This function is designed to be robust across a wide range of inputs. In
// particular, it splits between near-zero chord length and the happy path.
// In the former case, the parameters for the Euler spiral would not be valid,
// so it proposes a straight line and computes a pretty good (conservative)
// estimate of the Fréchet distance between that line and the source cubic.
//
// Computing an accurate estimate here fixes two tricky cases: very short
// lines, in which the error will be below threshold and the flatten logic will
// output a single line segment without subdividing, and loop cases with a
// short chord, in which case the error will exceed the threshold, and the
// chords of the subdivided pieces will be longer.
//
// An additional case is the near-cusp where the proposed Euler spiral has
// a 180 degree U-turn (or, more generally, one angle exceeds 90 degrees and
// the other does not). In that case, the resulting Euler spiral is quite
// well defined (with finite curvature, so that its offset will generate a
// near-semicircle, preserving G1 continuity), but the analytic error
// calculation would be a huge overestimate. In that case, we just return
// a rough estimate of the distance between the chord and the spiral segment.
func cubicParamsFromPointsDerivs(p0, p1, q0, q1 vec2, dt float32) cubicParams {
	chord := p1.sub(p0)
	chordSquared := chord.lengthSquared()
	chordLen := jmath.Sqrt32(chordSquared)
	// Chord is near-zero; straight line case.
	if chordSquared < tangentThresh*tangentThresh {
		// This error estimate was determined empirically through randomized
		// testing, though it is likely it can be derived analytically.
		chordError := jmath.Sqrt32((9.0/32.0)*(q0.lengthSquared()+q1.lengthSquared())) * dt
		return cubicParams{
			th0:      0.0,
			th1:      0.0,
			chordLen: tangentThresh,
			err:      chordError,
		}
	}
	scale := dt / chordSquared
	h0 := vec2{
		q0.x*chord.x + q0.y*chord.y,
		q0.y*chord.x - q0.x*chord.y,
	}
	th0 := h0.atan2()
	d0 := h0.length() * scale
	h1 := vec2{
		q1.x*chord.x + q1.y*chord.y,
		q1.x*chord.y - q1.y*chord.x,
	}
	th1 := h1.atan2()
	d1 := h1.length() * scale
	// Robustness note: we may want to clamp the magnitude of the angles to
	// a bit less than pi. Perhaps here, perhaps downstream.

	// Estimate error of geometric Hermite interpolation to Euler spiral.
	cth0 := jmath.Cos32(th0)
	cth1 := jmath.Cos32(th1)
	var err float32
	if cth0*cth1 < 0.0 {
		// A value of 2.0 represents the approximate worst case distance
		// from an Euler spiral with 0 and pi tangents to the chord. It
		// is not very critical; doubling the value would result in one more
		// subdivision in effectively a binary search for the cusp, while too
		// small a value may result in the actual error exceeding the bound.
		err = 2.0
	} else {
		// Protect against divide-by-zero. This happens with a double cusp, so
		// should in the general case cause subdivisions.
		e0 := (2. / 3.) / max(1.0+cth0, 1e-9)
		e1 := (2. / 3.) / max(1.0+cth1, 1e-9)
		s0 := jmath.Sin32(th0)
		s1 := jmath.Sin32(th1)
		// Note: some other versions take sin of s0 + s1 instead. Those are incorrect.
		// Strangely, calibration is the same, but more work could be done.
		s01 := cth0*s1 + cth1*s0
		amin := 0.15 * (2.*e0*s0 + 2.*e1*s1 - e0*e1*s01)
		a := 0.15 * (2.*d0*s0 + 2.*d1*s1 - d0*d1*s01)
		aerr := jmath.Abs32(a - amin)
		symm := jmath.Abs32(th0 + th1)
		asymm := jmath.Abs32(th0 - th1)
		dist := jmath.Hypot32(d0-e0, d1-e1)
		ctr := 4.625e-6*jmath.Pow32(symm, 5) + 7.5e-3*asymm*(symm*symm)
		haloSymm := 5e-3 * symm * dist
		haloAsymm := 7e-2 * asymm * dist
		err = ctr + 1.55*aerr + haloSymm + haloAsymm
	}
	err *= chordLen
	return cubicParams{
		th0,
		th1,
		chordLen,
		err,
	}
}

func eulerParamsFromAngles(th0, th1 float32) eulerParams {
	k0 := th0 + th1
	dth := th1 - th0
	d2 := dth * dth
	k2 := k0 * k0
	a := float32(6.0)
	a -= d2 * (1. / 70.)
	a -= (d2 * d2) * (1. / 10780.)
	a += (d2 * d2 * d2) * 2.769178184818219e-07
	b := -0.1 + d2*(1./4200.) + d2*d2*1.6959677820260655e-05
	c := -1./1400. + d2*6.84915970574303e-05 - k2*7.936475029053326e-06
	a += (b + c*k2) * k2
	k1 := dth * a

	// calculation of chord
	ch := float32(1.0)
	ch -= d2 * (1. / 40.)
	ch += (d2 * d2) * 0.00034226190482569864
	ch -= (d2 * d2 * d2) * 1.9349474568904524e-06
	b = -1./24. + d2*0.0024702380951963226 - d2*d2*3.7297408997537985e-05
	c = 1./1920. - d2*4.87350869747975e-05 - k2*3.1001936068463107e-06
	ch += (b + c*k2) * k2
	return eulerParams{
		th0,
		th1,
		k0,
		k1,
		ch,
	}
}

func (ep eulerParams) evalTh(t float32) float32 {
	return (ep.k0+0.5*ep.k1*(t-1.0))*t - ep.th0
}

func (ep eulerParams) eval(t float32) vec2 {
	thm := ep.evalTh(t * 0.5)
	k0 := ep.k0
	k1 := ep.k1
	u, v := integrateEuler10((k0+k1*(0.5*t-0.5))*t, k1*t*t)
	s := t / ep.ch * jmath.Sin32(thm)
	c := t / ep.ch * jmath.Cos32(thm)
	x := u*c - v*s
	y := -v*c - u*s
	return vec2{x, y}
}

func (ep eulerParams) evalWithOffset(t, offset float32) vec2 {
	th := ep.evalTh(t)
	v := vec2{offset * jmath.Sin32(th), offset * jmath.Cos32(th)}
	return ep.eval(t).add(v)
}

func eulerSegFromParams(p0, p1 vec2, params eulerParams) eulerSeg {
	return eulerSeg{p0, p1, params}
}

// Note: offset provided is normalized so that 1 = chord length, while
// the return value is in the same coordinate space as the endpoints.
func (es eulerSeg) evalWithOffset(t, normalizedOffset float32) vec2 {
	chord := es.p1.sub(es.p0)
	v := es.params.evalWithOffset(t, normalizedOffset)
	x, y := v.x, v.y
	return vec2{
		es.p0.x + chord.x*x - chord.y*y,
		es.p0.y + chord.x*y + chord.y*x,
	}
}

// Integrate Euler spiral.
//
// This is a 10th order polynomial. The evaluation method is explained in
// Raph's thesis in section 8.1.2.
//
// This choice of polynomial is fairly conservative, as it will produce
// very good accuracy for angles up to about 1 radian, and those angles
// should almost never happen (the exception being cusps). One possibility
// to explore is using a lower degree polynomial here, but changing the
// tuning parameters for subdivision so the additional error here is also
// factored into the error threshold. However, two cautions against that:
// First, it doesn't really address the cusp case, where angles will remain
// large even after further subdivision, and second, the cost of even this
// more conservative choice is modest; it's just some multiply-adds.
func integrateEuler10(k0, k1 float32) (float32, float32) {
	t1_1 := k0
	t1_2 := 0.5 * k1
	t2_2 := t1_1 * t1_1
	t2_3 := 2. * (t1_1 * t1_2)
	t2_4 := t1_2 * t1_2
	t3_4 := t2_2*t1_2 + t2_3*t1_1
	t3_6 := t2_4 * t1_2
	t4_4 := t2_2 * t2_2
	t4_5 := 2. * (t2_2 * t2_3)
	t4_6 := 2.*(t2_2*t2_4) + t2_3*t2_3
	t4_7 := 2. * (t2_3 * t2_4)
	t4_8 := t2_4 * t2_4
	t5_6 := t4_4*t1_2 + t4_5*t1_1
	t5_8 := t4_6*t1_2 + t4_7*t1_1
	t6_6 := t4_4 * t2_2
	t6_7 := t4_4*t2_3 + t4_5*t2_2
	t6_8 := t4_4*t2_4 + t4_5*t2_3 + t4_6*t2_2
	t7_8 := t6_6*t1_2 + t6_7*t1_1
	t8_8 := t6_6 * t2_2
	u := float32(1.0)
	u -= (1./24.)*t2_2 + (1./160.)*t2_4
	u += (1./1920.)*t4_4 + (1./10752.)*t4_6 + (1./55296.)*t4_8
	u -= (1./322560.)*t6_6 + (1./1658880.)*t6_8
	u += (1. / 92897280.) * t8_8
	v := (1. / 12.) * t1_2
	v -= (1./480.)*t3_4 + (1./2688.)*t3_6
	v += (1./53760.)*t5_6 + (1./276480.)*t5_8
	v -= (1. / 11612160.) * t7_8
	return u, v
}

const break1 = 0.8
const break2 = 1.25
const break3 = 2.1
const sinScale = 1.0976991822760038
const quadA1 = 0.6406
const quadB1 = -0.81
const quadC1 = 0.9148117935952064
const quadA2 = 0.5
const quadB2 = -0.156
const quadC2 = 0.16145779359520596

func espcIntApprox(x float32) float32 {
	y := jmath.Abs32(x)
	var a float32
	if y < break1 {
		a = jmath.Sin32(sinScale*y) * (1.0 / sinScale)
	} else if y < break2 {
		a = (jmath.Sqrt32(8.0)/3.0)*(y-1.0)*jmath.Sqrt32(jmath.Abs32(y-1.0)) + math.Pi/4
	} else {
		var a_, b, c float32
		if y < break3 {
			a_, b, c = quadA1, quadB1, quadC1
		} else {
			a_, b, c = quadA2, quadB2, quadC2
		}
		a = a_*y*y + b*y + c
	}
	return jmath.Copysign32(a, x)
}

func espcIntInvApprox(x float32) float32 {
	y := jmath.Abs32(x)
	var a float32
	if y < 0.7010707591262915 {
		a = jmath.Asin32(x*sinScale) * (1.0 / sinScale)
	} else if y < 0.903249293595206 {
		b := y - math.Pi/4
		u := jmath.Copysign32(jmath.Pow32(jmath.Abs32(b), (2.0/3.0)), b)
		a = u*jmath.Cbrt32(9.0/8.) + 1.0
	} else {
		var u, v, w float32
		if y < 2.038857793595206 {
			const B = 0.5 * quadB1 / quadA1
			u, v, w = B*B-quadC1/quadA1, 1.0/quadA1, B
		} else {
			const B = 0.5 * quadB2 / quadA2
			u, v, w = B*B-quadC2/quadA2, 1.0/quadA2, B
		}
		a = jmath.Sqrt32(u+v*y) - w
	}
	return jmath.Copysign32(a, x)
}
