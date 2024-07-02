package jello

import "math"

const epsilon = 1e-12

func abs32(f float32) float32 {
	return float32(math.Abs(float64(f)))
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

func (opt option[T]) unwrapOr(alt T) T {
	if opt.isSet {
		return opt.value
	} else {
		return alt
	}
}

func (opt option[T]) expect(msg string) T {
	if opt.isSet {
		return opt.value
	} else {
		panic(msg)
	}
}

func (opt *option[T]) take() option[T] {
	out := *opt
	opt.clear()
	return out
}

func some[T any](v T) option[T] {
	return option[T]{
		isSet: true,
		value: v,
	}
}
