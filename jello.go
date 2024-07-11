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
