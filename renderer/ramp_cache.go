// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package renderer

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"honnef.co/go/color"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/jmath"
	"honnef.co/go/safeish"
)

type Ramps struct {
	Data   [][4]jmath.Float16
	Width  uint32
	Height uint32
}

type rampCacheEntry struct {
	id    uint32
	epoch uint64
}

type rampCache struct {
	epoch uint64
	// mapping from []ColorStop
	mapping map[string]*rampCacheEntry
	data    [][4]jmath.Float16

	// slice reused across calls to add, used for building the map key.
	key []byte
}

const numSamples = 512
const retainedCount = 64

func (rc *rampCache) maintain() {
	rc.epoch++
	if len(rc.mapping) > retainedCount {
		for k, v := range rc.mapping {
			if v.id >= retainedCount {
				delete(rc.mapping, k)
			}
		}
		rc.data = rc.data[:retainedCount*numSamples]
	}
}

func (rc *rampCache) add(stops []gfx.ColorStop) uint32 {
	key := rc.key[:0]
	// Adding the number of stops makes the key unique for different length
	// sequences of colors that would have the same concatenation.
	key = binary.LittleEndian.AppendUint64(key, uint64(len(stops)))
	for _, stop := range stops {
		key = binary.LittleEndian.AppendUint32(key, math.Float32bits(stop.Offset))
		key = binary.LittleEndian.AppendUint64(key, math.Float64bits(stop.Color.Values[0]))
		key = binary.LittleEndian.AppendUint64(key, math.Float64bits(stop.Color.Values[1]))
		key = binary.LittleEndian.AppendUint64(key, math.Float64bits(stop.Color.Values[2]))
		key = binary.LittleEndian.AppendUint64(key, math.Float64bits(stop.Color.Alpha))
		key = append(key, stop.Color.Space.ID...)
	}
	rc.key = key[:0]

	keyStr := safeish.Cast[string](key)
	if entry, ok := rc.mapping[keyStr]; ok {
		entry.epoch = rc.epoch
		return entry.id
	} else if len(rc.mapping) < retainedCount {
		id := uint32(len(rc.data) / numSamples)
		rc.data = append(rc.data, makeRamp(stops)...)
		// Copy the key so it no longer aliases a slice
		rc.mapping[strings.Clone(keyStr)] = &rampCacheEntry{id, rc.epoch}
		return id
	} else {
		var reuseID uint32
		var reuseStops string
		var found bool
		for stops, entry := range rc.mapping {
			if entry.epoch+2 < rc.epoch {
				reuseID = entry.id
				reuseStops = stops
				found = true
				break
			}
		}
		if found {
			delete(rc.mapping, reuseStops)
			start := int(reuseID) * numSamples
			copy(rc.data[start:start+numSamples], makeRamp(stops))
			rc.mapping[string(safeish.SliceCast[[]byte](stops))] = &rampCacheEntry{reuseID, rc.epoch}
			return reuseID
		} else {
			id := uint32(len(rc.data) / numSamples)
			rc.data = append(rc.data, makeRamp(stops)...)
			return id
		}
	}
}

func (rc *rampCache) ramps() Ramps {
	return Ramps{
		Data:   rc.data,
		Width:  numSamples,
		Height: uint32(len(rc.data) / numSamples),
	}
}

func makeRamp(stops []gfx.ColorStop) [][4]jmath.Float16 {
	if len(stops) < 2 {
		panic("internal error: makeRamp needs at least two stops")
	}
	if stops[0].Offset != 0 {
		stops_ := make([]gfx.ColorStop, len(stops)+1)
		copy(stops_[1:], stops)
		stops_[0] = stops_[1]
		stops_[0].Offset = 0
		stops = stops_
	}
	out := make([][4]jmath.Float16, 0, numSamples)
	remaining := numSamples
	for i := 1; i < len(stops); i++ {
		prevStop := &stops[i-1]
		stop := &stops[i]
		var n int
		if i == len(stops)-1 {
			n = remaining
		} else {
			frac := (stop.Offset - prevStop.Offset)
			n = int(jmath.Round32(float32(numSamples) * frac))
			n = min(remaining, n)
		}
		remaining -= n
		// We use sRGB for the gradient because that's what people expect...
		// Eventually we'll support doing gradients in other color spaces, at
		// which point we'll default to something nicer, maybe even Oklch.
		switch n {
		case 0:
		case 1:
			out = append(out, gfx.Premul16(&stop.Color))
		default:
			for step := range color.Step(&prevStop.Color, &stop.Color, color.SRGB, color.LinearSRGB, n) {
				out = append(out, gfx.Premul16(&step))
			}
		}
	}
	if len(out) != numSamples {
		panic(fmt.Sprintf("internal error: tried to generate %d colors but ended up with %d", numSamples, len(out)))
	}
	return out
}
