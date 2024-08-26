// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package renderer

import (
	"encoding/binary"
	"math"
	"strings"
	"unsafe"

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
		if stop.Color == nil {
			panic("nil color in gradient")
		}

		key = binary.LittleEndian.AppendUint32(key, math.Float32bits(stop.Offset))
		type iface struct {
			tab *struct {
				_   uintptr
				typ *struct {
					size uintptr
					_    uintptr
					hash uint32
				}
			}
			data unsafe.Pointer
		}
		v := safeish.Cast[*iface](&stop.Color)
		// This assumes that the type doesn't move in memory. But even if it
		// does, because we're also incorporating the type hash, the odds of a
		// collision are miniscule, so at worst we fail to reuse a cached entry.
		key = binary.LittleEndian.AppendUint64(key, uint64(uintptr(unsafe.Pointer(v.tab.typ))))
		key = binary.LittleEndian.AppendUint32(key, v.tab.typ.hash)
		if v.data != nil {
			key = append(key, unsafe.Slice((*byte)(v.data), v.tab.typ.size)...)
		}
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
	// OPT(dh): this could be an iterator instead
	out := make([][4]jmath.Float16, numSamples)

	lastU := float64(0.0)
	lastC := stops[0].Color
	thisU := lastU
	thisC := lastC
	j := 0
	for i := range numSamples {
		u := float64(i) / (numSamples - 1)
		for u > thisU {
			lastU = thisU
			lastC = thisC
			if j+1 < len(stops) {
				s := stops[j+1]
				thisU = float64(s.Offset)
				thisC = s.Color
				j++
			} else {
				break
			}
		}
		du := thisU - lastU
		var c gfx.Color
		if du < 1e-9 {
			c = thisC
		} else {
			c = lastC.Lerp(thisC, (u-lastU)/du)
		}
		out[i] = c.LinearSRGB().Premul16()
	}

	return out
}
