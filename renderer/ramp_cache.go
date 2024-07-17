package renderer

import (
	"strings"

	"honnef.co/go/jello/gfx"
	"honnef.co/go/safeish"
)

type Ramps struct {
	Data   []uint32
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
	data    []uint32
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
	key := safeish.Cast[string](safeish.SliceCast[[]byte](stops))
	if entry, ok := rc.mapping[key]; ok {
		entry.epoch = rc.epoch
		return entry.id
	} else if len(rc.mapping) < retainedCount {
		id := uint32(len(rc.data) / numSamples)
		rc.data = append(rc.data, makeRamp(stops)...)
		// Copy the key so it no longer aliases a slice
		rc.mapping[strings.Clone(key)] = &rampCacheEntry{id, rc.epoch}
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

func makeRamp(stops []gfx.ColorStop) []uint32 {
	// OPT(dh): this could be an iterator instead

	out := make([]uint32, numSamples)

	lastU := float32(0.0)
	lastC := stops[0].Color
	thisU := lastU
	thisC := lastC
	j := 0
	for i := range numSamples {
		u := float32(i) / (numSamples - 1)
		for u > thisU {
			lastU = thisU
			lastC = thisC
			if j+1 < len(stops) {
				s := stops[j+1]
				thisU = float32(s.Offset)
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
			c = lerp(lastC, thisC, (u-lastU)/du)
		}
		out[i] = c.PremulUint32()
	}

	return out
}

func lerp(c gfx.Color, other gfx.Color, a float32) gfx.Color {
	l := func(x, y, a float32) float32 {
		return x*(1.0-a) + y*a
	}
	return gfx.Color{
		R: l(c.R, other.R, a),
		G: l(c.G, other.G, a),
		B: l(c.B, other.B, a),
		A: l(c.A, other.A, a),
	}
}
