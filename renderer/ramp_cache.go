package renderer

import (
	"honnef.co/go/brush"
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

func (rc *rampCache) add(stops []brush.ColorStop) uint32 {
	key := string(safeish.SliceCast[[]byte](stops))
	if entry, ok := rc.mapping[key]; ok {
		entry.epoch = rc.epoch
		return entry.id
	} else if len(rc.mapping) < retainedCount {
		id := uint32(len(rc.data) / numSamples)
		rc.data = append(rc.data, makeRamp(stops)...)
		rc.mapping[key] = &rampCacheEntry{id, rc.epoch}
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

func makeRamp(stops []brush.ColorStop) []uint32 {
	// OPT(dh): this could be an iterator instead

	out := make([]uint32, numSamples)

	lastU := 0.0
	lastC := colorf64{
		float64(stops[0].Color.R) / 255.0,
		float64(stops[0].Color.G) / 255.0,
		float64(stops[0].Color.B) / 255.0,
		float64(stops[0].Color.A) / 255.0,
	}
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
				thisC = colorF64FromColor(s.Color)
				j++
			} else {
				break
			}
		}
		du := thisU - lastU
		var c colorf64
		if du < 1e-9 {
			c = thisC
		} else {
			c = lastC.lerp(&thisC, (u-lastU)/du)
		}
		out[i] = c.asPremulU32()
	}

	return out
}

type colorf64 [4]float64

func colorF64FromColor(c brush.Color) colorf64 {
	return colorf64{
		float64(c.R) / 255.0,
		float64(c.G) / 255.0,
		float64(c.B) / 255.0,
		float64(c.A) / 255.0,
	}
}

func (c *colorf64) lerp(other *colorf64, a float64) colorf64 {
	l := func(x, y, a float64) float64 {
		return x*(1.0-a) + y*a
	}
	return colorf64{
		l(c[0], other[0], a),
		l(c[1], other[1], a),
		l(c[2], other[2], a),
		l(c[3], other[3], a),
	}
}

func (c *colorf64) asPremulU32() uint32 {
	clamp := func(v, low, high float64) float64 {
		if v < low {
			return low
		}
		if v > high {
			return high
		}
		return v
	}
	a := clamp(c[3], 0.0, 1.0)
	r := uint32(clamp(c[0]*a, 0.0, 1.0) * 255.0)
	g := uint32(clamp(c[1]*a, 0.0, 1.0) * 255.0)
	b := uint32(clamp(c[2]*a, 0.0, 1.0) * 255.0)

	ua := uint32(a * 255.0)
	return r | (g << 8) | (b << 16) | (ua << 24)
}
