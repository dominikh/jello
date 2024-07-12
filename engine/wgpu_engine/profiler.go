package wgpu_engine

import (
	"time"

	"honnef.co/go/jello/profiler"
	"honnef.co/go/safeish"
	"honnef.co/go/wgpu"
)

const maxProfilerTimestamps = 1024

type Profiler struct {
	dev *wgpu.Device

	// started groups that haven't been resolved yet
	groups         []*ProfilerGroup
	resolvedGroups []*ProfilerGroup
	mappedGroups   []*ProfilerGroup

	// free list of query sets
	querySets []*wgpu.QuerySet
	// free list of buffers
	resolveBuffers []*wgpu.Buffer
	// free list of buffers
	mapBuffers []*wgpu.Buffer
}

func NewProfiler(dev *wgpu.Device) *Profiler {
	return &Profiler{
		dev: dev,
	}
}

func (p *Profiler) Start(tag uint64) *ProfilerGroup {
	g := &ProfilerGroup{
		Tag:        tag,
		set:        &profilerQuerySet{set: p.getQuerySet()},
		cpuStart:   time.Now(),
		resolveBuf: p.getResolveBuffer(),
		mapBuf:     p.getMapBuffer(),
		ch:         make(chan error, 1),
	}
	p.groups = append(p.groups, g)
	return g
}

type profilerQuerySet struct {
	set *wgpu.QuerySet
	id  uint32
}

func (set *profilerQuerySet) nextID() uint32 {
	id := set.id
	set.id++
	return id
}

type ProfilerGroup struct {
	Tag        uint64
	Label      string
	set        *profilerQuerySet
	cpuStart   time.Time
	cpuEnd     time.Time
	children   []*ProfilerGroup
	gpuQueries []ProfilerQuery

	// set for top-level groups only
	resolveBuf *wgpu.Buffer
	mapBuf     *wgpu.Buffer
	ch         <-chan error
}

func (g *ProfilerGroup) End() {
	g.cpuEnd = time.Now()
}

// TODO(dh): having both Start and Nest sucks, but we need Start to implement
// profiler.ProfileGroup, and we need the interface so packages don't need a
// direct dependency on wgpu.

func (g *ProfilerGroup) Start(label string) profiler.ProfilerGroup {
	return g.Nest(label)
}

func (g *ProfilerGroup) Nest(label string) *ProfilerGroup {
	cg := &ProfilerGroup{
		Label:    label,
		set:      g.set,
		cpuStart: time.Now(),
	}
	g.children = append(g.children, cg)
	return cg
}

type ProfilerQuery struct {
	Label   string
	startID uint32
	endID   uint32
}

func (g *ProfilerGroup) Compute(label string) *wgpu.ComputePassTimestampWrites {
	startID, endID := g.set.nextID(), g.set.nextID()
	q := ProfilerQuery{
		Label:   label,
		startID: startID,
		endID:   endID,
	}
	g.gpuQueries = append(g.gpuQueries, q)

	return &wgpu.ComputePassTimestampWrites{
		QuerySet:                  g.set.set,
		BeginningOfPassWriteIndex: startID,
		EndOfPassWriteIndex:       endID,
	}
}

func (g *ProfilerGroup) Render(label string) *wgpu.RenderPassTimestampWrites {
	startID, endID := g.set.nextID(), g.set.nextID()
	q := ProfilerQuery{
		Label:   label,
		startID: startID,
		endID:   endID,
	}
	g.gpuQueries = append(g.gpuQueries, q)

	return &wgpu.RenderPassTimestampWrites{
		QuerySet:                  g.set.set,
		BeginningOfPassWriteIndex: startID,
		EndOfPassWriteIndex:       endID,
	}
}

func (g *ProfilerGroup) Begin(enc *wgpu.CommandEncoder, label string) ProfilerSpan {
	startID, endID := g.set.nextID(), g.set.nextID()
	q := ProfilerQuery{
		Label:   label,
		startID: startID,
		endID:   endID,
	}
	g.gpuQueries = append(g.gpuQueries, q)

	enc.WriteTimestamp(g.set.set, startID)
	return ProfilerSpan{
		set:   g.set.set,
		endID: endID,
	}
}

type ProfilerSpan struct {
	set   *wgpu.QuerySet
	endID uint32
}

func (span ProfilerSpan) End(enc *wgpu.CommandEncoder) {
	enc.WriteTimestamp(span.set, span.endID)
}

func (p *Profiler) getQuerySet() *wgpu.QuerySet {
	if len(p.querySets) == 0 {
		return p.dev.CreateQuerySet(&wgpu.QuerySetDescriptor{
			Type:  wgpu.QueryTypeTimestamp,
			Count: maxProfilerTimestamps,
		})
	}
	q := p.querySets[len(p.querySets)-1]
	p.querySets = p.querySets[:len(p.querySets)-1]
	return q
}

func (p *Profiler) getResolveBuffer() *wgpu.Buffer {
	if len(p.resolveBuffers) == 0 {
		return p.dev.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: wgpu.BufferUsageQueryResolve | wgpu.BufferUsageCopySrc,
			Size:  maxProfilerTimestamps * 16,
		})
	}
	buf := p.resolveBuffers[len(p.resolveBuffers)-1]
	p.resolveBuffers = p.resolveBuffers[:len(p.resolveBuffers)-1]
	return buf
}

func (p *Profiler) getMapBuffer() *wgpu.Buffer {
	if len(p.mapBuffers) == 0 {
		return p.dev.CreateBuffer(&wgpu.BufferDescriptor{
			Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
			Size:  maxProfilerTimestamps * 16,
		})
	}
	buf := p.mapBuffers[len(p.mapBuffers)-1]
	p.mapBuffers = p.mapBuffers[:len(p.mapBuffers)-1]
	return buf
}

func (p *Profiler) Resolve(enc *wgpu.CommandEncoder) {
	for _, g := range p.groups {
		enc.ResolveQuerySet(g.set.set, 0, g.set.id, g.resolveBuf, 0)
		enc.CopyBufferToBuffer(g.resolveBuf, 0, g.mapBuf, 0, uint64(g.set.id)*16)
	}
	p.resolvedGroups = p.groups
	// OPT(dh): keep a pool of slices to reuse
	p.groups = nil
}

func (p *Profiler) Map() {
	for _, g := range p.resolvedGroups {
		g.ch = g.mapBuf.Map(wgpu.MapModeRead, 0, int(g.set.id)*16)
	}
	p.mappedGroups = append(p.mappedGroups, p.resolvedGroups...)
	// OPT(dh): keep a pool of slices to reuse
	p.resolvedGroups = nil
}

type ProfilerResult struct {
	Tag      uint64
	Label    string
	CPUStart time.Time
	CPUEnd   time.Time
	Queries  []ProfilerQueryResult
	Children []ProfilerResult
}

type ProfilerQueryResult struct {
	Label string
	Start uint64
	End   uint64
}

func (p *Profiler) populateResult(g *ProfilerGroup, res *ProfilerResult, values []uint64) {
	*res = ProfilerResult{
		Tag:      g.Tag,
		Label:    g.Label,
		CPUStart: g.cpuStart,
		CPUEnd:   g.cpuEnd,
		Queries:  make([]ProfilerQueryResult, len(g.gpuQueries)),
		Children: make([]ProfilerResult, len(g.children)),
	}

	for qi, q := range g.gpuQueries {
		res.Queries[qi] = ProfilerQueryResult{
			Label: q.Label,
			Start: values[q.startID],
			End:   values[q.endID],
		}
	}

	for ci, c := range g.children {
		p.populateResult(c, &res.Children[ci], values)
	}
}

func (p *Profiler) Collect() []ProfilerResult {
	var out []ProfilerResult

	for i, g := range p.mappedGroups {
		select {
		case err := <-g.ch:
			if err != nil {
				panic(err)
			}
			out = append(out, ProfilerResult{})
			data := safeish.SliceCast[[]uint64](g.mapBuf.ReadOnlyMappedRange(0, int(g.set.id)*16))
			p.populateResult(g, &out[len(out)-1], data)
			g.mapBuf.Unmap()
			p.querySets = append(p.querySets, g.set.set)
			p.mapBuffers = append(p.mapBuffers, g.mapBuf)
			p.resolveBuffers = append(p.resolveBuffers, g.resolveBuf)
		default:
			// We stop at the first missing group so that we return groups in
			// order of creation.
			copy(p.mappedGroups, p.mappedGroups[i:])
			clear(p.mappedGroups[len(p.mappedGroups)-i:])
			p.mappedGroups = p.mappedGroups[:len(p.mappedGroups)-i]
			return out
		}
	}
	// If we get here then all groups have been collected
	// OPT(dh): reuse slice
	p.mappedGroups = nil
	return out
}
