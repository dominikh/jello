// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package wgpu_engine

import (
	"time"

	"honnef.co/go/jello/mem"
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
	// slice to reuse for next frame
	spare []*ProfilerGroup

	// free list of query sets
	querySets []*wgpu.QuerySet
	// free list of buffers
	resolveBuffers []*wgpu.Buffer
	// free list of buffers
	mapBuffers []*wgpu.Buffer
	// free list of profiler groups
	freeGroups []*ProfilerGroup
	// free list of profiler results
	results []ProfilerResult
}

func NewProfiler(dev *wgpu.Device) *Profiler {
	return &Profiler{
		dev: dev,
	}
}

func NewNopProfiler() *Profiler {
	return nil
}

func (p *Profiler) Start(tag uint64) *ProfilerGroup {
	if p == nil {
		return nil
	}

	g := p.getGroup()
	// Don't use *g = ProfilerGroup{...} so that we reuse g.children and
	// g.gpuQueries.
	g.profiler = p
	g.Tag = tag
	g.set = profilerQuerySet{set: p.getQuerySet()}
	g.cpuStart = time.Now()
	g.resolveBuf = p.getResolveBuffer()
	g.mapBuf = p.getMapBuffer()
	p.groups = append(p.groups, g)
	return g
}

func (p *Profiler) getGroup() *ProfilerGroup {
	if len(p.freeGroups) > 0 {
		g := p.freeGroups[len(p.freeGroups)-1]
		p.freeGroups = p.freeGroups[:len(p.freeGroups)-1]
		clear(g.children)
		clear(g.gpuQueries)
		g.children = g.children[:0]
		g.gpuQueries = g.gpuQueries[:0]
		g.cpuEnd = time.Time{}
		g.resolveBuf = nil
		g.mapBuf = nil
		g.ch = nil
		return g
	} else {
		return &ProfilerGroup{}
	}
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
	set        profilerQuerySet
	cpuStart   time.Time
	cpuEnd     time.Time
	children   []*ProfilerGroup
	gpuQueries []ProfilerQuery
	profiler   *Profiler
	parent     *ProfilerGroup

	// set for top-level groups only
	resolveBuf *wgpu.Buffer
	mapBuf     *wgpu.Buffer
	ch         <-chan error
}

func (g *ProfilerGroup) End() {
	if g == nil {
		return
	}

	if !g.cpuEnd.IsZero() {
		panic("trying to end same group twice")
	}
	g.cpuEnd = time.Now()
	if g.parent != nil {
		g.parent.set.id = g.set.id
	}
}

// TODO(dh): having both Start and Nest sucks, but we need Start to implement
// profiler.ProfileGroup, and we need the interface so packages don't need a
// direct dependency on wgpu.

func (g *ProfilerGroup) Start(label string) profiler.ProfilerGroup {
	if g == nil {
		return (*ProfilerGroup)(nil)
	}
	return g.Nest(label)
}

func (g *ProfilerGroup) Nest(label string) *ProfilerGroup {
	if g == nil {
		return nil
	}
	cg := g.profiler.getGroup()
	// Don't use *cg = ProfilerGroup{...} so that we reuse cg.children and
	// cg.gpuQueries.
	cg.profiler = g.profiler
	cg.Label = label
	cg.set = g.set
	cg.cpuStart = time.Now()
	cg.parent = g
	g.children = append(g.children, cg)
	return cg
}

type ProfilerQuery struct {
	Label   string
	startID uint32
	endID   uint32
}

func (g *ProfilerGroup) Compute(arena *mem.Arena, label string) *wgpu.ComputePassTimestampWrites {
	if g == nil {
		return nil
	}
	startID, endID := g.set.nextID(), g.set.nextID()
	q := ProfilerQuery{
		Label:   label,
		startID: startID,
		endID:   endID,
	}
	g.gpuQueries = append(g.gpuQueries, q)

	return mem.Make(arena, wgpu.ComputePassTimestampWrites{
		QuerySet:                  g.set.set,
		BeginningOfPassWriteIndex: startID,
		EndOfPassWriteIndex:       endID,
	})
}

func (g *ProfilerGroup) Render(arena *mem.Arena, label string) *wgpu.RenderPassTimestampWrites {
	if g == nil {
		return nil
	}
	startID, endID := g.set.nextID(), g.set.nextID()
	q := ProfilerQuery{
		Label:   label,
		startID: startID,
		endID:   endID,
	}
	g.gpuQueries = append(g.gpuQueries, q)

	return mem.Make(arena, wgpu.RenderPassTimestampWrites{
		QuerySet:                  g.set.set,
		BeginningOfPassWriteIndex: startID,
		EndOfPassWriteIndex:       endID,
	})
}

func (g *ProfilerGroup) Begin(enc *wgpu.CommandEncoder, label string) ProfilerSpan {
	if g == nil {
		return ProfilerSpan{}
	}
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
	if span.set == nil {
		return
	}
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
	if p == nil {
		return
	}
	for _, g := range p.groups {
		enc.ResolveQuerySet(g.set.set, 0, g.set.id, g.resolveBuf, 0)
		enc.CopyBufferToBuffer(g.resolveBuf, 0, g.mapBuf, 0, uint64(g.set.id)*16)
	}
	p.resolvedGroups = p.groups
	p.groups = p.spare[:0]
}

func (p *Profiler) Map() {
	if p == nil {
		return
	}
	for _, g := range p.resolvedGroups {
		g.ch = g.mapBuf.Map(p.dev, wgpu.MapModeRead, 0, int(g.set.id)*16)
	}
	p.mappedGroups = append(p.mappedGroups, p.resolvedGroups...)
	clear(p.resolvedGroups)
	p.spare = p.resolvedGroups[:0]
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
	// Don't use *res = ProfilerResult{...} so that we reuse res.Children and
	// c.Queries.
	res.Tag = g.Tag
	res.Label = g.Label
	res.CPUStart = g.cpuStart
	res.CPUEnd = g.cpuEnd
	if cap(res.Queries) >= len(g.gpuQueries) {
		res.Queries = res.Queries[:len(g.gpuQueries)]
	} else {
		res.Queries = make([]ProfilerQueryResult, len(g.gpuQueries))
	}
	if cap(res.Children) >= len(g.children) {
		res.Children = res.Children[:len(g.children)]
	} else {
		res.Children = make([]ProfilerResult, len(g.children))
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

// Collect returns all available profiler results. The return value is only
// valid until the next call to Collect.
func (p *Profiler) Collect() []ProfilerResult {
	if p == nil {
		return nil
	}
	out := p.results[:0]

	var returnGroups func(gs ...*ProfilerGroup)
	returnGroups = func(gs ...*ProfilerGroup) {
		p.freeGroups = append(p.freeGroups, gs...)
		for _, g := range gs {
			returnGroups(g.children...)
		}
	}

	for i, g := range p.mappedGroups {
		select {
		case err := <-g.ch:
			if err != nil {
				panic(err)
			}
			if cap(out) > len(out) {
				out = out[:len(out)+1]
			} else {
				out = append(out, ProfilerResult{})
			}
			data := safeish.SliceCast[[]uint64](g.mapBuf.ReadOnlyMappedRange(0, int(g.set.id)*16))
			p.populateResult(g, &out[len(out)-1], data)
			g.mapBuf.Unmap()
			p.querySets = append(p.querySets, g.set.set)
			p.mapBuffers = append(p.mapBuffers, g.mapBuf)
			p.resolveBuffers = append(p.resolveBuffers, g.resolveBuf)
		default:
			// We stop at the first missing group so that we return groups in
			// order of creation.
			returnGroups(p.mappedGroups[:i]...)
			copy(p.mappedGroups, p.mappedGroups[i:])
			clear(p.mappedGroups[len(p.mappedGroups)-i:])
			p.mappedGroups = p.mappedGroups[:len(p.mappedGroups)-i]
			p.results = out[:0]
			return out
		}
	}
	// If we get here then all groups have been collected
	returnGroups(p.mappedGroups...)
	clear(p.mappedGroups)
	p.mappedGroups = p.mappedGroups[:0]
	p.results = out[:0]
	return out
}
