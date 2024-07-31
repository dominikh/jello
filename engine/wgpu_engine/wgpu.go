// Copyright 2023 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package wgpu_engine

// OPT reuse bind groups

import (
	"fmt"
	"image"
	"math"
	"math/bits"
	"sync"

	"honnef.co/go/jello/engine/wgpu_engine/shaders/cpu"
	"honnef.co/go/jello/mem"
	"honnef.co/go/jello/renderer"
	"honnef.co/go/safeish"
	"honnef.co/go/wgpu"
)

type uninitializedShader struct {
	Wgsl     []byte
	Label    string
	Entries  []wgpu.BindGroupLayoutEntry
	ShaderID renderer.ShaderID
}

type Engine struct {
	Device              *wgpu.Device
	renderer            *renderer.Renderer
	shaders             []shader
	pool                resourcePool
	downloads           map[renderer.ResourceID]*wgpu.Buffer
	shadersToInitialize []uninitializedShader
	UseCPU              bool

	resident    persistentBindMap
	resolver    *renderer.Resolver
	blit        *blitPipeline
	fullShaders *renderer.FullShaders
	target      *targetTexture
}

type wgpuShader struct {
	label           string
	pipeline        *wgpu.ComputePipeline
	bindGroupLayout *wgpu.BindGroupLayout
}

type cpuShaderKind int

const (
	cpuShaderKindMissing cpuShaderKind = iota
	cpuShaderKindPresent
	cpuShaderKindSkipped
)

type cpuShader struct {
	kind   cpuShaderKind
	shader func(*mem.Arena, uint32, []cpu.CPUBinding)
}

type shader struct {
	Label string
	WGPU  *wgpuShader
	CPU   *cpuShader
}

func (s shader) Select() any {
	if s.CPU != nil {
		return s.CPU
	} else if s.WGPU != nil {
		return s.WGPU
	} else {
		panic(fmt.Sprintf("no available shader for %s", s.Label))
	}
}

type ExternalResource interface {
	// One of ExternalBuffer and ExternalImage
}

type ExternalBuffer struct {
	Proxy  renderer.BufferProxy
	Buffer *wgpu.Buffer
}

type ExternalImage struct {
	Proxy renderer.ImageProxy
	View  *wgpu.TextureView
}

type materializedBufferKind int

const (
	materializedBufferKindCPU materializedBufferKind = iota + 1
	materializedBufferKindGPU
)

type materializedBuffer struct {
	kind materializedBufferKind
	cpu  []byte
	gpu  *wgpu.Buffer
}

type persistentBindMap struct {
	bufMap   map[renderer.ResourceID]*bindMapBuffer
	imageMap map[renderer.ResourceID]bindMapImage
}

type bindMapBuffer struct {
	Buffer    materializedBuffer
	Label     string
	permanent bool
}

type bindMapImage struct {
	texture *wgpu.Texture
	view    *wgpu.TextureView
}

type bindMap struct {
	bufMap        mem.BinaryTreeMap[renderer.ResourceID, *bindMapBuffer]
	imageMap      mem.BinaryTreeMap[renderer.ResourceID, bindMapImage]
	pendingClears mem.BinaryTreeMap[renderer.ResourceID, struct{}]
}

type bufferProperties struct {
	size   uint64
	usages wgpu.BufferUsage
}

type resourcePool struct {
	bufs map[bufferProperties][]*wgpu.Buffer
}

type transientBindMap struct {
	bufs   mem.BinaryTreeMap[renderer.ResourceID, transientBuf]
	images mem.BinaryTreeMap[renderer.ResourceID, *wgpu.TextureView]
}

type transientBufKind int

const (
	transientBufKindCPU transientBufKind = iota + 1
	transientBufKindGPU
)

type transientBuf struct {
	kind transientBufKind
	cpu  []byte
	gpu  *wgpu.Buffer
}

func New(dev *wgpu.Device, options *RendererOptions) *Engine {
	eng := &Engine{
		Device:   dev,
		renderer: renderer.New(),
		pool: resourcePool{
			bufs: make(map[bufferProperties][]*wgpu.Buffer),
		},
		downloads: make(map[renderer.ResourceID]*wgpu.Buffer),
		UseCPU:    options.UseCPU,

		resolver: renderer.NewResolver(),
		resident: persistentBindMap{
			bufMap:   make(map[renderer.ResourceID]*bindMapBuffer),
			imageMap: make(map[renderer.ResourceID]bindMapImage),
		},
	}
	eng.fullShaders = eng.prepareShaders()
	eng.buildShaders()
	// XXX support surfaceless engine use
	eng.blit = newBlitPipeline(eng.Device, options.SurfaceFormat)
	return eng
}

func (eng *Engine) buildShaders() {
	var wg sync.WaitGroup
	wg.Add(len(eng.shadersToInitialize))
	for _, s := range eng.shadersToInitialize {
		go func() {
			sh := eng.createComputePipeline(s.Label, s.Wgsl, s.Entries)
			eng.shaders[s.ShaderID].WGPU = &sh
			wg.Done()
		}()
	}
	wg.Wait()
}

func (eng *Engine) prepareShader(
	label string,
	wgsl []byte,
	layout []renderer.BindType,
	cpuShader cpuShader,
) renderer.ShaderID {
	if eng.UseCPU {
		switch cpuShader.kind {
		case cpuShaderKindPresent:
			id := renderer.ShaderID(len(eng.shaders))
			eng.shaders = append(eng.shaders, shader{Label: label, CPU: &cpuShader})
			return id
		case cpuShaderKindSkipped:
			// This shader is unused in CPU mode, create a dummy shader
			id := renderer.ShaderID(len(eng.shaders))
			eng.shaders = append(eng.shaders, shader{Label: label})
			return id
		case cpuShaderKindMissing:
			// Create a GPU shader as we don't have a CPU shader
		default:
			panic(fmt.Sprintf("internal error: unhandled kind %v", cpuShader.kind))
		}
	}

	entries := make([]wgpu.BindGroupLayoutEntry, len(layout))
	for i, bindType := range layout {
		switch bindType.Type {
		case renderer.BindTypeBuffer, renderer.BindTypeBufReadOnly:
			var typ wgpu.BufferBindingType
			if bindType.Type == renderer.BindTypeBuffer {
				typ = wgpu.BufferBindingTypeStorage
			} else {
				typ = wgpu.BufferBindingTypeReadOnlyStorage
			}
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				Buffer: &wgpu.BufferBindingLayout{
					Type:             typ,
					HasDynamicOffset: false,
					MinBindingSize:   0, // XXX 0 or Undefined?
				},
			}
		case renderer.BindTypeUniform:
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				Buffer: &wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeUniform,
					HasDynamicOffset: false,
					MinBindingSize:   0, // XXX 0 or Undefined?
				},
			}

		case renderer.BindTypeImage:
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				StorageTexture: &wgpu.StorageTextureBindingLayout{
					Access:        wgpu.StorageTextureAccessWriteOnly,
					Format:        imageFormatToWGPU(bindType.ImageFormat),
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			}

		case renderer.BindTypeImageRead:
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				Texture: &wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
					Multisampled:  false,
				},
			}

		case renderer.BindTypeImageArrayRead:
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				Texture: &wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
					Multisampled:  false,
				},
				Count: 2048,
			}

		default:
			panic(fmt.Sprintf("invalid bind type %d", bindType.Type))
		}
	}

	id := renderer.ShaderID(len(eng.shaders))
	eng.shaders = append(eng.shaders, shader{Label: label})
	eng.shadersToInitialize = append(eng.shadersToInitialize, uninitializedShader{
		Wgsl:     wgsl,
		Label:    label,
		Entries:  entries,
		ShaderID: id,
	})
	return id
}

func imageData(img image.Image) []byte {
	switch img := img.(type) {
	case *image.NRGBA:
		// not premultiplied

		if img.Stride != 4*img.Rect.Dx() {
			// XXX
			panic("subimages are not supported")
		}

		// XXX we need to premultiply the data
		return img.Pix
	case *image.RGBA:
		// premultiplied
		if img.Stride != 4*img.Rect.Dx() {
			// XXX
			panic("subimages are not supported")
		}
		return img.Pix
	default:
		// XXX convert to a supported format
		panic(fmt.Sprintf("unsupported image type %T", img))
	}
}

func (eng *Engine) RunRecording(
	arena *mem.Arena,
	queue *wgpu.Queue,
	recording renderer.Recording,
	externalResources []ExternalResource,
	label string,
	pgroup *ProfilerGroup,
) {
	pgroup = pgroup.Nest("RunRecording")
	defer pgroup.End()

	var freeBufs, freeImages mem.BinaryTreeMap[renderer.ResourceID, struct{}]
	transientMap := newTransientBindMap(arena, externalResources)
	// Note that Vello reuses a single bind map for all frames, with the premise
	// that some buffers will be reused across frames. Right now, however, no
	// buffers seem to be reused. Once we do reuse buffers, we'll want to use a
	// persistent bind map, too. But because most buffers aren't reused, it'll
	// be cheaper to first track buffers locally, then remember only those
	// buffers that weren't freed by the end of the frame.
	bindMap := bindMap{}

	for k, v := range eng.resident.bufMap {
		bindMap.bufMap.Insert(arena, k, v)
	}
	for k, v := range eng.resident.imageMap {
		bindMap.imageMap.Insert(arena, k, v)
	}

	encoder := eng.Device.CreateCommandEncoder(mem.Make(arena, wgpu.CommandEncoderDescriptor{Label: label}))

	for _, cmd := range recording.Commands {
		switch cmd := cmd.(type) {
		case *renderer.Upload:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.bufs.Insert(arena, bufProxy.ID, transientBuf{kind: transientBufKindCPU, cpu: bytes})
			usage := wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst | wgpu.BufferUsageStorage
			buf := eng.pool.getBuf(bufProxy.Size, bufProxy.Name, usage, eng.Device)
			queue.WriteBuffer(buf, 0, bytes)
			bindMap.insertBuf(arena, bufProxy, buf)

		case *renderer.UploadUniform:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.bufs.Insert(arena, bufProxy.ID, transientBuf{kind: transientBufKindCPU, cpu: bytes})
			usage := wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst
			buf := eng.pool.getBuf(bufProxy.Size, bufProxy.Name, usage, eng.Device)
			queue.WriteBuffer(buf, 0, bytes)
			bindMap.insertBuf(arena, bufProxy, buf)

		case *renderer.UploadImage:
			imageProxy := cmd.Proxy
			format := imageFormatToWGPU(imageProxy.Format)
			blockSize, ok := format.BlockCopySize(wgpu.TextureAspectAll)
			if !ok {
				panic("image format must have a valid block size")
			}
			texture := eng.Device.CreateTexture(mem.Make(arena, wgpu.TextureDescriptor{
				Size: wgpu.Extent3D{
					Width:              imageProxy.Width,
					Height:             imageProxy.Height,
					DepthOrArrayLayers: 1,
				},
				MipLevelCount: 1,
				SampleCount:   1,
				Dimension:     wgpu.TextureDimension2D,
				Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
				Format:        format,
			}))
			textureView := texture.CreateView(mem.Make(arena, wgpu.TextureViewDescriptor{
				Dimension:       wgpu.TextureViewDimension2D,
				Aspect:          wgpu.TextureAspectAll,
				MipLevelCount:   ^uint32(0),
				ArrayLayerCount: ^uint32(0),
				BaseMipLevel:    0,
				BaseArrayLayer:  0,
				Format:          format,
			}))
			queue.WriteTexture(
				mem.Make(arena, wgpu.ImageCopyTexture{
					Texture:  texture,
					MipLevel: 0,
					Origin:   wgpu.Origin3D{X: 0, Y: 0, Z: 0},
					Aspect:   wgpu.TextureAspectAll,
				}),
				imageData(cmd.Image),
				mem.Make(arena, wgpu.TextureDataLayout{
					Offset: 0,
					// XXX can we use this to upload subimages?
					BytesPerRow:  imageProxy.Width * blockSize,
					RowsPerImage: ^uint32(0),
				}),
				mem.Make(arena, wgpu.Extent3D{
					Width:              imageProxy.Width,
					Height:             imageProxy.Height,
					DepthOrArrayLayers: 1,
				}),
			)
			bindMap.insertImage(arena, imageProxy.ID, texture, textureView)

		case *renderer.WriteImage:
			proxy := cmd.Proxy
			x := cmd.Coords[0]
			y := cmd.Coords[1]
			width := cmd.Coords[2]
			height := cmd.Coords[3]
			texture, _ := bindMap.getOrCreateImage(arena, proxy, eng.Device)
			format := imageFormatToWGPU(proxy.Format)
			blockSize, ok := format.BlockCopySize(wgpu.TextureAspectAll)
			if !ok {
				panic("image format must have a valid block size")
			}
			queue.WriteTexture(
				mem.Make(arena, wgpu.ImageCopyTexture{
					Texture:  texture,
					MipLevel: 0,
					Origin:   wgpu.Origin3D{X: x, Y: y, Z: 0},
					Aspect:   wgpu.TextureAspectAll,
				}),
				imageData(cmd.Image),
				mem.Make(arena, wgpu.TextureDataLayout{
					Offset:       0,
					BytesPerRow:  width * blockSize,
					RowsPerImage: 0, // XXX 0 or Undefined?
				}),
				mem.Make(arena, wgpu.Extent3D{
					Width:              width,
					Height:             height,
					DepthOrArrayLayers: 1,
				}),
			)

		case *renderer.Dispatch:
			shaderID := cmd.Shader
			wgSize := cmd.WorkgroupSize
			bindings := cmd.Bindings
			shader := eng.shaders[shaderID]
			switch s := shader.Select().(type) {
			case *cpuShader:
				// The current strategy is to run the CPU shader synchronously. This
				// works because there is currently the added constraint that data
				// can only flow from CPU to GPU, not the other way around. If and
				// when we implement that, we will need to defer the execution. Of
				// course, we will also need to wire up more async synchronization
				// mechanisms, as the CPU dispatch can't run until the preceding
				// command buffer submission completes (and, in WebGPU, the async
				// mapping operations on the buffers completes).
				resources := transientMap.createCPUResources(arena, &bindMap, bindings)
				s.shader(arena, wgSize[0], resources)
			case *wgpuShader:
				bindGroup := transientMap.createBindGroup(
					arena,
					&bindMap,
					&eng.pool,
					eng.Device,
					queue,
					encoder,
					s.bindGroupLayout,
					bindings,
				)

				cpass := encoder.BeginComputePass(mem.Make(arena, wgpu.ComputePassDescriptor{
					Label:           shader.Label,
					TimestampWrites: pgroup.Compute(arena, shader.Label),
				}))

				cpass.SetPipeline(s.pipeline)
				cpass.SetBindGroup(0, bindGroup, nil)
				cpass.DispatchWorkgroups(wgSize[0], wgSize[1], wgSize[2])
				cpass.End()
				bindGroup.Release()
				cpass.Release()
			default:
				panic(fmt.Sprintf("unhandled type %T", s))
			}

		case *renderer.DispatchIndirect:
			shaderID := cmd.Shader
			proxy := cmd.Buffer
			offset := cmd.Offset
			bindings := cmd.Bindings
			shader := eng.shaders[shaderID]
			switch s := shader.Select().(type) {
			case *cpuShader:
				// Same consideration as above about running the CPU shader synchronously.
				var n_wg uint32
				b := bindMap.getCPUBuf(proxy.ID).(cpu.CPUBuffer)
				indirect := safeish.SliceCast[[]uint32](b)
				n_wg = indirect[0]
				resources := transientMap.createCPUResources(arena, &bindMap, bindings)
				s.shader(arena, n_wg, resources)
			case *wgpuShader:
				bindGroup := transientMap.createBindGroup(
					arena,
					&bindMap,
					&eng.pool,
					eng.Device,
					queue,
					encoder,
					s.bindGroupLayout,
					bindings,
				)

				transientMap.materializeGPUBufForIndirect(
					&bindMap,
					&eng.pool,
					eng.Device,
					queue,
					proxy,
				)

				cpass := encoder.BeginComputePass(mem.Make(arena, wgpu.ComputePassDescriptor{
					Label:           s.label,
					TimestampWrites: pgroup.Compute(arena, shader.Label),
				}))

				cpass.SetPipeline(s.pipeline)
				cpass.SetBindGroup(0, bindGroup, nil)
				buf, ok := bindMap.getGPUBuf(proxy.ID)
				if !ok {
					panic("tried using unavailable buffer for indirect dispatch")
				}
				cpass.DispatchWorkgroupsIndirect(buf, offset)
				cpass.End()
				bindGroup.Release()
				cpass.Release()
			default:
				panic(fmt.Sprintf("unhandled type %T", s))
			}

		case *renderer.Download:
			proxy := cmd.Buffer
			srcBuf, ok := bindMap.getGPUBuf(proxy.ID)
			if !ok {
				panic("tried using unavailable buffer for download")
			}
			usage := wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst
			buf := eng.pool.getBuf(proxy.Size, "download", usage, eng.Device)
			encoder.CopyBufferToBuffer(srcBuf, 0, buf, 0, proxy.Size)
			eng.downloads[proxy.ID] = buf

		case *renderer.Clear:
			proxy := cmd.Buffer
			offset := cmd.Offset
			size := cmd.Size
			if buf, ok := bindMap.getBuf(proxy); ok {
				b := &buf.Buffer
				switch b.kind {
				case materializedBufferKindGPU:
					encoder.ClearBuffer(b.gpu, offset, uint64(size))
				case materializedBufferKindCPU:
					slice := b.cpu[offset:]
					if size >= 0 {
						slice = slice[:size]
					}
					clear(slice)
				default:
					panic(fmt.Sprintf("unhandled type %T", b))
				}
			} else {
				bindMap.pendingClears.Insert(arena, proxy.ID, struct{}{})
			}

		case *renderer.FreeBuffer:
			freeBufs.Insert(arena, cmd.Buffer.ID, struct{}{})

		case *renderer.FreeImage:
			freeImages.Insert(arena, cmd.Image.ID, struct{}{})

		default:
			panic(fmt.Sprintf("unhandled command %T", cmd))
		}
	}

	cmd := encoder.Finish(nil)
	encoder.Release()
	queue.Submit(mem.Varargs(arena, cmd)...)
	cmd.Release()

	for id := range freeBufs.Keys() {
		buf, ok := bindMap.bufMap.Get(id)
		if ok {
			bindMap.bufMap.Delete(id)
			delete(eng.resident.bufMap, id)
			if buf.Buffer.kind == materializedBufferKindGPU {
				gpuBuf := buf.Buffer.gpu
				props := bufferProperties{
					size:   gpuBuf.Size(),
					usages: gpuBuf.Usage(),
				}
				// TODO(dh): add a method to ResourcePool to return buffers
				eng.pool.bufs[props] = append(eng.pool.bufs[props], gpuBuf)
			}
		}
	}
	for id := range freeImages.Keys() {
		tex, ok := bindMap.imageMap.Get(id)
		if ok {
			bindMap.imageMap.Delete(id)
			delete(eng.resident.imageMap, id)
			// TODO: have a pool to avoid needless re-allocation
			tex.texture.Release()
			tex.view.Release()
		}
	}

	// TODO(dh): should we make persistence explicit, e.g. as a new command, or
	// a flag on resources? and if we do, should we complain about resources
	// that aren't persistent but weren't freed?
	for k, v := range bindMap.bufMap.All() {
		if v.permanent {
			continue
		}
		v.permanent = true
		eng.resident.bufMap[k] = v
	}
	for k, v := range bindMap.imageMap.All() {
		eng.resident.imageMap[k] = v
	}
}

func (eng *Engine) getDownload(buf renderer.BufferProxy) (*wgpu.Buffer, bool) {
	got, ok := eng.downloads[buf.ID]
	return got, ok
}

func (eng *Engine) freeDownload(buf renderer.BufferProxy) {
	delete(eng.downloads, buf.ID)
}

func (eng *Engine) createComputePipeline(
	label string,
	wgsl []byte,
	entries []wgpu.BindGroupLayoutEntry,
) wgpuShader {
	shaderModule := eng.Device.CreateShaderModule(wgpu.ShaderModuleDescriptor{
		Label:  label,
		Source: wgpu.ShaderSourceWGSL(wgsl),
	})
	bindGroupLayout := eng.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: entries,
	})
	computePipelineLayout := eng.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	pipeline := eng.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  label,
		Layout: computePipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shaderModule,
			EntryPoint: "main",
			// XXX compilation_options
		},
	})
	computePipelineLayout.Release()

	return wgpuShader{
		label:           label,
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
	}
}

func (m *bindMap) insertBuf(arena *mem.Arena, proxy renderer.BufferProxy, buffer *wgpu.Buffer) {
	m.bufMap.Insert(arena, proxy.ID, mem.Make(arena, bindMapBuffer{
		Buffer: materializedBuffer{kind: materializedBufferKindGPU, gpu: buffer},
		Label:  proxy.Name,
	}))
}

func (m *bindMap) getGPUBuf(id renderer.ResourceID) (*wgpu.Buffer, bool) {
	mbuf, ok := m.bufMap.Get(id)
	if !ok {
		return nil, false
	}
	if mbuf.Buffer.kind != materializedBufferKindGPU {
		return nil, false
	}
	buf := mbuf.Buffer.gpu
	return buf, true
}

func (m *bindMap) getCPUBuf(id renderer.ResourceID) cpu.CPUBinding {
	b, ok := m.bufMap.Get(id)
	if !ok || b.Buffer.kind != materializedBufferKindCPU {
		panic("getting CPU buffer, but it's on GPU")
	}
	return cpu.CPUBuffer(b.Buffer.cpu)
}

func (m *bindMap) materializeCPUBuf(arena *mem.Arena, proxy renderer.BufferProxy) {
	if _, ok := m.bufMap.Get(proxy.ID); !ok {
		// OPT(dh): can/should we use an arena here?
		buffer := make([]byte, proxy.Size)
		m.bufMap.Insert(arena, proxy.ID, mem.Make(arena, bindMapBuffer{
			Buffer: materializedBuffer{kind: materializedBufferKindCPU, cpu: buffer},
			Label:  proxy.Name,
		}))
	}
}

func (m *bindMap) insertImage(arena *mem.Arena, id renderer.ResourceID, image *wgpu.Texture, imageView *wgpu.TextureView) {
	m.imageMap.Insert(arena, id, bindMapImage{image, imageView})
}

func (m *bindMap) getBuf(proxy renderer.BufferProxy) (*bindMapBuffer, bool) {
	b, ok := m.bufMap.Get(proxy.ID)
	return b, ok
}

func (m *bindMap) getOrCreateImage(
	arena *mem.Arena,
	proxy renderer.ImageProxy,
	dev *wgpu.Device,
) (*wgpu.Texture, *wgpu.TextureView) {
	if entry, ok := m.imageMap.Get(proxy.ID); ok {
		return entry.texture, entry.view
	}

	format := imageFormatToWGPU(proxy.Format)
	texture := dev.CreateTexture(&wgpu.TextureDescriptor{
		Size: wgpu.Extent3D{
			Width:              proxy.Width,
			Height:             proxy.Height,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
		Format:        format,
	})
	textureView := texture.CreateView(&wgpu.TextureViewDescriptor{
		Dimension:       wgpu.TextureViewDimension2D,
		Aspect:          wgpu.TextureAspectAll,
		MipLevelCount:   ^uint32(0),
		BaseMipLevel:    0,
		BaseArrayLayer:  0,
		ArrayLayerCount: ^uint32(0),
		Format:          imageFormatToWGPU(proxy.Format),
	})
	m.imageMap.Insert(arena, proxy.ID, bindMapImage{
		texture, textureView,
	})

	return texture, textureView
}

func (pool *resourcePool) getBuf(
	size uint64,
	name string,
	usage wgpu.BufferUsage,
	dev *wgpu.Device,
) *wgpu.Buffer {
	const sizeClassBits = 1

	roundedSize := poolSizeClass(size, sizeClassBits)
	props := bufferProperties{
		size:   roundedSize,
		usages: usage,
	}
	if bufVec, ok := pool.bufs[props]; ok {
		if len(bufVec) > 0 {
			buf := bufVec[len(bufVec)-1]
			bufVec = bufVec[:len(bufVec)-1]
			pool.bufs[props] = bufVec
			return buf
		}
	}
	return dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: name,
		Size:  roundedSize,
		Usage: usage,
	})
}

func poolSizeClass(x uint64, numBits uint32) uint64 {
	if x > 1<<numBits {
		a := bits.LeadingZeros64(x - 1)
		b := (x - 1) | (((math.MaxUint64 / 2) >> numBits) >> a)
		return b + 1
	} else {
		return 1 << numBits
	}
}

func (b *bindMapBuffer) uploadIfNeeded(
	proxy renderer.BufferProxy,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	pool *resourcePool,
) {
	if b.Buffer.kind != materializedBufferKindCPU {
		return
	}
	cpuBuf := b.Buffer.cpu
	usage := wgpu.BufferUsageCopySrc |
		wgpu.BufferUsageCopyDst |
		wgpu.BufferUsageStorage |
		wgpu.BufferUsageIndirect
	buf := pool.getBuf(proxy.Size, proxy.Name, usage, dev)
	queue.WriteBuffer(buf, 0, cpuBuf)
	b.Buffer = materializedBuffer{kind: materializedBufferKindGPU, gpu: buf}
}

func newTransientBindMap(arena *mem.Arena, externalResources []ExternalResource) transientBindMap {
	bufs := mem.BinaryTreeMap[renderer.ResourceID, transientBuf]{}
	images := mem.BinaryTreeMap[renderer.ResourceID, *wgpu.TextureView]{}
	for _, res := range externalResources {
		switch res := res.(type) {
		case ExternalBuffer:
			bufs.Insert(arena, res.Proxy.ID, transientBuf{kind: transientBufKindGPU, gpu: res.Buffer})
		case ExternalImage:
			images.Insert(arena, res.Proxy.ID, res.View)
		}
	}
	return transientBindMap{
		bufs:   bufs,
		images: images,
	}
}

func (m *transientBindMap) materializeGPUBufForIndirect(
	bindMap *bindMap,
	pool *resourcePool,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	buf renderer.BufferProxy,
) {
	if _, ok := m.bufs.Get(buf.ID); ok {
		return
	}
	if b, ok := bindMap.bufMap.Get(buf.ID); ok {
		b.uploadIfNeeded(buf, dev, queue, pool)
	}
}

func (m *transientBindMap) createBindGroup(
	arena *mem.Arena,
	bindMap *bindMap,
	pool *resourcePool,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	encoder *wgpu.CommandEncoder,
	layout *wgpu.BindGroupLayout,
	bindings []renderer.ResourceProxy,
) *wgpu.BindGroup {

	doImage := func(proxy renderer.ImageProxy) {
		if _, ok := m.images.Get(proxy.ID); ok {
			return
		}
		if _, ok := bindMap.imageMap.Get(proxy.ID); ok {
			return
		}
		format := imageFormatToWGPU(proxy.Format)
		texture := dev.CreateTexture(&wgpu.TextureDescriptor{
			Size: wgpu.Extent3D{
				Width:              proxy.Width,
				Height:             proxy.Height,
				DepthOrArrayLayers: 1,
			},
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     wgpu.TextureDimension2D,
			Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
			Format:        format,
		})
		textureView := texture.CreateView(&wgpu.TextureViewDescriptor{
			Dimension:       wgpu.TextureViewDimension2D,
			Aspect:          wgpu.TextureAspectAll,
			MipLevelCount:   ^uint32(0),
			BaseMipLevel:    0,
			BaseArrayLayer:  0,
			ArrayLayerCount: ^uint32(0),
			Format:          imageFormatToWGPU(proxy.Format),
		})
		bindMap.imageMap.Insert(arena, proxy.ID, bindMapImage{
			texture, textureView,
		})
	}

	for _, proxy := range bindings {
		switch proxy.Kind {
		case renderer.ResourceProxyKindBuffer:
			if _, ok := m.bufs.Get(proxy.BufferProxy.ID); ok {
				continue
			}
			if o, ok := bindMap.bufMap.Get(proxy.BufferProxy.ID); ok {
				o.uploadIfNeeded(proxy.BufferProxy, dev, queue, pool)
			} else {
				// TODO: only some buffers will need indirect, but does it hurt?
				usage := wgpu.BufferUsageCopySrc |
					wgpu.BufferUsageCopyDst |
					wgpu.BufferUsageStorage |
					wgpu.BufferUsageIndirect
				buf := pool.getBuf(proxy.Size, proxy.Name, usage, dev)
				if _, ok := bindMap.pendingClears.Get(proxy.BufferProxy.ID); ok {
					bindMap.pendingClears.Delete(proxy.BufferProxy.ID)
					encoder.ClearBuffer(buf, 0, buf.Size())
				}
				bindMap.bufMap.Insert(arena, proxy.BufferProxy.ID, mem.Make(arena, bindMapBuffer{
					Buffer: materializedBuffer{kind: materializedBufferKindGPU, gpu: buf},
					Label:  proxy.Name,
				}))
			}
		case renderer.ResourceProxyKindImage:
			doImage(proxy.ImageProxy)
		case renderer.ResourceProxyKindImageArray:
			for _, proxy := range proxy.ImageArray {
				doImage(proxy)
			}
		default:
			panic(fmt.Sprintf("unhandled type %d", proxy.Kind))
		}
	}

	entries := mem.NewSlice[[]wgpu.BindGroupEntry](arena, len(bindings), len(bindings))
	for i, proxy := range bindings {
		switch proxy.Kind {
		case renderer.ResourceProxyKindBuffer:
			var buf *wgpu.Buffer
			b, _ := m.bufs.Get(proxy.BufferProxy.ID)
			switch b.kind {
			case transientBufKindGPU:
				buf = b.gpu
			default:
				var ok bool
				buf, ok = bindMap.getGPUBuf(proxy.BufferProxy.ID)
				if !ok {
					panic(fmt.Sprintf("unexpected ok == false for %s(%d)", proxy.Name, proxy.BufferProxy.ID))
				}
			}
			entries[i] = wgpu.BindGroupEntry{
				Binding: uint32(i),
				Buffer:  buf,
				Size:    ^uint64(0),
			}
		case renderer.ResourceProxyKindImage:
			view, ok := m.images.Get(proxy.ImageProxy.ID)
			if !ok {
				img, ok := bindMap.imageMap.Get(proxy.ImageProxy.ID)
				if !ok {
					panic("unexpected ok == false")
				}
				view = img.view
			}
			entries[i] = wgpu.BindGroupEntry{
				Binding:     uint32(i),
				TextureView: view,
				Size:        ^uint64(0),
			}
		case renderer.ResourceProxyKindImageArray:
			n := len(proxy.ImageArray)
			views := mem.NewSlice[[]*wgpu.TextureView](arena, n, n)
			for j, imgProxy := range proxy.ImageArray {
				view, ok := m.images.Get(imgProxy.ID)
				if !ok {
					img, ok := bindMap.imageMap.Get(imgProxy.ID)
					if !ok {
						panic("unexpected ok == false")
					}
					view = img.view
				}
				views[j] = view
			}
			entries[i] = wgpu.BindGroupEntry{
				Binding:      uint32(i),
				Size:         ^uint64(0),
				TextureViews: views,
			}
		default:
			panic(fmt.Sprintf("unhandled type %d", proxy.Kind))
		}
	}

	return dev.CreateBindGroup(mem.Make(arena, wgpu.BindGroupDescriptor{
		Layout:  layout,
		Entries: entries,
	}))
}

func (m *transientBindMap) createCPUResources(
	arena *mem.Arena,
	bindMap *bindMap,
	bindings []renderer.ResourceProxy,
) []cpu.CPUBinding {
	for _, resource := range bindings {
		switch resource.Kind {
		case renderer.ResourceProxyKindBuffer:
			tbuf, _ := m.bufs.Get(resource.BufferProxy.ID)
			switch tbuf.kind {
			case transientBufKindCPU:
			case transientBufKindGPU:
				panic("buffer was already materialized on GPU")
			case 0:
				bindMap.materializeCPUBuf(arena, resource.BufferProxy)
			default:
				panic(fmt.Sprintf("unhandled type %T", tbuf))
			}
		case renderer.ResourceProxyKindImage:
			panic("not implemented")
		default:
			panic(fmt.Sprintf("unhandled type %T", resource))
		}
	}

	out := make([]cpu.CPUBinding, len(bindings))
	for i, resource := range bindings {
		switch resource.Kind {
		case renderer.ResourceProxyKindBuffer:
			tbuf, _ := m.bufs.Get(resource.BufferProxy.ID)
			switch tbuf.kind {
			case transientBufKindCPU:
				out[i] = cpu.CPUBuffer(tbuf.cpu)
			default:
				out[i] = bindMap.getCPUBuf(resource.BufferProxy.ID)
			}
		case renderer.ResourceProxyKindImage:
			panic("not implemented")
		default:
			panic(fmt.Sprintf("unhandled type %T", resource))
		}
	}
	return out
}
