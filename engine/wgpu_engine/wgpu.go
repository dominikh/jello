package wgpu_engine

// OPT reuse bind groups

import (
	"fmt"
	"math"
	"math/bits"

	"honnef.co/go/jello/mem"
	"honnef.co/go/jello/renderer"
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
	shaders             []shader
	pool                resourcePool
	downloads           map[renderer.ResourceID]*wgpu.Buffer
	shadersToInitialize []uninitializedShader
	UseCPU              bool

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

type cpuShader struct {
	shader func(uint32, []cpuBinding)
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

type materializedBuffer interface {
	// One of wgpu.Buffer and []byte
}

type bindMapBuffer struct {
	Buffer materializedBuffer
	Label  string
}

type bindMapImage struct {
	texture *wgpu.Texture
	view    *wgpu.TextureView
}

type bindMap struct {
	bufMap        mem.BinaryTreeMap[renderer.ResourceID, *bindMapBuffer]
	imageMap      mem.BinaryTreeMap[renderer.ResourceID, *bindMapImage]
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
	transientBufKindBytes transientBufKind = iota + 1
	transientBufKindBuffer
)

type transientBuf struct {
	kind   transientBufKind
	bytes  []byte
	buffer *wgpu.Buffer
}

func New(dev *wgpu.Device, options *RendererOptions) *Engine {
	eng := &Engine{
		Device: dev,
		pool: resourcePool{
			bufs: make(map[bufferProperties][]*wgpu.Buffer),
		},
		downloads: make(map[renderer.ResourceID]*wgpu.Buffer),
		UseCPU:    options.UseCPU,

		resolver: renderer.NewResolver(),
	}
	eng.fullShaders = eng.newFullShaders()
	eng.buildShadersIfNeeded(1)
	// XXX support surfaceless engine use
	eng.blit = newBlitPipeline(eng.Device, options.SurfaceFormat)
	return eng
}

func (eng *Engine) UseParallelInitialization() {
	if eng.shadersToInitialize != nil {
		return
	}
	eng.shadersToInitialize = []uninitializedShader{}
}

func (eng *Engine) buildShadersIfNeeded(numThreads int) {
	if eng.shadersToInitialize == nil {
		return
	}
	newShaders := eng.shadersToInitialize
	// XXX implement parallelism
	for _, s := range newShaders {
		sh := eng.createComputePipeline(s.Label, s.Wgsl, s.Entries)
		if int(s.ShaderID) >= len(eng.shaders) {
			if cap(eng.shaders) <= int(s.ShaderID) {
				c := make([]shader, s.ShaderID+1)
				copy(c, eng.shaders)
				eng.shaders = c
			} else {
				eng.shaders = eng.shaders[:s.ShaderID+1]
			}
		}
		eng.shaders[s.ShaderID] = shader{WGPU: &sh}
	}
}

type cpuShaderType interface {
	// XXX implement
}

func (eng *Engine) addShader(
	label string,
	wgsl []byte,
	layout []renderer.BindType,
	cpuShader cpuShaderType,
) renderer.ShaderID {
	add := func(shader shader) renderer.ShaderID {
		id := len(eng.shaders)
		eng.shaders = append(eng.shaders, shader)
		return renderer.ShaderID(id)
	}

	if eng.UseCPU {
		panic("XXX unimplemented")
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

		default:
			panic(fmt.Sprintf("invalid bind type %d", bindType.Type))
		}
	}

	if eng.shadersToInitialize != nil {
		id := add(shader{Label: label})
		eng.shadersToInitialize = append(eng.shadersToInitialize, uninitializedShader{
			Wgsl:     wgsl,
			Label:    label,
			Entries:  entries,
			ShaderID: id,
		})
		return id
	}

	wgpu := eng.createComputePipeline(label, wgsl, entries)
	return add(shader{
		Label: label,
		WGPU:  &wgpu,
	})
}

func (eng *Engine) RunRecording(
	arena *mem.Arena,
	queue *wgpu.Queue,
	recording *renderer.Recording,
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

	// XXX why do we have a persistent bind map if we clear it at the end of the
	// frame, anyway? Vello made that change in
	// e47c5777ccc84b378145d0486d2b1a9b5c737fa0, apparently planning to persist
	// buffers across recordings in the future.

	encoder := eng.Device.CreateCommandEncoder(mem.Make(arena, wgpu.CommandEncoderDescriptor{Label: label}))

	for _, cmd := range recording.Commands {
		switch cmd := cmd.(type) {
		case *renderer.Upload:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.bufs.Insert(arena, bufProxy.ID, transientBuf{kind: transientBufKindBytes, bytes: bytes})
			usage := wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst | wgpu.BufferUsageStorage
			buf := eng.pool.getBuf(bufProxy.Size, bufProxy.Name, usage, eng.Device)
			queue.WriteBuffer(buf, 0, bytes)
			bindMap.insertBuf(arena, bufProxy, buf)

		case *renderer.UploadUniform:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.bufs.Insert(arena, bufProxy.ID, transientBuf{kind: transientBufKindBytes, bytes: bytes})
			usage := wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst
			// XXXXXX "config" buffer is created here
			buf := eng.pool.getBuf(bufProxy.Size, bufProxy.Name, usage, eng.Device)
			queue.WriteBuffer(buf, 0, bytes)
			bindMap.insertBuf(arena, bufProxy, buf)

		case *renderer.UploadImage:
			imageProxy := cmd.Image
			bytes := cmd.Data
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
				bytes,
				mem.Make(arena, wgpu.TextureDataLayout{
					Offset:       0,
					BytesPerRow:  imageProxy.Width * blockSize,
					RowsPerImage: ^uint32(0), // XXX 0 or Undefined?
				}),
				mem.Make(arena, wgpu.Extent3D{
					Width:              imageProxy.Width,
					Height:             imageProxy.Height,
					DepthOrArrayLayers: 1,
				}),
			)
			bindMap.insertImage(arena, imageProxy.ID, texture, textureView)

		case *renderer.WriteImage:
			proxy := cmd.Image
			x := cmd.Coords[0]
			y := cmd.Coords[1]
			width := cmd.Coords[2]
			height := cmd.Coords[3]
			data := cmd.Data
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
				data,
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
				panic("XXX no support for CPU shaders")
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
				panic("XXX no support for CPU shaders")
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
				switch b := buf.Buffer.(type) {
				case *wgpu.Buffer:
					encoder.ClearBuffer(b, offset, uint64(size))
				case []byte:
					slice := b[offset:]
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
	queue.Submit(cmd)
	cmd.Release()

	for id := range freeBufs.Keys() {
		buf, ok := bindMap.bufMap.Get(id)
		if ok {
			bindMap.bufMap.Delete(id)
			if gpuBuf, ok := buf.Buffer.(*wgpu.Buffer); ok {
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
			// TODO: have a pool to avoid needless re-allocation
			tex.texture.Release()
			tex.view.Release()
		}
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
	// OPT(dh): use SPIR-V instead of WGSL for faster engine creation.
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
	m.bufMap.Insert(arena, proxy.ID, &bindMapBuffer{
		Buffer: buffer,
		Label:  proxy.Name,
	})
}

func (m *bindMap) getGPUBuf(id renderer.ResourceID) (*wgpu.Buffer, bool) {
	mbuf, ok := m.bufMap.Get(id)
	if !ok {
		return nil, false
	}
	buf, ok := mbuf.Buffer.(*wgpu.Buffer)
	return buf, ok
}

func (m *bindMap) getCPUBuf(id renderer.ResourceID) cpuBinding {
	b, ok := m.bufMap.Get(id)
	buf, ok := b.Buffer.([]byte)
	if !ok {
		panic("getting CPU buffer, but it's on GPU")
	}
	return cpuBufferRW(buf)
}

func (m *bindMap) materializeCPUBuf(arena *mem.Arena, proxy renderer.BufferProxy) {
	if _, ok := m.bufMap.Get(proxy.ID); !ok {
		buffer := make([]byte, proxy.Size)
		m.bufMap.Insert(arena, proxy.ID, &bindMapBuffer{
			Buffer: buffer,
			Label:  proxy.Name,
		})
	}
}

func (m *bindMap) insertImage(arena *mem.Arena, id renderer.ResourceID, image *wgpu.Texture, imageView *wgpu.TextureView) {
	m.imageMap.Insert(arena, id, &bindMapImage{image, imageView})
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
	m.imageMap.Insert(arena, proxy.ID, &bindMapImage{
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
	cpuBuf, ok := b.Buffer.([]byte)
	if !ok {
		return
	}
	usage := wgpu.BufferUsageCopySrc |
		wgpu.BufferUsageCopyDst |
		wgpu.BufferUsageStorage |
		wgpu.BufferUsageIndirect
	buf := pool.getBuf(proxy.Size, proxy.Name, usage, dev)
	queue.WriteBuffer(buf, 0, cpuBuf)
	b.Buffer = buf
}

func newTransientBindMap(arena *mem.Arena, externalResources []ExternalResource) transientBindMap {
	bufs := mem.BinaryTreeMap[renderer.ResourceID, transientBuf]{}
	images := mem.BinaryTreeMap[renderer.ResourceID, *wgpu.TextureView]{}
	for _, res := range externalResources {
		switch res := res.(type) {
		case ExternalBuffer:
			bufs.Insert(arena, res.Proxy.ID, transientBuf{kind: transientBufKindBuffer, buffer: res.Buffer})
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
				bindMap.bufMap.Insert(arena, proxy.BufferProxy.ID, &bindMapBuffer{
					Buffer: buf,
					Label:  proxy.Name,
				})
			}
		case renderer.ResourceProxyKindImage:
			if _, ok := m.images.Get(proxy.ImageProxy.ID); ok {
				continue
			}
			if _, ok := bindMap.imageMap.Get(proxy.ImageProxy.ID); ok {
				continue
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
				// XXX this one needs storage binding, apparently?! this is line 887 in wgpu_engine.rs, and they don't set StorageBinding.
				Usage:  wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst | wgpu.TextureUsageStorageBinding,
				Format: format,
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
			bindMap.imageMap.Insert(arena, proxy.ImageProxy.ID, &bindMapImage{
				texture, textureView,
			})
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
			case transientBufKindBuffer:
				buf = b.buffer
			default:
				var ok bool
				buf, ok = bindMap.getGPUBuf(proxy.BufferProxy.ID)
				if !ok {
					panic("unexpected ok == false")
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
		default:
			panic(fmt.Sprintf("unhandled type %T", proxy))
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
) []cpuBinding {
	for _, resource := range bindings {
		switch resource.Kind {
		case renderer.ResourceProxyKindBuffer:
			tbuf, _ := m.bufs.Get(resource.BufferProxy.ID)
			switch tbuf.kind {
			case transientBufKindBytes:
			case transientBufKindBuffer:
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

	out := make([]cpuBinding, len(bindings))
	for i, resource := range bindings {
		switch resource.Kind {
		case renderer.ResourceProxyKindBuffer:
			tbuf, _ := m.bufs.Get(resource.BufferProxy.ID)
			switch tbuf.kind {
			case tbuf.kind:
				out[i] = cpuBuffer(tbuf.bytes)
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
