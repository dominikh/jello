package wgpu_engine

// OPT reuse bind groups

import (
	"fmt"
	"math"
	"math/bits"

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
	bindMap             bindMap
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
	bufMap        map[renderer.ResourceID]*bindMapBuffer
	imageMap      map[renderer.ResourceID]*bindMapImage
	pendingClears map[renderer.ResourceID]struct{}
}

type bufferProperties struct {
	size   uint64
	usages wgpu.BufferUsage
}

type resourcePool struct {
	bufs map[bufferProperties][]*wgpu.Buffer
}

type transientBindMap struct {
	bufs   map[renderer.ResourceID]transientBuf
	images map[renderer.ResourceID]*wgpu.TextureView
}

type transientBuf interface {
	// One of []byte and wgpu.Buffer
}

func New(dev *wgpu.Device, options *RendererOptions) *Engine {
	eng := &Engine{
		Device: dev,
		pool: resourcePool{
			bufs: make(map[bufferProperties][]*wgpu.Buffer),
		},
		bindMap: bindMap{
			bufMap:        make(map[renderer.ResourceID]*bindMapBuffer),
			imageMap:      make(map[renderer.ResourceID]*bindMapImage),
			pendingClears: make(map[renderer.ResourceID]struct{})},
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
	queue *wgpu.Queue,
	recording *renderer.Recording,
	externalResources []ExternalResource,
	label string,
	// TODO profiler
) {
	freeBufs := map[renderer.ResourceID]struct{}{}
	freeImages := map[renderer.ResourceID]struct{}{}
	transientMap := newTransientBindMap(externalResources)

	encoder := eng.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: label})
	defer encoder.Release()

	// TODO profiler
	// OPT release things after every command
	for _, cmd := range recording.Commands {
		switch cmd := cmd.(type) {
		case renderer.Upload:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.bufs[bufProxy.ID] = bytes
			usage := wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst | wgpu.BufferUsageStorage
			buf := eng.pool.getBuf(bufProxy.Size, bufProxy.Name, usage, eng.Device)
			queue.WriteBuffer(buf, 0, bytes)
			eng.bindMap.insertBuf(bufProxy, buf)

		case renderer.UploadUniform:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.bufs[bufProxy.ID] = bytes
			usage := wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst
			// XXXXXX "config" buffer is created here
			buf := eng.pool.getBuf(bufProxy.Size, bufProxy.Name, usage, eng.Device)
			queue.WriteBuffer(buf, 0, bytes)
			eng.bindMap.insertBuf(bufProxy, buf)

		case renderer.UploadImage:
			imageProxy := cmd.Image
			bytes := cmd.Data
			format := imageFormatToWGPU(imageProxy.Format)
			blockSize, ok := format.BlockCopySize(wgpu.TextureAspectAll)
			if !ok {
				panic("image format must have a valid block size")
			}
			texture := eng.Device.CreateTexture(&wgpu.TextureDescriptor{
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
			})
			textureView := texture.CreateView(&wgpu.TextureViewDescriptor{
				Dimension:      wgpu.TextureViewDimension2D,
				Aspect:         wgpu.TextureAspectAll,
				MipLevelCount:  ^uint32(0),
				BaseMipLevel:   0,
				BaseArrayLayer: 0,
				Format:         format,
			})
			queue.WriteTexture(
				&wgpu.ImageCopyTexture{
					Texture:  texture,
					MipLevel: 0,
					Origin:   wgpu.Origin3D{X: 0, Y: 0, Z: 0},
					Aspect:   wgpu.TextureAspectAll,
				},
				bytes,
				&wgpu.TextureDataLayout{
					Offset:       0,
					BytesPerRow:  imageProxy.Width * blockSize,
					RowsPerImage: 0, // XXX 0 or Undefined?
				},
				&wgpu.Extent3D{
					Width:              imageProxy.Width,
					Height:             imageProxy.Height,
					DepthOrArrayLayers: 1,
				},
			)
			eng.bindMap.insertImage(imageProxy.ID, texture, textureView)

		case renderer.WriteImage:
			proxy := cmd.Image
			x := cmd.Coords[0]
			y := cmd.Coords[1]
			width := cmd.Coords[2]
			height := cmd.Coords[3]
			data := cmd.Data
			texture, _ := eng.bindMap.getOrCreateImage(proxy, eng.Device)
			format := imageFormatToWGPU(proxy.Format)
			blockSize, ok := format.BlockCopySize(wgpu.TextureAspectAll)
			if !ok {
				panic("image format must have a valid block size")
			}
			queue.WriteTexture(
				&wgpu.ImageCopyTexture{
					Texture:  texture,
					MipLevel: 0,
					Origin:   wgpu.Origin3D{X: x, Y: y, Z: 0},
					Aspect:   wgpu.TextureAspectAll,
				},
				data,
				&wgpu.TextureDataLayout{
					Offset:       0,
					BytesPerRow:  width * blockSize,
					RowsPerImage: 0, // XXX 0 or Undefined?
				},
				&wgpu.Extent3D{
					Width:              width,
					Height:             height,
					DepthOrArrayLayers: 1,
				},
			)

		case renderer.Dispatch:
			shaderID := cmd.Shader
			wgSize := cmd.WorkgroupSize
			bindings := cmd.Bindings
			shader := eng.shaders[shaderID]
			switch s := shader.Select().(type) {
			case *cpuShader:
				panic("XXX no support for CPU shaders")
			case *wgpuShader:
				bindGroup := transientMap.createBindGroup(
					&eng.bindMap,
					&eng.pool,
					eng.Device,
					queue,
					encoder,
					s.bindGroupLayout,
					bindings,
				)

				cpass := encoder.BeginComputePass(nil)
				defer cpass.Release()

				// TODO profiling
				cpass.SetPipeline(s.pipeline)
				cpass.SetBindGroup(0, bindGroup, nil)
				cpass.DispatchWorkgroups(wgSize[0], wgSize[1], wgSize[2])
				cpass.End()
				// TODO profiling
			default:
				panic(fmt.Sprintf("unhandled type %T", s))
			}

		case renderer.DispatchIndirect:
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
					&eng.bindMap,
					&eng.pool,
					eng.Device,
					queue,
					encoder,
					s.bindGroupLayout,
					bindings,
				)

				transientMap.materializeGPUBufForIndirect(
					&eng.bindMap,
					&eng.pool,
					eng.Device,
					queue,
					proxy,
				)

				cpass := encoder.BeginComputePass(nil)
				defer cpass.Release()

				// TODO profiling
				cpass.SetPipeline(s.pipeline)
				cpass.SetBindGroup(0, bindGroup, nil)
				buf, ok := eng.bindMap.getGPUBuf(proxy.ID)
				if !ok {
					panic("tried using unavailable buffer for indirect dispatch")
				}
				cpass.DispatchWorkgroupsIndirect(buf, offset)
				cpass.End()
				// TODO profiling
			default:
				panic(fmt.Sprintf("unhandled type %T", s))
			}

		case renderer.Download:
			proxy := cmd.Buffer
			srcBuf, ok := eng.bindMap.getGPUBuf(proxy.ID)
			if !ok {
				panic("tried using unavailable buffer for download")
			}
			usage := wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst
			buf := eng.pool.getBuf(proxy.Size, "download", usage, eng.Device)
			encoder.CopyBufferToBuffer(srcBuf, 0, buf, 0, proxy.Size)
			eng.downloads[proxy.ID] = buf

		case renderer.Clear:
			proxy := cmd.Buffer
			offset := cmd.Offset
			size := cmd.Size
			if buf, ok := eng.bindMap.getBuf(proxy); ok {
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
				eng.bindMap.pendingClears[proxy.ID] = struct{}{}
			}

		case renderer.FreeBuffer:
			freeBufs[cmd.Buffer.ID] = struct{}{}

		case renderer.FreeImage:
			freeImages[cmd.Image.ID] = struct{}{}

		default:
			panic(fmt.Sprintf("unhandled command %T", cmd))
		}
	}
	// TODO profiling
	cmd := encoder.Finish(nil)
	defer cmd.Release()
	queue.Submit(cmd)

	for id := range freeBufs {
		buf, ok := eng.bindMap.bufMap[id]
		if ok {
			delete(eng.bindMap.bufMap, id)
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
	for id := range freeImages {
		tex, ok := eng.bindMap.imageMap[id]
		if ok {
			delete(eng.bindMap.imageMap, id)
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
	shaderModule := eng.Device.MustCreateShaderModule(wgpu.ShaderModuleDescriptor{
		Label:  label,
		Source: wgpu.ShaderSourceWGSL(wgsl),
	})
	bindGroupLayout := eng.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: entries,
	})
	computePipelineLayout := eng.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	defer computePipelineLayout.Release()
	pipeline := eng.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  label,
		Layout: computePipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shaderModule,
			EntryPoint: "main",
			// XXX compilation_options
		},
	})

	return wgpuShader{
		label:           label,
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
	}
}

func (m *bindMap) insertBuf(proxy renderer.BufferProxy, buffer *wgpu.Buffer) {
	m.bufMap[proxy.ID] = &bindMapBuffer{
		Buffer: buffer,
		Label:  proxy.Name,
	}
}

func (m *bindMap) getGPUBuf(id renderer.ResourceID) (*wgpu.Buffer, bool) {
	mbuf, ok := m.bufMap[id]
	if !ok {
		return nil, false
	}
	buf, ok := mbuf.Buffer.(*wgpu.Buffer)
	return buf, ok
}

func (m *bindMap) getCPUBuf(id renderer.ResourceID) cpuBinding {
	buf, ok := m.bufMap[id].Buffer.([]byte)
	if !ok {
		panic("getting CPU buffer, but it's on GPU")
	}
	return cpuBufferRW(buf)
}

func (m *bindMap) materializeCPUBuf(proxy renderer.BufferProxy) {
	if _, ok := m.bufMap[proxy.ID]; !ok {
		buffer := make([]byte, proxy.Size)
		m.bufMap[proxy.ID] = &bindMapBuffer{
			Buffer: buffer,
			Label:  proxy.Name,
		}
	}
}

func (m *bindMap) insertImage(id renderer.ResourceID, image *wgpu.Texture, imageView *wgpu.TextureView) {
	m.imageMap[id] = &bindMapImage{image, imageView}
}

func (m *bindMap) getBuf(proxy renderer.BufferProxy) (*bindMapBuffer, bool) {
	b, ok := m.bufMap[proxy.ID]
	return b, ok
}

func (m *bindMap) getOrCreateImage(
	proxy renderer.ImageProxy,
	dev *wgpu.Device,
) (*wgpu.Texture, *wgpu.TextureView) {
	if entry, ok := m.imageMap[proxy.ID]; ok {
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
	m.imageMap[proxy.ID] = &bindMapImage{
		texture, textureView,
	}

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

func newTransientBindMap(externalResources []ExternalResource) transientBindMap {
	bufs := map[renderer.ResourceID]transientBuf{}
	images := map[renderer.ResourceID]*wgpu.TextureView{}
	for _, res := range externalResources {
		switch res := res.(type) {
		case ExternalBuffer:
			bufs[res.Proxy.ID] = res.Buffer
		case ExternalImage:
			images[res.Proxy.ID] = res.View
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
	if _, ok := m.bufs[buf.ID]; ok {
		return
	}
	if b, ok := bindMap.bufMap[buf.ID]; ok {
		b.uploadIfNeeded(buf, dev, queue, pool)
	}
}

func (m *transientBindMap) createBindGroup(
	bindMap *bindMap,
	pool *resourcePool,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	encoder *wgpu.CommandEncoder,
	layout *wgpu.BindGroupLayout,
	bindings []renderer.ResourceProxy,
) *wgpu.BindGroup {
	for _, proxy := range bindings {
		switch proxy := proxy.(type) {
		case renderer.BufferProxy:
			if _, ok := m.bufs[proxy.ID]; ok {
				continue
			}
			if o, ok := bindMap.bufMap[proxy.ID]; ok {
				o.uploadIfNeeded(proxy, dev, queue, pool)
			} else {
				// TODO: only some buffers will need indirect, but does it hurt?
				usage := wgpu.BufferUsageCopySrc |
					wgpu.BufferUsageCopyDst |
					wgpu.BufferUsageStorage |
					wgpu.BufferUsageIndirect
				buf := pool.getBuf(proxy.Size, proxy.Name, usage, dev)
				if _, ok := bindMap.pendingClears[proxy.ID]; ok {
					delete(bindMap.pendingClears, proxy.ID)
					encoder.ClearBuffer(buf, 0, buf.Size())
				}
				bindMap.bufMap[proxy.ID] = &bindMapBuffer{
					Buffer: buf,
					Label:  proxy.Name,
				}
			}
		case renderer.ImageProxy:
			if _, ok := m.images[proxy.ID]; ok {
				continue
			}
			if _, ok := bindMap.imageMap[proxy.ID]; ok {
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
			bindMap.imageMap[proxy.ID] = &bindMapImage{
				texture, textureView,
			}
		default:
			panic(fmt.Sprintf("unhandled type %T", proxy))
		}
	}

	entries := make([]wgpu.BindGroupEntry, len(bindings))
	for i, proxy := range bindings {
		switch proxy := proxy.(type) {
		case renderer.BufferProxy:
			var buf *wgpu.Buffer
			switch b := m.bufs[proxy.ID].(type) {
			case *wgpu.Buffer:
				buf = b
			default:
				var ok bool
				buf, ok = bindMap.getGPUBuf(proxy.ID)
				if !ok {
					panic("unexpected ok == false")
				}
			}
			entries[i] = wgpu.BindGroupEntry{
				Binding: uint32(i),
				Buffer:  buf,
				Size:    ^uint64(0),
			}
		case renderer.ImageProxy:
			view, ok := m.images[proxy.ID]
			if !ok {
				img, ok := bindMap.imageMap[proxy.ID]
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

	return dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  layout,
		Entries: entries,
	})
}

func (m *transientBindMap) createCPUResources(
	bindMap *bindMap,
	bindings []renderer.ResourceProxy,
) []cpuBinding {
	for _, resource := range bindings {
		switch resource := resource.(type) {
		case renderer.BufferProxy:
			switch tbuf := m.bufs[resource.ID].(type) {
			case []byte:
			case *wgpu.Buffer:
				panic("buffer was already materialized on GPU")
			case nil:
				bindMap.materializeCPUBuf(resource)
			default:
				panic(fmt.Sprintf("unhandled type %T", tbuf))
			}
		case renderer.ImageProxy:
			panic("not implemented")
		default:
			panic(fmt.Sprintf("unhandled type %T", resource))
		}
	}

	out := make([]cpuBinding, len(bindings))
	for i, resource := range bindings {
		switch resource := resource.(type) {
		case renderer.BufferProxy:
			switch tbuf := m.bufs[resource.ID].(type) {
			case []byte:
				out[i] = cpuBuffer(tbuf)
			default:
				out[i] = bindMap.getCPUBuf(resource.ID)
			}
		case renderer.ImageProxy:
			panic("not implemented")
		default:
			panic(fmt.Sprintf("unhandled type %T", resource))
		}
	}
	return out
}
