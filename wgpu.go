package jello

// OPT reuse bind groups

import (
	"fmt"
	"math"
	"math/bits"

	"honnef.co/go/wgpu"
)

type UninitializedShader struct {
	Wgsl     []byte
	Label    string
	Entries  []wgpu.BindGroupLayoutEntry
	ShaderID ShaderID
}

type WGPUEngine struct {
	Shaders             []Shader
	Pool                ResourcePool
	BindMap             BindMap
	Downloads           map[ResourceID]*wgpu.Buffer
	ShadersToInitialize []UninitializedShader
	UseCPU              bool
}

type WGPUShader struct {
	Label           string
	Pipeline        *wgpu.ComputePipeline
	BindGroupLayout *wgpu.BindGroupLayout
}

type CPUShader struct {
	Shader func(uint32, []CPUBinding)
}

type Shader struct {
	Label string
	WGPU  *WGPUShader
	CPU   *CPUShader
}

func (s Shader) Select() any {
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
	Proxy  BufferProxy
	Buffer *wgpu.Buffer
}

type ExternalImage struct {
	Proxy ImageProxy
	View  *wgpu.TextureView
}

type MaterializedBuffer interface {
	// One of wgpu.Buffer and []byte
}

type BindMapBuffer struct {
	Buffer MaterializedBuffer
	Label  string
}

type BindMap struct {
	BufMap   map[ResourceID]*BindMapBuffer
	ImageMap map[ResourceID]struct {
		Texture *wgpu.Texture
		View    *wgpu.TextureView
	}
	PendingClears map[ResourceID]struct{}
}

type BufferProperties struct {
	Size   uint64
	Usages wgpu.BufferUsage
}

type ResourcePool struct {
	Bufs map[BufferProperties][]*wgpu.Buffer
}

// XXX we probably don't need this type, it's probably only useful with Rust
// lifetimes
type TransientBindMap struct {
	Bufs   map[ResourceID]TransientBuf
	Images map[ResourceID]*wgpu.TextureView
}

type TransientBuf interface {
	// One of []byte and wgpu.Buffer
}

func NewWGPUEngine(useCPU bool) *WGPUEngine {
	return &WGPUEngine{
		Pool: ResourcePool{
			Bufs: make(map[BufferProperties][]*wgpu.Buffer),
		},
		BindMap: BindMap{
			BufMap: make(map[ResourceID]*BindMapBuffer),
			ImageMap: make(map[ResourceID]struct {
				Texture *wgpu.Texture
				View    *wgpu.TextureView
			}),
			PendingClears: make(map[ResourceID]struct{})},
		Downloads: make(map[ResourceID]*wgpu.Buffer),
		UseCPU:    useCPU,
	}
}

func (eng *WGPUEngine) UseParallelInitialization() {
	if eng.ShadersToInitialize != nil {
		return
	}
	eng.ShadersToInitialize = []UninitializedShader{}
}

func (eng *WGPUEngine) BuildShadersIfNeeded(dev *wgpu.Device, numThreads int) {
	if eng.ShadersToInitialize == nil {
		return
	}
	newShaders := eng.ShadersToInitialize
	// XXX implement parallelism
	for _, s := range newShaders {
		shader := eng.CreateComputePipeline(dev, s.Label, s.Wgsl, s.Entries)
		if int(s.ShaderID) >= len(eng.Shaders) {
			if cap(eng.Shaders) <= int(s.ShaderID) {
				c := make([]Shader, s.ShaderID+1)
				copy(c, eng.Shaders)
				eng.Shaders = c
			} else {
				eng.Shaders = eng.Shaders[:s.ShaderID+1]
			}
		}
		eng.Shaders[s.ShaderID] = Shader{WGPU: &shader}
	}
}

type CPUShaderType interface {
	// XXX implement
}

func (eng *WGPUEngine) AddShader(
	dev *wgpu.Device,
	label string,
	wgsl []byte,
	layout []BindType,
	cpuShader CPUShaderType,
) ShaderID {
	add := func(shader Shader) ShaderID {
		id := len(eng.Shaders)
		eng.Shaders = append(eng.Shaders, shader)
		return ShaderID(id)
	}

	if eng.UseCPU {
		panic("XXX unimplemented")
	}

	entries := make([]wgpu.BindGroupLayoutEntry, len(layout))
	for i, bindType := range layout {
		switch bindType.Type {
		case BindTypeBuffer, BindTypeBufReadOnly:
			var typ wgpu.BufferBindingType
			if bindType.Type == BindTypeBuffer {
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
		case BindTypeUniform:
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				Buffer: &wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeUniform,
					HasDynamicOffset: false,
					MinBindingSize:   0, // XXX 0 or Undefined?
				},
			}

		case BindTypeImage:
			entries[i] = wgpu.BindGroupLayoutEntry{
				Binding:    uint32(i),
				Visibility: wgpu.ShaderStageCompute,
				StorageTexture: &wgpu.StorageTextureBindingLayout{
					Access:        wgpu.StorageTextureAccessWriteOnly,
					Format:        bindType.ImageFormat.toWGPU(),
					ViewDimension: wgpu.TextureViewDimension2D,
				},
			}

		case BindTypeImageRead:
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

	if eng.ShadersToInitialize != nil {
		id := add(Shader{Label: label})
		eng.ShadersToInitialize = append(eng.ShadersToInitialize, UninitializedShader{
			Wgsl:     wgsl,
			Label:    label,
			Entries:  entries,
			ShaderID: id,
		})
		return id
	}

	wgpu := eng.CreateComputePipeline(dev, label, wgsl, entries)
	return add(Shader{
		Label: label,
		WGPU:  &wgpu,
	})
}

func (eng *WGPUEngine) RunRecording(
	dev *wgpu.Device,
	queue *wgpu.Queue,
	recording *Recording,
	externalResources []ExternalResource,
	label string,
	// TODO profiler
) {
	freeBufs := map[ResourceID]struct{}{}
	freeImages := map[ResourceID]struct{}{}
	transientMap := NewTransientBindMap(externalResources)
	defer transientMap.Clear()

	encoder := dev.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: label})
	defer encoder.Release()

	// TODO profiler
	// OPT release things after every command
	for _, cmd := range recording.Commands {
		switch cmd := cmd.(type) {
		case Upload:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.Bufs[bufProxy.ID] = bytes
			usage := wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst | wgpu.BufferUsageStorage
			buf := eng.Pool.GetBuf(bufProxy.Size, bufProxy.Name, usage, dev)
			queue.WriteBuffer(buf, 0, bytes)
			eng.BindMap.InsertBuf(bufProxy, buf)

		case UploadUniform:
			bufProxy := cmd.Buffer
			bytes := cmd.Data
			transientMap.Bufs[bufProxy.ID] = bytes
			usage := wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst
			// XXXXXX "config" buffer is created here
			buf := eng.Pool.GetBuf(bufProxy.Size, bufProxy.Name, usage, dev)
			queue.WriteBuffer(buf, 0, bytes)
			eng.BindMap.InsertBuf(bufProxy, buf)

		case UploadImage:
			imageProxy := cmd.Image
			bytes := cmd.Data
			format := imageProxy.Format.toWGPU()
			blockSize, ok := format.BlockCopySize(wgpu.TextureAspectAll)
			if !ok {
				panic("image format must have a valid block size")
			}
			texture := dev.CreateTexture(&wgpu.TextureDescriptor{
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
			eng.BindMap.InsertImage(imageProxy.ID, texture, textureView)

		case WriteImage:
			proxy := cmd.Image
			x := cmd.Coords[0]
			y := cmd.Coords[1]
			width := cmd.Coords[2]
			height := cmd.Coords[3]
			data := cmd.Data
			texture, _ := eng.BindMap.GetOrCreateImage(proxy, dev)
			format := proxy.Format.toWGPU()
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

		case Dispatch:
			shaderID := cmd.Shader
			wgSize := cmd.WorkgroupSize
			bindings := cmd.Bindings
			shader := eng.Shaders[shaderID]
			switch s := shader.Select().(type) {
			case *CPUShader:
				panic("XXX no support for CPU shaders")
			case *WGPUShader:
				bindGroup := transientMap.CreateBindGroup(
					&eng.BindMap,
					&eng.Pool,
					dev,
					queue,
					encoder,
					s.BindGroupLayout,
					bindings,
				)

				cpass := encoder.BeginComputePass(nil)
				defer cpass.Release()

				// TODO profiling
				cpass.SetPipeline(s.Pipeline)
				cpass.SetBindGroup(0, bindGroup, nil)
				cpass.DispatchWorkgroups(wgSize[0], wgSize[1], wgSize[2])
				cpass.End()
				// TODO profiling
			default:
				panic(fmt.Sprintf("unhandled type %T", s))
			}

		case DispatchIndirect:
			shaderID := cmd.Shader
			proxy := cmd.Buffer
			offset := cmd.Offset
			bindings := cmd.Bindings
			shader := eng.Shaders[shaderID]
			switch s := shader.Select().(type) {
			case *CPUShader:
				panic("XXX no support for CPU shaders")
			case *WGPUShader:
				bindGroup := transientMap.CreateBindGroup(
					&eng.BindMap,
					&eng.Pool,
					dev,
					queue,
					encoder,
					s.BindGroupLayout,
					bindings,
				)

				transientMap.MaterializeGPUBufForIndirect(
					&eng.BindMap,
					&eng.Pool,
					dev,
					queue,
					proxy,
				)

				cpass := encoder.BeginComputePass(nil)
				defer cpass.Release()

				// TODO profiling
				cpass.SetPipeline(s.Pipeline)
				cpass.SetBindGroup(0, bindGroup, nil)
				buf, ok := eng.BindMap.GetGPUBuf(proxy.ID)
				if !ok {
					panic("tried using unavailable buffer for indirect dispatch")
				}
				cpass.DispatchWorkgroupsIndirect(buf, offset)
				cpass.End()
				// TODO profiling
			default:
				panic(fmt.Sprintf("unhandled type %T", s))
			}

		case Download:
			proxy := cmd.Buffer
			srcBuf, ok := eng.BindMap.GetGPUBuf(proxy.ID)
			if !ok {
				panic("tried using unavailable buffer for download")
			}
			usage := wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst
			buf := eng.Pool.GetBuf(proxy.Size, "download", usage, dev)
			encoder.CopyBufferToBuffer(srcBuf, 0, buf, 0, proxy.Size)
			eng.Downloads[proxy.ID] = buf

		case Clear:
			proxy := cmd.Buffer
			offset := cmd.Offset
			size := cmd.Size
			if buf, ok := eng.BindMap.GetBuf(proxy); ok {
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
				eng.BindMap.PendingClears[proxy.ID] = struct{}{}
			}

		case FreeBuffer:
			freeBufs[cmd.Buffer.ID] = struct{}{}

		case FreeImage:
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
		buf, ok := eng.BindMap.BufMap[id]
		if ok {
			delete(eng.BindMap.BufMap, id)
			if gpuBuf, ok := buf.Buffer.(*wgpu.Buffer); ok {
				props := BufferProperties{
					Size:   gpuBuf.Size(),
					Usages: gpuBuf.Usage(),
				}
				// TODO(dh): add a method to ResourcePool to return buffers
				eng.Pool.Bufs[props] = append(eng.Pool.Bufs[props], gpuBuf)
			}
		}
	}
	for id := range freeImages {
		tex, ok := eng.BindMap.ImageMap[id]
		if ok {
			delete(eng.BindMap.ImageMap, id)
			// TODO: have a pool to avoid needless re-allocation
			tex.Texture.Release()
			tex.View.Release()
		}
	}
}

func (eng *WGPUEngine) GetDownload(buf BufferProxy) (*wgpu.Buffer, bool) {
	got, ok := eng.Downloads[buf.ID]
	return got, ok
}

func (eng *WGPUEngine) FreeDownload(buf BufferProxy) {
	delete(eng.Downloads, buf.ID)
}

func (eng *WGPUEngine) CreateComputePipeline(
	dev *wgpu.Device,
	label string,
	wgsl []byte,
	entries []wgpu.BindGroupLayoutEntry,
) WGPUShader {
	shaderModule := dev.MustCreateShaderModule(wgpu.ShaderModuleDescriptor{
		Label:  label,
		Source: wgpu.ShaderSourceWGSL(wgsl),
	})
	bindGroupLayout := dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: entries,
	})
	computePipelineLayout := dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	defer computePipelineLayout.Release()
	pipeline := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  label,
		Layout: computePipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shaderModule,
			EntryPoint: "main",
			// XXX compilation_options
		},
	})

	return WGPUShader{
		Label:           label,
		Pipeline:        pipeline,
		BindGroupLayout: bindGroupLayout,
	}
}

func (m *BindMap) InsertBuf(proxy BufferProxy, buffer *wgpu.Buffer) {
	m.BufMap[proxy.ID] = &BindMapBuffer{
		Buffer: buffer,
		Label:  proxy.Name,
	}
}

func (m *BindMap) GetGPUBuf(id ResourceID) (*wgpu.Buffer, bool) {
	mbuf, ok := m.BufMap[id]
	if !ok {
		return nil, false
	}
	buf, ok := mbuf.Buffer.(*wgpu.Buffer)
	return buf, ok
}

func (m *BindMap) GetCPUBuf(id ResourceID) CPUBinding {
	buf, ok := m.BufMap[id].Buffer.([]byte)
	if !ok {
		panic("getting CPU buffer, but it's on GPU")
	}
	return CPUBufferRW(buf)
}

func (m *BindMap) MaterializeCPUBuf(proxy BufferProxy) {
	if _, ok := m.BufMap[proxy.ID]; !ok {
		buffer := make([]byte, proxy.Size)
		m.BufMap[proxy.ID] = &BindMapBuffer{
			Buffer: buffer,
			Label:  proxy.Name,
		}
	}
}

func (m *BindMap) InsertImage(id ResourceID, image *wgpu.Texture, imageView *wgpu.TextureView) {
	m.ImageMap[id] = struct {
		Texture *wgpu.Texture
		View    *wgpu.TextureView
	}{
		image, imageView,
	}
}

func (m *BindMap) GetBuf(proxy BufferProxy) (*BindMapBuffer, bool) {
	b, ok := m.BufMap[proxy.ID]
	return b, ok
}

func (m *BindMap) GetOrCreateImage(
	proxy ImageProxy,
	dev *wgpu.Device,
) (*wgpu.Texture, *wgpu.TextureView) {
	if entry, ok := m.ImageMap[proxy.ID]; ok {
		return entry.Texture, entry.View
	}

	format := proxy.Format.toWGPU()
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
		Format:          proxy.Format.toWGPU(),
	})
	m.ImageMap[proxy.ID] = struct {
		Texture *wgpu.Texture
		View    *wgpu.TextureView
	}{
		texture, textureView,
	}

	return texture, textureView
}

func (pool *ResourcePool) GetBuf(
	size uint64,
	name string,
	usage wgpu.BufferUsage,
	dev *wgpu.Device,
) *wgpu.Buffer {
	const sizeClassBits = 1

	roundedSize := poolSizeClass(size, sizeClassBits)
	props := BufferProperties{
		Size:   roundedSize,
		Usages: usage,
	}
	if bufVec, ok := pool.Bufs[props]; ok {
		if len(bufVec) > 0 {
			buf := bufVec[len(bufVec)-1]
			bufVec = bufVec[:len(bufVec)-1]
			pool.Bufs[props] = bufVec
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

func (b *BindMapBuffer) UploadIfNeeded(
	proxy BufferProxy,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	pool *ResourcePool,
) {
	cpuBuf, ok := b.Buffer.([]byte)
	if !ok {
		return
	}
	usage := wgpu.BufferUsageCopySrc |
		wgpu.BufferUsageCopyDst |
		wgpu.BufferUsageStorage |
		wgpu.BufferUsageIndirect
	buf := pool.GetBuf(proxy.Size, proxy.Name, usage, dev)
	queue.WriteBuffer(buf, 0, cpuBuf)
	b.Buffer = buf
}

func NewTransientBindMap(externalResources []ExternalResource) TransientBindMap {
	bufs := map[ResourceID]TransientBuf{}
	images := map[ResourceID]*wgpu.TextureView{}
	for _, res := range externalResources {
		switch res := res.(type) {
		case ExternalBuffer:
			bufs[res.Proxy.ID] = res.Buffer
		case ExternalImage:
			images[res.Proxy.ID] = res.View
		}
	}
	return TransientBindMap{
		Bufs:   bufs,
		Images: images,
	}
}

func (m *TransientBindMap) Clear() {
	return
	// XXX is this correct? is this what transient bind map is meant to be used for?
	for _, buf := range m.Bufs {
		if buf, ok := buf.(*wgpu.Buffer); ok {
			buf.Release()
		}
	}
	for _, img := range m.Images {
		img.Release()
	}
}

func (m *TransientBindMap) MaterializeGPUBufForIndirect(
	bindMap *BindMap,
	pool *ResourcePool,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	buf BufferProxy,
) {
	if _, ok := m.Bufs[buf.ID]; ok {
		return
	}
	if b, ok := bindMap.BufMap[buf.ID]; ok {
		b.UploadIfNeeded(buf, dev, queue, pool)
	}
}

func (m *TransientBindMap) CreateBindGroup(
	bindMap *BindMap,
	pool *ResourcePool,
	dev *wgpu.Device,
	queue *wgpu.Queue,
	encoder *wgpu.CommandEncoder,
	layout *wgpu.BindGroupLayout,
	bindings []ResourceProxy,
) *wgpu.BindGroup {
	for _, proxy := range bindings {
		switch proxy := proxy.(type) {
		case BufferProxy:
			if _, ok := m.Bufs[proxy.ID]; ok {
				continue
			}
			if o, ok := bindMap.BufMap[proxy.ID]; ok {
				o.UploadIfNeeded(proxy, dev, queue, pool)
			} else {
				// TODO: only some buffers will need indirect, but does it hurt?
				usage := wgpu.BufferUsageCopySrc |
					wgpu.BufferUsageCopyDst |
					wgpu.BufferUsageStorage |
					wgpu.BufferUsageIndirect
				buf := pool.GetBuf(proxy.Size, proxy.Name, usage, dev)
				if _, ok := bindMap.PendingClears[proxy.ID]; ok {
					delete(bindMap.PendingClears, proxy.ID)
					encoder.ClearBuffer(buf, 0, buf.Size())
				}
				bindMap.BufMap[proxy.ID] = &BindMapBuffer{
					Buffer: buf,
					Label:  proxy.Name,
				}
			}
		case ImageProxy:
			if _, ok := m.Images[proxy.ID]; ok {
				continue
			}
			if _, ok := bindMap.ImageMap[proxy.ID]; ok {
				continue
			}
			format := proxy.Format.toWGPU()
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
				Format:          proxy.Format.toWGPU(),
			})
			bindMap.ImageMap[proxy.ID] = struct {
				Texture *wgpu.Texture
				View    *wgpu.TextureView
			}{
				texture, textureView,
			}
		default:
			panic(fmt.Sprintf("unhandled type %T", proxy))
		}
	}

	entries := make([]wgpu.BindGroupEntry, len(bindings))
	for i, proxy := range bindings {
		switch proxy := proxy.(type) {
		case BufferProxy:
			var buf *wgpu.Buffer
			switch b := m.Bufs[proxy.ID].(type) {
			case *wgpu.Buffer:
				buf = b
			default:
				var ok bool
				buf, ok = bindMap.GetGPUBuf(proxy.ID)
				if !ok {
					panic("unexpected ok == false")
				}
			}
			entries[i] = wgpu.BindGroupEntry{
				Binding: uint32(i),
				Buffer:  buf,
				Size:    ^uint64(0),
			}
		case ImageProxy:
			view, ok := m.Images[proxy.ID]
			if !ok {
				img, ok := bindMap.ImageMap[proxy.ID]
				if !ok {
					panic("unexpected ok == false")
				}
				view = img.View
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

func (m *TransientBindMap) CreateCPUResources(
	bindMap *BindMap,
	bindings []ResourceProxy,
) []CPUBinding {
	for _, resource := range bindings {
		switch resource := resource.(type) {
		case BufferProxy:
			switch tbuf := m.Bufs[resource.ID].(type) {
			case []byte:
			case *wgpu.Buffer:
				panic("buffer was already materialized on GPU")
			case nil:
				bindMap.MaterializeCPUBuf(resource)
			default:
				panic(fmt.Sprintf("unhandled type %T", tbuf))
			}
		case ImageProxy:
			panic("not implemented")
		default:
			panic(fmt.Sprintf("unhandled type %T", resource))
		}
	}

	out := make([]CPUBinding, len(bindings))
	for i, resource := range bindings {
		switch resource := resource.(type) {
		case BufferProxy:
			switch tbuf := m.Bufs[resource.ID].(type) {
			case []byte:
				out[i] = CPUBuffer(tbuf)
			default:
				out[i] = bindMap.GetCPUBuf(resource.ID)
			}
		case ImageProxy:
			panic("not implemented")
		default:
			panic(fmt.Sprintf("unhandled type %T", resource))
		}
	}
	return out
}
