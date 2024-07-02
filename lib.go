package jello

import (
	"honnef.co/go/brush"
	"honnef.co/go/wgpu"
)

type AaConfig int

const (
	Area AaConfig = iota
	Msaa8
	Msaa16
)

type RenderParams struct {
	BaseColor          brush.Color
	Width              uint32
	Height             uint32
	AntialiasingMethod AaConfig
}

type Renderer struct {
	options  RendererOptions
	engine   *WGPUEngine
	resolver *Resolver
	shaders  *FullShaders
	blit     *BlitPipeline
	target   *TargetTexture
	// TODO profiling
}

type RendererOptions struct {
	SurfaceFormat       wgpu.TextureFormat
	UseCPU              bool
	AntialiasingSupport AaSupport
	// TODO threading for shader init
}

func NewRenderer(dev *wgpu.Device, options RendererOptions) *Renderer {
	engine := NewWGPUEngine(options.UseCPU)
	// TODO threading for shader init
	shaders := NewFullShaders(dev, engine, &options)
	engine.BuildShadersIfNeeded(dev, 1)
	// XXX support surfaceless engine use
	blit := NewBlitPipeline(dev, options.SurfaceFormat)

	return &Renderer{
		options:  options,
		engine:   engine,
		resolver: NewResolver(),
		shaders:  shaders,
		blit:     blit,
		target:   nil,
		// TODO profiling
	}
}

func RenderFull(
	scene *Scene,
	resolver *Resolver,
	shaders *FullShaders,
	params *RenderParams,
) (*Recording, ResourceProxy) {
	return RenderEncodingFull(&scene.Encoding, resolver, shaders, params)
}

func RenderEncodingFull(
	encoding *Encoding,
	resolver *Resolver,
	shaders *FullShaders,
	params *RenderParams,
) (*Recording, ResourceProxy) {
	var render Render
	recording := render.RenderEncodingCoarse(encoding, resolver, shaders, params, false)
	outImage := render.OutImage()
	render.RecordFine(shaders, recording)
	return recording, outImage
}

func (r *Renderer) RenderToTexture(
	dev *wgpu.Device,
	queue *wgpu.Queue,
	scene *Scene,
	texture *wgpu.TextureView,
	params *RenderParams,
) {
	recording, target := RenderFull(scene, r.resolver, r.shaders, params)
	externalResources := []ExternalResource{
		ExternalImage{
			Proxy: target.(ImageProxy),
			View:  texture,
		},
	}
	r.engine.RunRecording(dev, queue, recording, externalResources, "render_to_texture")
}

func (r *Renderer) RenderToSurface(
	dev *wgpu.Device,
	queue *wgpu.Queue,
	scene *Scene,
	surface *wgpu.SurfaceTexture,
	params *RenderParams,
) {
	width := params.Width
	height := params.Height
	if r.target == nil {
		r.target = NewTargetTexture(dev, width, height)
	} else if r.target.Width != width || r.target.Height != height {
		r.target = NewTargetTexture(dev, width, height)
	}
	r.RenderToTexture(dev, queue, scene, r.target.View, params)
	encoder := dev.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "blitter"})
	defer encoder.Release()

	surfaceView := surface.Texture.CreateView(nil)
	// surfaceView := surface.Texture.CreateView(&wgpu.TextureViewDescriptor{
	// 	Format:          surface.Texture.Format(),
	// 	Dimension:       wgpu.TextureViewDimension2D,
	// 	BaseMipLevel:    0,
	// 	MipLevelCount:   1,
	// 	BaseArrayLayer:  0,
	// 	ArrayLayerCount: 1,
	// 	Aspect:          wgpu.TextureAspectAll,
	// })
	defer surfaceView.Release()

	bindGroup := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: r.blit.BindLayout,
		Entries: []wgpu.BindGroupEntry{
			{
				Binding:     0,
				TextureView: r.target.View,
			},
		},
	})
	defer bindGroup.Release()
	renderPass := encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:       surfaceView,
				LoadOp:     wgpu.LoadOpClear,
				StoreOp:    wgpu.StoreOpStore,
				ClearValue: wgpu.Color{0, 255, 0, 255},
			},
		},
	})
	defer renderPass.Release()

	// TODO profiling
	renderPass.SetPipeline(r.blit.Pipeline)
	renderPass.SetBindGroup(0, bindGroup, nil)
	renderPass.Draw(6, 1, 0, 0)
	renderPass.End()

	// TODO profiling
	cmd := encoder.Finish(nil)
	defer cmd.Release()
	queue.Submit(cmd)
	// TODO profiling
}

type BlitPipeline struct {
	BindLayout *wgpu.BindGroupLayout
	Pipeline   *wgpu.RenderPipeline
}

func NewBlitPipeline(dev *wgpu.Device, format wgpu.TextureFormat) *BlitPipeline {
	const src = `
			@vertex
			fn vs_main(@builtin(vertex_index) ix: u32) -> @builtin(position) vec4<f32> {
				// Generate a full screen quad in normalized device coordinates
				var vertex = vec2(-1.0, 1.0);
				switch ix {
					case 1u: {
						vertex = vec2(-1.0, -1.0);
					}
					case 2u, 4u: {
						vertex = vec2(1.0, -1.0);
					}
					case 5u: {
						vertex = vec2(1.0, 1.0);
					}
					default: {}
				}
				return vec4(vertex, 0.0, 1.0);
			}

			@group(0) @binding(0)
			var fine_output: texture_2d<f32>;

			@fragment
			fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
				let rgba_sep = textureLoad(fine_output, vec2<i32>(pos.xy), 0);
				return vec4(rgba_sep.rgb * rgba_sep.a, rgba_sep.a);
			}
`

	shader := dev.MustCreateShaderModule(wgpu.ShaderModuleDescriptor{
		Label:  "blit shaders",
		Source: wgpu.ShaderSourceWGSL(src),
	})
	bindLayout := dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Visibility: wgpu.ShaderStageFragment,
				Binding:    0,
				Texture: &wgpu.TextureBindingLayout{
					SampleType:    wgpu.TextureSampleTypeFloat,
					ViewDimension: wgpu.TextureViewDimension2D,
					Multisampled:  false,
				},
			},
		},
	})
	pipelineLayout := dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "blit pipeline layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindLayout},
	})
	pipeline := dev.MustCreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  "blit pipeline",
		Layout: pipelineLayout,
		Vertex: &wgpu.VertexState{
			Module:     shader,
			EntryPoint: "vs_main",
		},
		Fragment: &wgpu.FragmentState{
			Module:     shader,
			EntryPoint: "fs_main",
			Targets: []wgpu.ColorTargetState{
				{
					Format:    format,
					WriteMask: wgpu.ColorWriteMaskAll,
				},
			},
		},
		Primitive: &wgpu.PrimitiveState{
			Topology:         wgpu.PrimitiveTopologyTriangleList,
			StripIndexFormat: ^wgpu.IndexFormat(0), // XXX 0 or Undefined?
			FrontFace:        wgpu.FrontFaceCCW,
			CullMode:         wgpu.CullModeBack,
		},
		Multisample: &wgpu.MultisampleState{
			Count:                  1,
			Mask:                   ^uint32(0),
			AlphaToCoverageEnabled: false,
		},
	})
	return &BlitPipeline{
		BindLayout: bindLayout,
		Pipeline:   pipeline,
	}
}

type TargetTexture struct {
	View   *wgpu.TextureView
	Width  uint32
	Height uint32
}

func NewTargetTexture(dev *wgpu.Device, width, height uint32) *TargetTexture {
	tex := dev.CreateTexture(&wgpu.TextureDescriptor{
		Label: "target texture",
		Size: wgpu.Extent3D{
			Width:              width,
			Height:             height,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
		Format:        wgpu.TextureFormatRGBA8Unorm,
	})
	defer tex.Release()
	view := tex.CreateView(nil)
	return &TargetTexture{
		View:   view,
		Width:  width,
		Height: height,
	}
}

type AaSupport struct {
	Area   bool
	MSAA8  bool
	MSAA16 bool
}
