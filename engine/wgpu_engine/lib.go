package wgpu_engine

import (
	"fmt"
	"reflect"

	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/engine/wgpu_engine/shaders"
	"honnef.co/go/jello/mem"
	"honnef.co/go/jello/renderer"
	"honnef.co/go/wgpu"
)

type AaSupport struct {
	Area   bool
	MSAA8  bool
	MSAA16 bool
}

type RendererOptions struct {
	SurfaceFormat       wgpu.TextureFormat
	UseCPU              bool
	AntialiasingSupport AaSupport
	// TODO threading for shader init
}

var bindTypeMapping = [...]renderer.BindType{
	shaders.Buffer:      {Type: renderer.BindTypeBuffer},
	shaders.BufReadOnly: {Type: renderer.BindTypeBufReadOnly},
	shaders.Uniform:     {Type: renderer.BindTypeUniform},
	shaders.Image:       {Type: renderer.BindTypeImage, ImageFormat: renderer.Rgba8},
	shaders.ImageRead:   {Type: renderer.BindTypeImageRead, ImageFormat: renderer.Rgba8},
}

func (engine *Engine) newFullShaders() *renderer.FullShaders {
	// XXX make use of options.AntialiasingSupport
	// XXX support CPU shaders
	var out renderer.FullShaders
	outV := reflect.ValueOf(&out).Elem()
	v := reflect.ValueOf(&shaders.Collection)
	for i := range v.Elem().NumField() {
		fieldName := v.Elem().Type().Field(i).Name
		outField := outV.FieldByName(fieldName)
		if !outField.IsValid() {
			continue
		}
		shader := v.Elem().Field(i).Addr().Interface().(*shaders.ComputeShader)
		bindings := make([]renderer.BindType, len(shader.Bindings))
		for i, b := range shader.Bindings {
			bindings[i] = bindTypeMapping[b]
		}
		if len(shader.WGSL.Code) == 0 {
			panic(fmt.Sprintf("shader %q has no code", shader.Name))
		}
		id := engine.addShader(shader.Name, shader.WGSL.Code, bindings, nil)
		outField.Set(reflect.ValueOf(id))
	}
	return &out
}

type blitPipeline struct {
	BindLayout *wgpu.BindGroupLayout
	Pipeline   *wgpu.RenderPipeline
}

func newBlitPipeline(dev *wgpu.Device, format wgpu.TextureFormat) *blitPipeline {
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
			}`

	shader := dev.CreateShaderModule(wgpu.ShaderModuleDescriptor{
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
	pipeline := dev.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
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
			StripIndexFormat: ^wgpu.IndexFormat(0),
			FrontFace:        wgpu.FrontFaceCCW,
			CullMode:         wgpu.CullModeBack,
		},
		Multisample: &wgpu.MultisampleState{
			Count:                  1,
			Mask:                   ^uint32(0),
			AlphaToCoverageEnabled: false,
		},
	})
	return &blitPipeline{
		BindLayout: bindLayout,
		Pipeline:   pipeline,
	}
}

type targetTexture struct {
	View   *wgpu.TextureView
	Width  uint32
	Height uint32
}

func newTargetTexture(dev *wgpu.Device, width, height uint32) *targetTexture {
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
	return &targetTexture{
		View:   view,
		Width:  width,
		Height: height,
	}
}

func imageFormatToWGPU(f renderer.ImageFormat) wgpu.TextureFormat {
	switch f {
	case renderer.Rgba8:
		return wgpu.TextureFormatRGBA8Unorm
	case renderer.Bgra8:
		return wgpu.TextureFormatBGRA8Unorm
	default:
		panic(fmt.Sprintf("unhandled value %d", f))
	}
}

func (eng *Engine) RenderToTexture(
	arena *mem.Arena,
	queue *wgpu.Queue,
	enc *encoding.Encoding,
	texture *wgpu.TextureView,
	params *renderer.RenderParams,
	pgroup *ProfilerGroup,
) {
	pgroup = pgroup.Nest("RenderToTexture")
	defer pgroup.End()

	recording, target := renderer.RenderFull(arena, enc, eng.resolver, eng.fullShaders, params, pgroup)

	externalResources := []ExternalResource{
		ExternalImage{
			Proxy: target.ImageProxy,
			View:  texture,
		},
	}
	eng.RunRecording(arena, queue, recording, externalResources, "render_to_texture", pgroup)
}

func (eng *Engine) RenderToSurface(
	arena *mem.Arena,
	queue *wgpu.Queue,
	enc *encoding.Encoding,
	surface *wgpu.SurfaceTexture,
	params *renderer.RenderParams,
	pgroup *ProfilerGroup,
) {
	pgroup = pgroup.Nest("RenderToSurface")
	defer pgroup.End()

	width := params.Width
	height := params.Height
	if eng.target == nil {
		eng.target = newTargetTexture(eng.Device, width, height)
	} else if eng.target.Width != width || eng.target.Height != height {
		eng.target.View.Release()
		eng.target = newTargetTexture(eng.Device, width, height)
	}

	ency := eng.Device.CreateCommandEncoder(nil)
	span := pgroup.Begin(ency, "total")
	cmdy := ency.Finish(nil)
	defer cmdy.Release()
	queue.Submit(cmdy)

	eng.RenderToTexture(arena, queue, enc, eng.target.View, params, pgroup)

	surfaceView := surface.Texture.CreateView(nil)
	defer surfaceView.Release()

	bindGroup := eng.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: eng.blit.BindLayout,
		Entries: []wgpu.BindGroupEntry{
			{
				Binding:     0,
				TextureView: eng.target.View,
			},
		},
	})
	defer bindGroup.Release()

	encoder := eng.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "blitter"})
	defer encoder.Release()
	renderPass := encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:       surfaceView,
				LoadOp:     wgpu.LoadOpClear,
				StoreOp:    wgpu.StoreOpStore,
				ClearValue: wgpu.Color{R: 0, G: 255, B: 0, A: 255},
			},
		},
		TimestampWrites: pgroup.Render(arena, "blit"),
	})
	defer renderPass.Release()

	renderPass.SetPipeline(eng.blit.Pipeline)
	renderPass.SetBindGroup(0, bindGroup, nil)
	renderPass.Draw(6, 1, 0, 0)
	renderPass.End()

	span.End(encoder)
	cmd := encoder.Finish(nil)
	defer cmd.Release()
	queue.Submit(cmd)

}
