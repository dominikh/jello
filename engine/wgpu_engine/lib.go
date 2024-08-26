// Copyright 2022 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package wgpu_engine

import (
	"fmt"
	"reflect"

	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/engine/wgpu_engine/shaders"
	"honnef.co/go/jello/engine/wgpu_engine/shaders/cpu"
	"honnef.co/go/jello/mem"
	"honnef.co/go/jello/renderer"
	"honnef.co/go/wgpu"
)

type RendererOptions struct {
	SurfaceFormat wgpu.TextureFormat
	UseCPU        bool
}

var bindTypeMapping = [...]renderer.BindType{
	shaders.Buffer:           {Type: renderer.BindTypeBuffer},
	shaders.BufReadOnly:      {Type: renderer.BindTypeBufReadOnly},
	shaders.Uniform:          {Type: renderer.BindTypeUniform},
	shaders.Output:           {Type: renderer.BindTypeImage, ImageFormat: renderer.Rgba16Float},
	shaders.Image:            {Type: renderer.BindTypeImage, ImageFormat: renderer.Rgba8},
	shaders.ImageRead:        {Type: renderer.BindTypeImageRead, ImageFormat: renderer.Rgba8},
	shaders.ImageReadFloat16: {Type: renderer.BindTypeImageRead, ImageFormat: renderer.Rgba16Float},
	shaders.ImageArrayRead:   {Type: renderer.BindTypeImageArrayRead, ImageFormat: renderer.Rgba8},
}

func (engine *Engine) prepareShaders() *renderer.FullShaders {
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

		// TODO(dh): replace this hard coded switch with something more dynamic
		var cpus cpuShader
		switch shader.Name {
		case "backdrop_dyn":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.Backdrop}
		case "bbox_clear":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.BboxClear}
		case "binning":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.Binning}
		case "clip_leaf":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.ClipLeaf}
		case "clip_reduce":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.ClipReduce}
		case "coarse":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.Coarse}
		case "draw_leaf":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.DrawLeaf}
		case "draw_reduce":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.DrawReduce}
		case "fine_area":
			// No CPU equivalent
		case "fine_msaa16":
			// No CPU equivalent
		case "fine_msaa8":
			// No CPU equivalent
		case "flatten":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.Flatten}
		case "path_count":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.PathCount}
		case "path_count_setup":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.PathCountSetup}
		case "path_tiling":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.PathTiling}
		case "path_tiling_setup":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.PathTilingSetup}
		case "pathtag_reduce":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.PathTagReduce}
		case "pathtag_reduce2":
			cpus = cpuShader{kind: cpuShaderKindSkipped}
		case "pathtag_scan1":
			cpus = cpuShader{kind: cpuShaderKindSkipped}
		case "pathtag_scan_large":
			cpus = cpuShader{kind: cpuShaderKindSkipped}
		case "pathtag_scan_small":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.PathTagScan}
		case "tile_alloc":
			cpus = cpuShader{kind: cpuShaderKindPresent, shader: cpu.TileAlloc}
		}

		id := engine.prepareShader(shader.Name, shader.WGSL.Code, bindings, cpus)
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
		Format:        wgpu.TextureFormatRGBA16Float,
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
	case renderer.Rgba8Srgb:
		return wgpu.TextureFormatRGBA8UnormSrgb
	case renderer.Rgba16Float:
		return wgpu.TextureFormatRGBA16Float
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
) *wgpu.CommandBuffer {
	pgroup = pgroup.Nest("RenderToTexture")
	defer pgroup.End()

	recording, target := eng.renderer.RenderFull(arena, enc, eng.resolver, eng.fullShaders, params, pgroup)

	externalResources := []ExternalResource{
		ExternalImage{
			Proxy: target.ImageProxy,
			View:  texture,
		},
	}
	return eng.RunRecording(arena, queue, recording, externalResources, "render_to_texture", pgroup)
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
	profileCmds := ency.Finish(nil)
	defer profileCmds.Release()

	renderCmds := eng.RenderToTexture(arena, queue, enc, eng.target.View, params, pgroup)
	defer renderCmds.Release()

	surfaceView := surface.Texture.CreateView(nil)
	defer surfaceView.Release()

	bindGroup := eng.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: eng.blit.BindLayout,
		Entries: mem.MakeSlice(arena, []wgpu.BindGroupEntry{
			{
				Binding:     0,
				TextureView: eng.target.View,
			},
		}),
	})
	defer bindGroup.Release()

	encoder := eng.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "blitter"})
	defer encoder.Release()
	renderPass := encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: mem.MakeSlice(arena, []wgpu.RenderPassColorAttachment{
			{
				View:       surfaceView,
				LoadOp:     wgpu.LoadOpClear,
				StoreOp:    wgpu.StoreOpStore,
				ClearValue: wgpu.Color{R: 0, G: 255, B: 0, A: 255},
			},
		}),
		TimestampWrites: pgroup.Render(arena, "blit"),
	})
	defer renderPass.Release()

	renderPass.SetPipeline(eng.blit.Pipeline)
	renderPass.SetBindGroup(0, bindGroup, nil)
	renderPass.Draw(6, 1, 0, 0)
	renderPass.End()

	span.End(encoder)
	blitCmds := encoder.Finish(nil)
	defer blitCmds.Release()
	queue.Submit(mem.Varargs(arena, profileCmds, renderCmds, blitCmds)...)

}
