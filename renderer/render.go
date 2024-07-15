package renderer

import (
	"honnef.co/go/jello/encoding"
	"honnef.co/go/jello/gfx"
	"honnef.co/go/jello/profiler"
	"honnef.co/go/safeish"
)

type FullShaders struct {
	PathtagReduce    ShaderID
	PathtagReduce2   ShaderID
	PathtagScan1     ShaderID
	PathtagScanSmall ShaderID
	PathtagScanLarge ShaderID
	BboxClear        ShaderID
	Flatten          ShaderID
	DrawReduce       ShaderID
	DrawLeaf         ShaderID
	ClipReduce       ShaderID
	ClipLeaf         ShaderID
	Binning          ShaderID
	TileAlloc        ShaderID
	BackdropDyn      ShaderID
	PathCountSetup   ShaderID
	PathCount        ShaderID
	Coarse           ShaderID
	PathTilingSetup  ShaderID
	PathTiling       ShaderID
	FineArea         ShaderID
	FineMSAA8        ShaderID
	FineMSAA16       ShaderID
	// 2-level dispatch works for CPU pathtag scan even for large
	// inputs 3-level is not yet implemented.
	PathtagIsCPU bool
}

type Render struct {
	fineWgCount   WorkgroupSize
	fineResources fineResources
	maskBuf       ResourceProxy
}

type AaConfig int

const (
	Area AaConfig = iota
	Msaa8
	Msaa16
)

type RenderParams struct {
	BaseColor          gfx.Color
	Width              uint32
	Height             uint32
	AntialiasingMethod AaConfig
}

type fineResources struct {
	aaConfig AaConfig

	configBuf      ResourceProxy
	bumpBuf        ResourceProxy
	tileBuf        ResourceProxy
	segmentsBuf    ResourceProxy
	ptclBuf        ResourceProxy
	gradientImage  ResourceProxy
	infoBinDataBuf ResourceProxy
	imageAtlas     ResourceProxy

	outImage ImageProxy
}

func (r *Render) RenderEncodingCoarse(
	encoding *encoding.Encoding,
	resolver *Resolver,
	shaders *FullShaders,
	params *RenderParams,
	robust bool,
	pgroup profiler.ProfilerGroup,
) *Recording {
	pgroup = pgroup.Start("RenderEncodingCoarse")
	defer pgroup.End()

	var recording Recording
	layout, ramps, images, packed := resolver.Resolve(encoding, nil)
	var gradientImage ImageProxy
	if ramps.Height == 0 {
		gradientImage = NewImageProxy(1, 1, Rgba8)
	} else {
		data := safeish.SliceCast[[]byte](ramps.Data)
		gradientImage = recording.UploadImage(
			ramps.Width,
			ramps.Height,
			Rgba8,
			data,
		)
	}
	var imageAtlas ImageProxy
	if len(images.Images) == 0 {
		imageAtlas = NewImageProxy(1, 1, Rgba8)
	} else {
		panic("images not implemented")
	}
	// XXX write images to atlas

	cpuConfig := NewRenderConfig(&layout, params.Width, params.Height, params.BaseColor)
	bufferSizes := &cpuConfig.bufferSizes
	wgCounts := &cpuConfig.workgroupCounts

	sceneBuf := recording.Upload("scene", packed)
	configBuf := recording.UploadUniform("config", safeish.AsBytes(&cpuConfig.gpu))
	infoBinDataBuf := NewBufferProxy(
		uint64(bufferSizes.BinData.sizeInBytes()),
		"infoBinDataBuf",
	)
	tileBuf := NewBufferProxy(uint64(bufferSizes.Tiles.sizeInBytes()), "tileBuf")
	segmentsBuf := NewBufferProxy(uint64(bufferSizes.Segments.sizeInBytes()), "segmentsBuf")
	ptclBuf := NewBufferProxy(uint64(bufferSizes.Ptcl.sizeInBytes()), "ptclBuf")
	reducedBuf := NewBufferProxy(
		uint64(bufferSizes.PathReduced.sizeInBytes()),
		"reducedBuf",
	)
	// TODO: really only need pathtagWgs - 1
	recording.Dispatch(
		shaders.PathtagReduce,
		wgCounts.PathReduce,
		[]ResourceProxy{configBuf.Resource(), sceneBuf.Resource(), reducedBuf.Resource()},
	)
	pathtagParent := reducedBuf
	var largePathtagBufs *[2]ResourceProxy
	useLargePathScan := wgCounts.UseLargePathScan && !shaders.PathtagIsCPU
	if useLargePathScan {
		reduced2Buf := NewBufferProxy(
			uint64(bufferSizes.PathReduced2.sizeInBytes()),
			"reduced2Buf",
		)
		recording.Dispatch(
			shaders.PathtagReduce2,
			wgCounts.PathReduce2,
			[]ResourceProxy{reducedBuf.Resource(), reduced2Buf.Resource()},
		)
		reducedScanBuf := NewBufferProxy(
			uint64(bufferSizes.PathReducedScan.sizeInBytes()),
			"reducedScanBuf",
		)
		recording.Dispatch(
			shaders.PathtagScan1,
			wgCounts.PathScan1,
			[]ResourceProxy{reducedBuf.Resource(), reduced2Buf.Resource(), reducedScanBuf.Resource()},
		)
		pathtagParent = reducedScanBuf
		largePathtagBufs = &[2]ResourceProxy{reduced2Buf.Resource(), reducedScanBuf.Resource()}
	}

	tagmonoidBuf := NewBufferProxy(
		uint64(bufferSizes.PathMonoids.sizeInBytes()),
		"tagmonoidBuf",
	)
	var pathtagScan ShaderID
	if useLargePathScan {
		pathtagScan = shaders.PathtagScanLarge
	} else {
		pathtagScan = shaders.PathtagScanSmall
	}
	recording.Dispatch(
		pathtagScan,
		wgCounts.PathScan,
		[]ResourceProxy{configBuf.Resource(), sceneBuf.Resource(), pathtagParent.Resource(), tagmonoidBuf.Resource()},
	)
	recording.FreeResource(reducedBuf.Resource())
	if largePathtagBufs != nil {
		recording.FreeResource(largePathtagBufs[0])
		recording.FreeResource(largePathtagBufs[1])
	}
	pathBboxBuf := NewBufferProxy(
		uint64(bufferSizes.PathBboxes.sizeInBytes()),
		"pathBboxBuf",
	)
	recording.Dispatch(
		shaders.BboxClear,
		wgCounts.BboxClear,
		[]ResourceProxy{configBuf.Resource(), pathBboxBuf.Resource()},
	)
	bumpBuf := NewBufferProxy(uint64(bufferSizes.BumpAlloc.sizeInBytes()), "bumpBuf")
	recording.ClearAll(bumpBuf)
	linesBuf := NewBufferProxy(uint64(bufferSizes.Lines.sizeInBytes()), "linesBuf")
	recording.Dispatch(
		shaders.Flatten,
		wgCounts.Flatten,
		[]ResourceProxy{
			configBuf.Resource(),
			sceneBuf.Resource(),
			tagmonoidBuf.Resource(),
			pathBboxBuf.Resource(),
			bumpBuf.Resource(),
			linesBuf.Resource(),
		},
	)
	drawReducedBuf := NewBufferProxy(
		uint64(bufferSizes.DrawReduced.sizeInBytes()),
		"drawReducedBuf",
	)
	recording.Dispatch(
		shaders.DrawReduce,
		wgCounts.DrawReduce,
		[]ResourceProxy{configBuf.Resource(), sceneBuf.Resource(), drawReducedBuf.Resource()},
	)
	drawMonoidBuf := NewBufferProxy(
		uint64(bufferSizes.DrawMonoids.sizeInBytes()),
		"drawMonoidBuf",
	)
	clipInpBuf := NewBufferProxy(
		uint64(bufferSizes.ClipInps.sizeInBytes()),
		"clipInpBuf",
	)
	recording.Dispatch(
		shaders.DrawLeaf,
		wgCounts.DrawLeaf,
		[]ResourceProxy{
			configBuf.Resource(),
			sceneBuf.Resource(),
			drawReducedBuf.Resource(),
			pathBboxBuf.Resource(),
			drawMonoidBuf.Resource(),
			infoBinDataBuf.Resource(),
			clipInpBuf.Resource(),
		},
	)
	recording.FreeResource(drawReducedBuf.Resource())
	clipElBuf := NewBufferProxy(uint64(bufferSizes.ClipEls.sizeInBytes()), "clipElBuf")
	clipBicBuf := NewBufferProxy(
		uint64(bufferSizes.ClipBics.sizeInBytes()),
		"clipBicBuf",
	)
	if wgCounts.ClipReduce[0] > 0 {
		recording.Dispatch(
			shaders.ClipReduce,
			wgCounts.ClipReduce,
			[]ResourceProxy{clipInpBuf.Resource(), pathBboxBuf.Resource(), clipBicBuf.Resource(), clipElBuf.Resource()},
		)
	}
	clipBboxBuf := NewBufferProxy(
		uint64(bufferSizes.ClipBboxes.sizeInBytes()),
		"clipBboxBuf",
	)
	if wgCounts.ClipLeaf[0] > 0 {
		recording.Dispatch(
			shaders.ClipLeaf,
			wgCounts.ClipLeaf,
			[]ResourceProxy{
				configBuf.Resource(),
				clipInpBuf.Resource(),
				pathBboxBuf.Resource(),
				clipBicBuf.Resource(),
				clipElBuf.Resource(),
				drawMonoidBuf.Resource(),
				clipBboxBuf.Resource(),
			},
		)
	}
	recording.FreeResource(clipInpBuf.Resource())
	recording.FreeResource(clipBicBuf.Resource())
	recording.FreeResource(clipElBuf.Resource())
	drawBboxBuf := NewBufferProxy(
		uint64(bufferSizes.DrawBboxes.sizeInBytes()),
		"drawBboxBuf",
	)
	binHeaderBuf := NewBufferProxy(
		uint64(bufferSizes.BinHeaders.sizeInBytes()),
		"binHeaderBuf",
	)
	recording.Dispatch(
		shaders.Binning,
		wgCounts.Binning,
		[]ResourceProxy{
			configBuf.Resource(),
			drawMonoidBuf.Resource(),
			pathBboxBuf.Resource(),
			clipBboxBuf.Resource(),
			drawBboxBuf.Resource(),
			bumpBuf.Resource(),
			infoBinDataBuf.Resource(),
			binHeaderBuf.Resource(),
		},
	)
	recording.FreeResource(drawMonoidBuf.Resource())
	recording.FreeResource(pathBboxBuf.Resource())
	recording.FreeResource(clipBboxBuf.Resource())
	// Note: this only needs to be rounded up because of the workaround to store the tileOffset
	// in storage rather than workgroup memory.
	pathBuf := NewBufferProxy(uint64(bufferSizes.Paths.sizeInBytes()), "pathBuf")
	recording.Dispatch(
		shaders.TileAlloc,
		wgCounts.TileAlloc,
		[]ResourceProxy{
			configBuf.Resource(),
			sceneBuf.Resource(),
			drawBboxBuf.Resource(),
			bumpBuf.Resource(),
			pathBuf.Resource(),
			tileBuf.Resource(),
		},
	)
	recording.FreeResource(drawBboxBuf.Resource())
	recording.FreeResource(tagmonoidBuf.Resource())
	indirectCountBuf := NewBufferProxy(
		uint64(bufferSizes.IndirectCount.sizeInBytes()),
		"indirectCount",
	)
	recording.Dispatch(
		shaders.PathCountSetup,
		wgCounts.PathCountSetup,
		[]ResourceProxy{bumpBuf.Resource(), indirectCountBuf.Resource()},
	)
	segCountsBuf := NewBufferProxy(
		uint64(bufferSizes.SegCounts.sizeInBytes()),
		"segCountsBuf",
	)
	recording.DispatchIndirect(
		shaders.PathCount,
		indirectCountBuf,
		0,
		[]ResourceProxy{
			configBuf.Resource(),
			bumpBuf.Resource(),
			linesBuf.Resource(),
			pathBuf.Resource(),
			tileBuf.Resource(),
			segCountsBuf.Resource(),
		},
	)
	recording.Dispatch(
		shaders.BackdropDyn,
		wgCounts.Backdrop,
		[]ResourceProxy{configBuf.Resource(), bumpBuf.Resource(), pathBuf.Resource(), tileBuf.Resource()},
	)
	recording.Dispatch(
		shaders.Coarse,
		wgCounts.Coarse,
		[]ResourceProxy{
			configBuf.Resource(),
			sceneBuf.Resource(),
			drawMonoidBuf.Resource(),
			binHeaderBuf.Resource(),
			infoBinDataBuf.Resource(),
			pathBuf.Resource(),
			tileBuf.Resource(),
			bumpBuf.Resource(),
			ptclBuf.Resource(),
		},
	)
	recording.Dispatch(
		shaders.PathTilingSetup,
		wgCounts.PathTilingSetup,
		[]ResourceProxy{bumpBuf.Resource(), indirectCountBuf.Resource(), ptclBuf.Resource()},
	)
	recording.DispatchIndirect(
		shaders.PathTiling,
		indirectCountBuf,
		0,
		[]ResourceProxy{
			bumpBuf.Resource(),
			segCountsBuf.Resource(),
			linesBuf.Resource(),
			pathBuf.Resource(),
			tileBuf.Resource(),
			segmentsBuf.Resource(),
		},
	)
	recording.FreeBuffer(indirectCountBuf)
	recording.FreeResource(segCountsBuf.Resource())
	recording.FreeResource(linesBuf.Resource())
	recording.FreeResource(sceneBuf.Resource())
	recording.FreeResource(drawMonoidBuf.Resource())
	recording.FreeResource(binHeaderBuf.Resource())
	recording.FreeResource(pathBuf.Resource())
	outImage := NewImageProxy(params.Width, params.Height, Rgba8)
	r.fineWgCount = wgCounts.Fine
	r.fineResources = fineResources{
		aaConfig:       params.AntialiasingMethod,
		configBuf:      configBuf.Resource(),
		bumpBuf:        bumpBuf.Resource(),
		tileBuf:        tileBuf.Resource(),
		segmentsBuf:    segmentsBuf.Resource(),
		ptclBuf:        ptclBuf.Resource(),
		gradientImage:  gradientImage.Resource(),
		infoBinDataBuf: infoBinDataBuf.Resource(),
		imageAtlas:     imageAtlas.Resource(),
		outImage:       outImage,
	}
	if robust {
		recording.Download(bumpBuf)
	}
	recording.FreeResource(bumpBuf.Resource())
	return &recording
}

func (r *Render) RecordFine(shaders *FullShaders, recording *Recording, pgroup profiler.ProfilerGroup) {
	pgroup = pgroup.Start("RecordFine")
	defer pgroup.End()

	fineWgCount := r.fineWgCount
	fine := r.fineResources
	switch fine.aaConfig {
	case Area:
		recording.Dispatch(
			shaders.FineArea,
			fineWgCount,
			[]ResourceProxy{
				fine.configBuf,
				fine.segmentsBuf,
				fine.ptclBuf,
				fine.infoBinDataBuf,
				fine.outImage.Resource(),
				fine.gradientImage,
				fine.imageAtlas,
			},
		)
	default:
		if r.maskBuf.Kind == 0 {
			var maskLUT []uint8
			switch fine.aaConfig {
			case Msaa16:
				maskLUT = maskLUT16
			case Msaa8:
				maskLUT = maskLUT8
			default:
				panic("unreachable")
			}
			buf := recording.Upload("mask lut", maskLUT)
			r.maskBuf = buf.Resource()
		}
		var fineShader ShaderID
		switch fine.aaConfig {
		case Msaa16:
			fineShader = shaders.FineMSAA16
		case Msaa8:
			fineShader = shaders.FineMSAA8
		default:
			panic("unreachable")
		}

		recording.Dispatch(
			fineShader,
			fineWgCount,
			[]ResourceProxy{
				fine.configBuf,
				fine.segmentsBuf,
				fine.ptclBuf,
				fine.infoBinDataBuf,
				fine.outImage.Resource(),
				fine.gradientImage,
				fine.imageAtlas,
				r.maskBuf,
			},
		)
	}

	recording.FreeResource(fine.configBuf)
	recording.FreeResource(fine.tileBuf)
	recording.FreeResource(fine.segmentsBuf)
	recording.FreeResource(fine.ptclBuf)
	recording.FreeResource(fine.gradientImage)
	recording.FreeResource(fine.imageAtlas)
	recording.FreeResource(fine.infoBinDataBuf)
	// TODO: make mask buf persistent
	if r.maskBuf.Kind != 0 {
		recording.FreeResource(r.maskBuf)
		r.maskBuf = ResourceProxy{}
	}
}

func (r *Render) OutImage() ImageProxy {
	return r.fineResources.outImage
}

func RenderFull(
	enc *encoding.Encoding,
	resolver *Resolver,
	shaders *FullShaders,
	params *RenderParams,
	pgroup profiler.ProfilerGroup,
) (*Recording, ResourceProxy) {
	pgroup = pgroup.Start("RenderFull")
	defer pgroup.End()

	var render Render
	recording := render.RenderEncodingCoarse(enc, resolver, shaders, params, false, pgroup)
	outImage := render.OutImage()
	render.RecordFine(shaders, recording, pgroup)
	return recording, outImage.Resource()
}
