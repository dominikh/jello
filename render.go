package jello

import (
	"honnef.co/go/safeish"
)

type Render struct {
	fineWgCount   WorkgroupSize
	fineResources fineResources
	maskBuf       ResourceProxy
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
	encoding *Encoding,
	resolver *Resolver,
	shaders *FullShaders,
	params *RenderParams,
	robust bool,
) *Recording {
	var recording Recording
	layout, ramps, images, packed := resolver.Resolve(encoding, nil)
	var gradientImage ResourceProxy
	if ramps.Height == 0 {
		gradientImage = NewImageProxy(1, 1, Rgba8)
	} else {
		panic("gradients not implemented")
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
		[]ResourceProxy{configBuf, sceneBuf, reducedBuf},
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
			[]ResourceProxy{reducedBuf, reduced2Buf},
		)
		reducedScanBuf := NewBufferProxy(
			uint64(bufferSizes.PathReducedScan.sizeInBytes()),
			"reducedScanBuf",
		)
		recording.Dispatch(
			shaders.PathtagScan1,
			wgCounts.PathScan1,
			[]ResourceProxy{reducedBuf, reduced2Buf, reducedScanBuf},
		)
		pathtagParent = reducedScanBuf
		largePathtagBufs = &[2]ResourceProxy{reduced2Buf, reducedScanBuf}
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
		[]ResourceProxy{configBuf, sceneBuf, pathtagParent, tagmonoidBuf},
	)
	recording.FreeResource(reducedBuf)
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
		[]ResourceProxy{configBuf, pathBboxBuf},
	)
	bumpBuf := NewBufferProxy(uint64(bufferSizes.BumpAlloc.sizeInBytes()), "bumpBuf")
	recording.ClearAll(bumpBuf)
	linesBuf := NewBufferProxy(uint64(bufferSizes.Lines.sizeInBytes()), "linesBuf")
	recording.Dispatch(
		shaders.Flatten,
		wgCounts.Flatten,
		[]ResourceProxy{
			configBuf,
			sceneBuf,
			tagmonoidBuf,
			pathBboxBuf,
			bumpBuf,
			linesBuf,
		},
	)
	drawReducedBuf := NewBufferProxy(
		uint64(bufferSizes.DrawReduced.sizeInBytes()),
		"drawReducedBuf",
	)
	recording.Dispatch(
		shaders.DrawReduce,
		wgCounts.DrawReduce,
		[]ResourceProxy{configBuf, sceneBuf, drawReducedBuf},
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
			configBuf,
			sceneBuf,
			drawReducedBuf,
			pathBboxBuf,
			drawMonoidBuf,
			infoBinDataBuf,
			clipInpBuf,
		},
	)
	recording.FreeResource(drawReducedBuf)
	clipElBuf := NewBufferProxy(uint64(bufferSizes.ClipEls.sizeInBytes()), "clipElBuf")
	clipBicBuf := NewBufferProxy(
		uint64(bufferSizes.ClipBics.sizeInBytes()),
		"clipBicBuf",
	)
	if wgCounts.ClipReduce[0] > 0 {
		recording.Dispatch(
			shaders.ClipReduce,
			wgCounts.ClipReduce,
			[]ResourceProxy{clipInpBuf, pathBboxBuf, clipBicBuf, clipElBuf},
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
				configBuf,
				clipInpBuf,
				pathBboxBuf,
				clipBicBuf,
				clipElBuf,
				drawMonoidBuf,
				clipBboxBuf,
			},
		)
	}
	recording.FreeResource(clipInpBuf)
	recording.FreeResource(clipBicBuf)
	recording.FreeResource(clipElBuf)
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
			configBuf,
			drawMonoidBuf,
			pathBboxBuf,
			clipBboxBuf,
			drawBboxBuf,
			bumpBuf,
			infoBinDataBuf,
			binHeaderBuf,
		},
	)
	recording.FreeResource(drawMonoidBuf)
	recording.FreeResource(pathBboxBuf)
	recording.FreeResource(clipBboxBuf)
	// Note: this only needs to be rounded up because of the workaround to store the tileOffset
	// in storage rather than workgroup memory.
	pathBuf := NewBufferProxy(uint64(bufferSizes.Paths.sizeInBytes()), "pathBuf")
	recording.Dispatch(
		shaders.TileAlloc,
		wgCounts.TileAlloc,
		[]ResourceProxy{
			configBuf,
			sceneBuf,
			drawBboxBuf,
			bumpBuf,
			pathBuf,
			tileBuf,
		},
	)
	recording.FreeResource(drawBboxBuf)
	recording.FreeResource(tagmonoidBuf)
	indirectCountBuf := NewBufferProxy(
		uint64(bufferSizes.IndirectCount.sizeInBytes()),
		"indirectCount",
	)
	recording.Dispatch(
		shaders.PathCountSetup,
		wgCounts.PathCountSetup,
		[]ResourceProxy{bumpBuf, indirectCountBuf},
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
			configBuf,
			bumpBuf,
			linesBuf,
			pathBuf,
			tileBuf,
			segCountsBuf,
		},
	)
	recording.Dispatch(
		shaders.BackdropDyn,
		wgCounts.Backdrop,
		[]ResourceProxy{configBuf, bumpBuf, pathBuf, tileBuf},
	)
	recording.Dispatch(
		shaders.Coarse,
		wgCounts.Coarse,
		[]ResourceProxy{
			configBuf,
			sceneBuf,
			drawMonoidBuf,
			binHeaderBuf,
			infoBinDataBuf,
			pathBuf,
			tileBuf,
			bumpBuf,
			ptclBuf,
		},
	)
	recording.Dispatch(
		shaders.PathTilingSetup,
		wgCounts.PathTilingSetup,
		[]ResourceProxy{bumpBuf, indirectCountBuf, ptclBuf},
	)
	recording.DispatchIndirect(
		shaders.PathTiling,
		indirectCountBuf,
		0,
		[]ResourceProxy{
			bumpBuf,
			segCountsBuf,
			linesBuf,
			pathBuf,
			tileBuf,
			segmentsBuf,
		},
	)
	recording.FreeBuffer(indirectCountBuf)
	recording.FreeResource(segCountsBuf)
	recording.FreeResource(linesBuf)
	recording.FreeResource(sceneBuf)
	recording.FreeResource(drawMonoidBuf)
	recording.FreeResource(binHeaderBuf)
	recording.FreeResource(pathBuf)
	outImage := NewImageProxy(params.Width, params.Height, Rgba8)
	r.fineWgCount = wgCounts.Fine
	r.fineResources = fineResources{
		aaConfig:       params.AntialiasingMethod,
		configBuf:      configBuf,
		bumpBuf:        bumpBuf,
		tileBuf:        tileBuf,
		segmentsBuf:    segmentsBuf,
		ptclBuf:        ptclBuf,
		gradientImage:  gradientImage,
		infoBinDataBuf: infoBinDataBuf,
		imageAtlas:     imageAtlas,
		outImage:       outImage,
	}
	if robust {
		recording.Download(bumpBuf)
	}
	recording.FreeResource(bumpBuf)
	return &recording
}

func (r *Render) RecordFine(shaders *FullShaders, recording *Recording) {
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
				fine.outImage,
				fine.gradientImage,
				fine.imageAtlas,
			},
		)
	default:
		if r.maskBuf == nil {
			var maskLUT []uint8
			switch fine.aaConfig {
			case Msaa16:
				maskLUT = makeMaskLUT16()
			case Msaa8:
				maskLUT = makeMaskLUT()
			default:
				panic("unreachable")
			}
			buf := recording.Upload("mask lut", maskLUT)
			r.maskBuf = buf
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
				fine.outImage,
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
	if r.maskBuf != nil {
		recording.FreeResource(r.maskBuf)
		r.maskBuf = nil
	}
}

func (r *Render) OutImage() ImageProxy {
	return r.fineResources.outImage
}
