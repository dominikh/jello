package jello

import (
	"honnef.co/go/safeish"
)

type Render struct {
	fine_wg_count  option[WorkgroupSize]
	fine_resources option[fineResources]
	mask_buf       option[ResourceProxy]
}

type fineResources struct {
	aa_config AaConfig

	config_buf        ResourceProxy
	bump_buf          ResourceProxy
	tile_buf          ResourceProxy
	segments_buf      ResourceProxy
	ptcl_buf          ResourceProxy
	gradient_image    ResourceProxy
	info_bin_data_buf ResourceProxy
	image_atlas       ResourceProxy

	out_image ImageProxy
}

func (self *Render) RenderEncodingCoarse(
	encoding *Encoding,
	resolver *Resolver,
	shaders *FullShaders,
	params *RenderParams,
	robust bool,
) *Recording {
	var recording Recording
	layout, ramps, images, packed := resolver.Resolve(encoding, nil)
	var gradient_image ResourceProxy
	if ramps.Height == 0 {
		gradient_image = NewImageProxy(1, 1, Rgba8)
	} else {
		panic("gradients not implemented")
	}
	var image_atlas ImageProxy
	if len(images.Images) == 0 {
		image_atlas = NewImageProxy(1, 1, Rgba8)
	} else {
		panic("images not implemented")
	}
	// XXX write images to atlas

	cpu_config := NewRenderConfig(&layout, params.Width, params.Height, params.BaseColor)
	buffer_sizes := &cpu_config.buffer_sizes
	wg_counts := &cpu_config.workgroup_counts

	scene_buf := recording.Upload("scene", packed)
	config_buf := recording.UploadUniform("config", safeish.AsBytes(&cpu_config.gpu))
	info_bin_data_buf := NewBufferProxy(
		uint64(buffer_sizes.Bin_data.size_in_bytes()),
		"info_bin_data_buf",
	)
	tile_buf := NewBufferProxy(uint64(buffer_sizes.Tiles.size_in_bytes()), "tile_buf")
	segments_buf := NewBufferProxy(uint64(buffer_sizes.Segments.size_in_bytes()), "segments_buf")
	ptcl_buf := NewBufferProxy(uint64(buffer_sizes.Ptcl.size_in_bytes()), "ptcl_buf")
	reduced_buf := NewBufferProxy(
		uint64(buffer_sizes.Path_reduced.size_in_bytes()),
		"reduced_buf",
	)
	// TODO: really only need pathtag_wgs - 1
	recording.Dispatch(
		shaders.PathtagReduce,
		wg_counts.Path_reduce,
		[]ResourceProxy{config_buf, scene_buf, reduced_buf},
	)
	pathtag_parent := reduced_buf
	var large_pathtag_bufs option[[2]ResourceProxy]
	use_large_path_scan := wg_counts.Use_large_path_scan && !shaders.PathtagIsCPU
	if use_large_path_scan {
		reduced2_buf := NewBufferProxy(
			uint64(buffer_sizes.Path_reduced2.size_in_bytes()),
			"reduced2_buf",
		)
		recording.Dispatch(
			shaders.PathtagReduce2,
			wg_counts.Path_reduce2,
			[]ResourceProxy{reduced_buf, reduced2_buf},
		)
		reduced_scan_buf := NewBufferProxy(
			uint64(buffer_sizes.Path_reduced_scan.size_in_bytes()),
			"reduced_scan_buf",
		)
		recording.Dispatch(
			shaders.PathtagScan1,
			wg_counts.Path_scan1,
			[]ResourceProxy{reduced_buf, reduced2_buf, reduced_scan_buf},
		)
		pathtag_parent = reduced_scan_buf
		large_pathtag_bufs.set([2]ResourceProxy{reduced2_buf, reduced_scan_buf})
	}

	tagmonoid_buf := NewBufferProxy(
		uint64(buffer_sizes.Path_monoids.size_in_bytes()),
		"tagmonoid_buf",
	)
	var pathtag_scan ShaderID
	if use_large_path_scan {
		pathtag_scan = shaders.PathtagScanLarge
	} else {
		pathtag_scan = shaders.PathtagScanSmall
	}
	recording.Dispatch(
		pathtag_scan,
		wg_counts.Path_scan,
		[]ResourceProxy{config_buf, scene_buf, pathtag_parent, tagmonoid_buf},
	)
	recording.FreeResource(reduced_buf)
	if large_pathtag_bufs.isSet {
		recording.FreeResource(large_pathtag_bufs.value[0])
		recording.FreeResource(large_pathtag_bufs.value[1])
	}
	path_bbox_buf := NewBufferProxy(
		uint64(buffer_sizes.Path_bboxes.size_in_bytes()),
		"path_bbox_buf",
	)
	recording.Dispatch(
		shaders.BboxClear,
		wg_counts.Bbox_clear,
		[]ResourceProxy{config_buf, path_bbox_buf},
	)
	bump_buf := NewBufferProxy(uint64(buffer_sizes.Bump_alloc.size_in_bytes()), "bump_buf")
	recording.ClearAll(bump_buf)
	lines_buf := NewBufferProxy(uint64(buffer_sizes.Lines.size_in_bytes()), "lines_buf")
	recording.Dispatch(
		shaders.Flatten,
		wg_counts.Flatten,
		[]ResourceProxy{
			config_buf,
			scene_buf,
			tagmonoid_buf,
			path_bbox_buf,
			bump_buf,
			lines_buf,
		},
	)
	draw_reduced_buf := NewBufferProxy(
		uint64(buffer_sizes.Draw_reduced.size_in_bytes()),
		"draw_reduced_buf",
	)
	recording.Dispatch(
		shaders.DrawReduce,
		wg_counts.Draw_reduce,
		[]ResourceProxy{config_buf, scene_buf, draw_reduced_buf},
	)
	draw_monoid_buf := NewBufferProxy(
		uint64(buffer_sizes.Draw_monoids.size_in_bytes()),
		"draw_monoid_buf",
	)
	clip_inp_buf := NewBufferProxy(
		uint64(buffer_sizes.Clip_inps.size_in_bytes()),
		"clip_inp_buf",
	)
	recording.Dispatch(
		shaders.DrawLeaf,
		wg_counts.Draw_leaf,
		[]ResourceProxy{
			config_buf,
			scene_buf,
			draw_reduced_buf,
			path_bbox_buf,
			draw_monoid_buf,
			info_bin_data_buf,
			clip_inp_buf,
		},
	)
	recording.FreeResource(draw_reduced_buf)
	clip_el_buf := NewBufferProxy(uint64(buffer_sizes.Clip_els.size_in_bytes()), "clip_el_buf")
	clip_bic_buf := NewBufferProxy(
		uint64(buffer_sizes.Clip_bics.size_in_bytes()),
		"clip_bic_buf",
	)
	if wg_counts.Clip_reduce[0] > 0 {
		recording.Dispatch(
			shaders.ClipReduce,
			wg_counts.Clip_reduce,
			[]ResourceProxy{clip_inp_buf, path_bbox_buf, clip_bic_buf, clip_el_buf},
		)
	}
	clip_bbox_buf := NewBufferProxy(
		uint64(buffer_sizes.Clip_bboxes.size_in_bytes()),
		"clip_bbox_buf",
	)
	if wg_counts.Clip_leaf[0] > 0 {
		recording.Dispatch(
			shaders.ClipLeaf,
			wg_counts.Clip_leaf,
			[]ResourceProxy{
				config_buf,
				clip_inp_buf,
				path_bbox_buf,
				clip_bic_buf,
				clip_el_buf,
				draw_monoid_buf,
				clip_bbox_buf,
			},
		)
	}
	recording.FreeResource(clip_inp_buf)
	recording.FreeResource(clip_bic_buf)
	recording.FreeResource(clip_el_buf)
	draw_bbox_buf := NewBufferProxy(
		uint64(buffer_sizes.Draw_bboxes.size_in_bytes()),
		"draw_bbox_buf",
	)
	bin_header_buf := NewBufferProxy(
		uint64(buffer_sizes.Bin_headers.size_in_bytes()),
		"bin_header_buf",
	)
	recording.Dispatch(
		shaders.Binning,
		wg_counts.Binning,
		[]ResourceProxy{
			config_buf,
			draw_monoid_buf,
			path_bbox_buf,
			clip_bbox_buf,
			draw_bbox_buf,
			bump_buf,
			info_bin_data_buf,
			bin_header_buf,
		},
	)
	recording.FreeResource(draw_monoid_buf)
	recording.FreeResource(path_bbox_buf)
	recording.FreeResource(clip_bbox_buf)
	// Note: this only needs to be rounded up because of the workaround to store the tile_offset
	// in storage rather than workgroup memory.
	path_buf := NewBufferProxy(uint64(buffer_sizes.Paths.size_in_bytes()), "path_buf")
	recording.Dispatch(
		shaders.TileAlloc,
		wg_counts.Tile_alloc,
		[]ResourceProxy{
			config_buf,
			scene_buf,
			draw_bbox_buf,
			bump_buf,
			path_buf,
			tile_buf,
		},
	)
	recording.FreeResource(draw_bbox_buf)
	recording.FreeResource(tagmonoid_buf)
	indirect_count_buf := NewBufferProxy(
		uint64(buffer_sizes.Indirect_count.size_in_bytes()),
		"indirect_count",
	)
	recording.Dispatch(
		shaders.PathCountSetup,
		wg_counts.Path_count_setup,
		[]ResourceProxy{bump_buf, indirect_count_buf},
	)
	seg_counts_buf := NewBufferProxy(
		uint64(buffer_sizes.Seg_counts.size_in_bytes()),
		"seg_counts_buf",
	)
	recording.DispatchIndirect(
		shaders.PathCount,
		indirect_count_buf,
		0,
		[]ResourceProxy{
			config_buf,
			bump_buf,
			lines_buf,
			path_buf,
			tile_buf,
			seg_counts_buf,
		},
	)
	recording.Dispatch(
		shaders.BackdropDyn,
		wg_counts.Backdrop,
		[]ResourceProxy{config_buf, bump_buf, path_buf, tile_buf},
	)
	recording.Dispatch(
		shaders.Coarse,
		wg_counts.Coarse,
		[]ResourceProxy{
			config_buf,
			scene_buf,
			draw_monoid_buf,
			bin_header_buf,
			info_bin_data_buf,
			path_buf,
			tile_buf,
			bump_buf,
			ptcl_buf,
		},
	)
	recording.Dispatch(
		shaders.PathTilingSetup,
		wg_counts.Path_tiling_setup,
		[]ResourceProxy{bump_buf, indirect_count_buf, ptcl_buf},
	)
	recording.DispatchIndirect(
		shaders.PathTiling,
		indirect_count_buf,
		0,
		[]ResourceProxy{
			bump_buf,
			seg_counts_buf,
			lines_buf,
			path_buf,
			tile_buf,
			segments_buf,
		},
	)
	recording.FreeBuffer(indirect_count_buf)
	recording.FreeResource(seg_counts_buf)
	recording.FreeResource(lines_buf)
	recording.FreeResource(scene_buf)
	recording.FreeResource(draw_monoid_buf)
	recording.FreeResource(bin_header_buf)
	recording.FreeResource(path_buf)
	out_image := NewImageProxy(params.Width, params.Height, Rgba8)
	self.fine_wg_count.set(wg_counts.Fine)
	self.fine_resources.set(fineResources{
		aa_config:         params.AntialiasingMethod,
		config_buf:        config_buf,
		bump_buf:          bump_buf,
		tile_buf:          tile_buf,
		segments_buf:      segments_buf,
		ptcl_buf:          ptcl_buf,
		gradient_image:    gradient_image,
		info_bin_data_buf: info_bin_data_buf,
		image_atlas:       image_atlas,
		out_image:         out_image,
	})
	if robust {
		recording.Download(bump_buf)
	}
	recording.FreeResource(bump_buf)
	return &recording
}

func (self *Render) RecordFine(shaders *FullShaders, recording *Recording) {
	fine_wg_count := self.fine_wg_count.take().unwrap()
	fine := self.fine_resources.take().unwrap()
	switch fine.aa_config {
	case Area:
		recording.Dispatch(
			shaders.FineArea,
			fine_wg_count,
			[]ResourceProxy{
				fine.config_buf,
				fine.segments_buf,
				fine.ptcl_buf,
				fine.info_bin_data_buf,
				fine.out_image,
				fine.gradient_image,
				fine.image_atlas,
			},
		)
	default:
		if !self.mask_buf.isSet {
			var mask_lut []uint8
			switch fine.aa_config {
			case Msaa16:
				mask_lut = make_mask_lut_16()
			case Msaa8:
				mask_lut = make_mask_lut()
			default:
				panic("unreachable")
			}
			buf := recording.Upload("mask lut", mask_lut)
			self.mask_buf.set(buf)
		}
		var fine_shader ShaderID
		switch fine.aa_config {
		case Msaa16:
			fine_shader = shaders.FineMSAA16
		case Msaa8:
			fine_shader = shaders.FineMSAA8
		default:
			panic("unreachable")
		}

		recording.Dispatch(
			fine_shader,
			fine_wg_count,
			[]ResourceProxy{
				fine.config_buf,
				fine.segments_buf,
				fine.ptcl_buf,
				fine.info_bin_data_buf,
				fine.out_image,
				fine.gradient_image,
				fine.image_atlas,
				self.mask_buf.unwrap(),
			},
		)
	}

	recording.FreeResource(fine.config_buf)
	recording.FreeResource(fine.tile_buf)
	recording.FreeResource(fine.segments_buf)
	recording.FreeResource(fine.ptcl_buf)
	recording.FreeResource(fine.gradient_image)
	recording.FreeResource(fine.image_atlas)
	recording.FreeResource(fine.info_bin_data_buf)
	// TODO: make mask buf persistent
	mask_buf := self.mask_buf.take()
	if mask_buf.isSet {
		recording.FreeResource(mask_buf.value)
	}
}

func (r *Render) OutImage() ImageProxy {
	return r.fine_resources.value.out_image
}
