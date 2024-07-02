package jello

import (
	"fmt"
	"reflect"

	"honnef.co/go/jello/shaders"
	"honnef.co/go/wgpu"
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

var bindTypeMapping = [...]BindType{
	shaders.Buffer:      BindType{Type: BindTypeBuffer},
	shaders.BufReadOnly: BindType{Type: BindTypeBufReadOnly},
	shaders.Uniform:     BindType{Type: BindTypeUniform},
	shaders.Image:       BindType{Type: BindTypeImage, ImageFormat: Rgba8},
	shaders.ImageRead:   BindType{Type: BindTypeImageRead, ImageFormat: Rgba8},
}

func NewFullShaders(dev *wgpu.Device, engine *WGPUEngine, options *RendererOptions) *FullShaders {
	// XXX make use of options.AntialiasingSupport
	// XXX support CPU shaders
	var out FullShaders
	outV := reflect.ValueOf(&out).Elem()
	v := reflect.ValueOf(&shaders.Collection)
	for i := range v.Elem().NumField() {
		fieldName := v.Elem().Type().Field(i).Name
		outField := outV.FieldByName(fieldName)
		if !outField.IsValid() {
			continue
		}
		shader := v.Elem().Field(i).Addr().Interface().(*shaders.ComputeShader)
		bindings := make([]BindType, len(shader.Bindings))
		for i, b := range shader.Bindings {
			bindings[i] = bindTypeMapping[b]
		}
		if len(shader.WGSL.Code) == 0 {
			panic(fmt.Sprintf("shader %q has no code", shader.Name))
		}
		id := engine.AddShader(dev, shader.Name, shader.WGSL.Code, bindings, nil)
		outField.Set(reflect.ValueOf(id))
	}
	return &out
}
