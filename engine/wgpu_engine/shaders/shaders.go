package shaders

type BindType int

const (
	Buffer BindType = iota + 1
	BufReadOnly
	Uniform
	Image
	ImageRead
)

func (typ BindType) IsMutable() bool {
	return typ == Buffer || typ == Image
}

type WorkgroupBufferInfo struct {
	size_in_bytes uint32
	index         uint32
}

type ComputeShader struct {
	Name             string
	WorkgroupSize    [3]uint32
	Bindings         []BindType
	WorkgroupBuffers []WorkgroupBufferInfo
	WGSL             WGSLSource
}

type WGSLSource struct {
	Code           []byte
	BindingIndices []uint8
}
