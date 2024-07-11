package wgpu_engine

type cpuBinding interface {
	// One of CPUBuffer, CPUBufferRW, CPUTexture
}

type cpuBuffer []byte
type cpuBufferRW []byte
type cpuTexture struct{}
