package jello

type CPUBinding interface {
	// One of CPUBuffer, CPUBufferRW, CPUTexture
}

type CPUBuffer []byte
type CPUBufferRW []byte
type CPUTexture struct{}
