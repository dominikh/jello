package jello

import "structs"

type BinHeader struct {
	_ structs.HostLayout

	ElementCount uint32
	ChunkOffset  uint32
}
