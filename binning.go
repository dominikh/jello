package jello

import "structs"

type BinHeader struct {
	_ structs.HostLayout

	Element_count uint32
	Chunk_offset  uint32
}
