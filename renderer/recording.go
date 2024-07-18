package renderer

import (
	"fmt"
	"sync/atomic"

	"honnef.co/go/jello/mem"
)

var resourceID atomic.Uint64

func nextResourceID() ResourceID {
	return ResourceID(resourceID.Add(1))
}

type ResourceID uint64

type ResourceProxyKind int

const (
	ResourceProxyKindBuffer ResourceProxyKind = iota + 1
	ResourceProxyKindImage
	ResourceProxyKindImageArray
)

type ResourceProxy struct {
	Kind ResourceProxyKind
	BufferProxy
	ImageProxy
	ImageArray []ImageProxy
}

type Recording struct {
	Commands []Command
}

func (rec *Recording) push(arena *mem.Arena, cmd Command) {
	rec.Commands = mem.Append(arena, rec.Commands, cmd)
}

func (rec *Recording) Upload(arena *mem.Arena, name string, data []byte) BufferProxy {
	buf := NewBufferProxy(uint64(len(data)), name)
	rec.push(arena, mem.Make(arena, Upload{buf, data}))
	return buf
}

func (rec *Recording) UploadUniform(arena *mem.Arena, name string, data []byte) BufferProxy {
	buf := NewBufferProxy(uint64(len(data)), name)
	rec.push(arena, mem.Make(arena, UploadUniform{buf, data}))
	return buf
}

func (rec *Recording) UploadImage(arena *mem.Arena, width, height uint32, format ImageFormat, data []byte) ImageProxy {
	imageProxy := NewImageProxy(width, height, format)
	rec.push(arena, mem.Make(arena, UploadImage{imageProxy, data}))
	return imageProxy
}

func (rec *Recording) Dispatch(arena *mem.Arena, shader ShaderID, wgSize [3]uint32, resources []ResourceProxy) {
	rec.push(arena, mem.Make(arena, Dispatch{shader, wgSize, resources}))
}

func (rec *Recording) DispatchIndirect(
	arena *mem.Arena,
	shader ShaderID,
	buf BufferProxy,
	offset uint64,
	resources []ResourceProxy,
) {
	rec.push(arena, mem.Make(arena, DispatchIndirect{shader, buf, offset, resources}))
}

func (rec *Recording) Download(arena *mem.Arena, buf BufferProxy) {
	rec.push(arena, mem.Make(arena, Download{buf}))
}

func (rec *Recording) ClearAll(arena *mem.Arena, buf BufferProxy) {
	rec.push(arena, mem.Make(arena, Clear{buf, 0, -1}))
}

func (rec *Recording) FreeBuffer(arena *mem.Arena, buf BufferProxy) {
	rec.push(arena, mem.Make(arena, FreeBuffer{buf}))
}

func (rec *Recording) FreeImage(arena *mem.Arena, image ImageProxy) {
	rec.push(arena, mem.Make(arena, FreeImage{image}))
}

func (rec *Recording) FreeResource(arena *mem.Arena, resource ResourceProxy) {
	switch resource.Kind {
	case ResourceProxyKindBuffer:
		rec.FreeBuffer(arena, resource.BufferProxy)
	case ResourceProxyKindImage:
		rec.FreeImage(arena, resource.ImageProxy)
	default:
		panic(fmt.Sprintf("unhandled type %T", resource))
	}
}

func NewBufferProxy(size uint64, name string) BufferProxy {
	id := nextResourceID()
	return BufferProxy{size, id, name}
}

func NewImageProxy(width, height uint32, format ImageFormat) ImageProxy {
	id := nextResourceID()
	return ImageProxy{
		Width:  width,
		Height: height,
		Format: format,
		ID:     id,
	}
}

type BufferProxy struct {
	Size uint64
	ID   ResourceID
	Name string
}

func (p BufferProxy) Resource() ResourceProxy {
	return ResourceProxy{
		Kind:        ResourceProxyKindBuffer,
		BufferProxy: p,
	}
}

type ImageFormat int

const (
	Rgba8 ImageFormat = iota
	Rgba8Srgb
	Bgra8
)

type ImageProxy struct {
	Width  uint32
	Height uint32
	Format ImageFormat
	ID     ResourceID
}

func (p ImageProxy) Resource() ResourceProxy {
	return ResourceProxy{
		Kind:       ResourceProxyKindImage,
		ImageProxy: p,
	}
}

type ShaderID int

type Command interface {
	isCommand()
}

func (*Upload) isCommand()           {}
func (*UploadUniform) isCommand()    {}
func (*UploadImage) isCommand()      {}
func (*WriteImage) isCommand()       {}
func (*Dispatch) isCommand()         {}
func (*DispatchIndirect) isCommand() {}
func (*Download) isCommand()         {}
func (*Clear) isCommand()            {}
func (*FreeBuffer) isCommand()       {}
func (*FreeImage) isCommand()        {}

type BindTypeType int

const (
	BindTypeBuffer BindTypeType = iota + 1
	BindTypeBufReadOnly
	BindTypeUniform
	BindTypeImage
	BindTypeImageRead
	BindTypeImageArrayRead
)

type BindType struct {
	Type        BindTypeType
	ImageFormat ImageFormat
}

type Upload struct {
	Buffer BufferProxy
	Data   []byte
}

type UploadUniform struct {
	Buffer BufferProxy
	Data   []byte
}

type UploadImage struct {
	Image ImageProxy
	Data  []byte
}

type WriteImage struct {
	Image  ImageProxy
	Coords [4]uint32
	Data   []byte
}

type Dispatch struct {
	Shader        ShaderID
	WorkgroupSize [3]uint32
	Bindings      []ResourceProxy
}

type DispatchIndirect struct {
	Shader   ShaderID
	Buffer   BufferProxy
	Offset   uint64
	Bindings []ResourceProxy
}

type Download struct {
	Buffer BufferProxy
}

type Clear struct {
	Buffer BufferProxy
	Offset uint64
	Size   int64
}

type FreeBuffer struct {
	Buffer BufferProxy
}

type FreeImage struct {
	Image ImageProxy
}
