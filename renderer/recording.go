package renderer

import (
	"fmt"
	"sync/atomic"
)

var resourceID atomic.Uint64

func nextResourceID() ResourceID {
	return ResourceID(resourceID.Add(1))
}

type ResourceID uint64

type ResourceProxy interface {
	// One of BufferProxy and ImageProxy
	isResourceProxy()
}

type Recording struct {
	Commands []Command
}

func (rec *Recording) push(cmd Command) {
	rec.Commands = append(rec.Commands, cmd)
}

func (rec *Recording) Upload(name string, data []byte) BufferProxy {
	buf := NewBufferProxy(uint64(len(data)), name)
	rec.push(Upload{buf, data})
	return buf
}

func (rec *Recording) UploadUniform(name string, data []byte) BufferProxy {
	buf := NewBufferProxy(uint64(len(data)), name)
	rec.push(UploadUniform{buf, data})
	return buf
}

func (rec *Recording) UploadImage(width, height uint32, format ImageFormat, data []byte) ImageProxy {
	imageProxy := NewImageProxy(width, height, format)
	rec.Commands = append(rec.Commands, UploadImage{imageProxy, data})
	return imageProxy
}

func (rec *Recording) Dispatch(shader ShaderID, wgSize [3]uint32, resources []ResourceProxy) {
	rec.push(Dispatch{shader, wgSize, resources})
}

func (rec *Recording) DispatchIndirect(
	shader ShaderID,
	buf BufferProxy,
	offset uint64,
	resources []ResourceProxy,
) {
	rec.push(DispatchIndirect{shader, buf, offset, resources})
}

func (rec *Recording) Download(buf BufferProxy) {
	rec.push(Download{buf})
}

func (rec *Recording) ClearAll(buf BufferProxy) {
	rec.push(Clear{buf, 0, -1})
}

func (rec *Recording) FreeBuffer(buf BufferProxy) {
	rec.push(FreeBuffer{buf})
}

func (rec *Recording) FreeImage(image ImageProxy) {
	rec.push(FreeImage{image})
}

func (rec *Recording) FreeResource(resource ResourceProxy) {
	switch resource := resource.(type) {
	case BufferProxy:
		rec.FreeBuffer(resource)
	case ImageProxy:
		rec.FreeImage(resource)
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

func (BufferProxy) isResourceProxy() {}

type ImageFormat int

const (
	Rgba8 ImageFormat = iota
	Bgra8
)

type ImageProxy struct {
	Width  uint32
	Height uint32
	Format ImageFormat
	ID     ResourceID
}

func (ImageProxy) isResourceProxy() {}

type ShaderID int

type Command interface {
	isCommand()
}

func (Upload) isCommand()           {}
func (UploadUniform) isCommand()    {}
func (UploadImage) isCommand()      {}
func (WriteImage) isCommand()       {}
func (Dispatch) isCommand()         {}
func (DispatchIndirect) isCommand() {}
func (Download) isCommand()         {}
func (Clear) isCommand()            {}
func (FreeBuffer) isCommand()       {}
func (FreeImage) isCommand()        {}

type BindTypeType int

const (
	BindTypeBuffer BindTypeType = iota + 1
	BindTypeBufReadOnly
	BindTypeUniform
	BindTypeImage
	BindTypeImageRead
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
