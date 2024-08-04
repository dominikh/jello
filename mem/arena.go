// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package mem

import (
	"cmp"
	"iter"
	"reflect"
	"sort"
	"unsafe"

	"golang.org/x/exp/constraints"
)

func NewArena() *Arena {
	return &Arena{
		typedSlabs: make(map[reflect.Type][]slab),
	}
}

func New[T any](a *Arena) *T {
	var t *T
	// We cannot use TypeOf(*new(T)) when T is an interface type, because that
	// passes a nil interface to TypeOf, which returns nil.
	typ := reflect.TypeOf(t).Elem()
	return (*T)(a.alloc(typ, 1))
}

func Make[T any](a *Arena, v T) *T {
	ptr := New[T](a)
	*ptr = v
	return ptr
}

func NewSlice[T ~[]E, E any](a *Arena, len, cap int) T {
	if cap == 0 {
		return nil
	}
	// We cannot use TypeOf(*new(T)) when T is an interface type, because that
	// passes a nil interface to TypeOf, which returns nil.
	var e *E
	ptr := a.alloc(reflect.TypeOf(e).Elem(), cap)
	return T(unsafe.Slice((*E)(ptr), cap)[:len])
}

func MakeSlice[T ~[]E, E any](a *Arena, values T) T {
	// MakeSlice inlines, which means that MakeSlice(a, []T{...}) won't have to
	// allocate to pass the values to us.
	s := NewSlice[T, E](a, len(values), len(values))
	copy(s, values)
	return s
}

func Varargs[E any](a *Arena, values ...E) []E {
	return MakeSlice[[]E, E](a, values)
}

func Append[T ~[]E, E any](a *Arena, s T, data ...E) T {
	s = growSlice(a, s, len(data))
	s = append(s, data...)
	return s
}

func Grow[T ~[]E, E any](a *Arena, s T, n int) T {
	if n -= cap(s) - len(s); n > 0 {
		s = growSlice(a, s, n)
	}
	return s
}

func growSlice[T ~[]E, E any](a *Arena, s T, n int) T {
	const growThreshold = 256
	newLen := len(s) + n
	newCap := cap(s)

	if newCap > 0 {
		for newLen > newCap {
			if newCap < growThreshold {
				newCap *= 2
			} else {
				newCap += newCap / 4
			}
		}
	} else {
		newCap = n
	}
	if newCap == cap(s) {
		return s
	}
	s2 := NewSlice[T, E](a, len(s), newCap)
	copy(s2, s)
	return s2
}

type BinaryTreeMap[K constraints.Ordered, V any] struct {
	entries []BinaryTreeMapEntry[K, V]
}

type BinaryTreeMapEntry[K constraints.Ordered, V any] struct {
	key     K
	value   V
	deleted bool
}

func (m *BinaryTreeMap[K, V]) find(key K) (*BinaryTreeMapEntry[K, V], bool) {
	idx, ok := sort.Find(len(m.entries), func(i int) int {
		return cmp.Compare(key, m.entries[i].key)
	})
	if !ok {
		return nil, false
	}
	return &m.entries[idx], true
}

func (m *BinaryTreeMap[K, V]) Insert(a *Arena, key K, value V) {
	idx := sort.Search(len(m.entries), func(i int) bool {
		return key <= m.entries[i].key
	})

	if idx == len(m.entries) || m.entries[idx].key != key {
		m.entries = insert(a, m.entries, idx, BinaryTreeMapEntry[K, V]{key, value, false})
	} else {
		e := &m.entries[idx]
		e.value = value
		e.deleted = false
	}
}

func (m *BinaryTreeMap[K, V]) Get(key K) (V, bool) {
	if e, ok := m.find(key); ok {
		return e.value, true
	} else {
		return *new(V), false
	}
}

func (m *BinaryTreeMap[K, V]) Delete(key K) bool {
	if e, ok := m.find(key); ok {
		wasDeleted := e.deleted
		e.deleted = true
		return !wasDeleted
	} else {
		return false
	}
}

func (m *BinaryTreeMap[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for _, e := range m.entries {
			if e.deleted {
				continue
			}
			if !yield(e.key, e.value) {
				return
			}
		}
	}
}

func (m *BinaryTreeMap[K, V]) Keys() iter.Seq[K] {
	return func(yield func(K) bool) {
		for _, e := range m.entries {
			if e.deleted {
				continue
			}
			if !yield(e.key) {
				return
			}
		}
	}
}

func (m *BinaryTreeMap[K, V]) Values() iter.Seq[V] {
	return func(yield func(V) bool) {
		for _, e := range m.entries {
			if e.deleted {
				continue
			}
			if !yield(e.value) {
				return
			}
		}
	}
}

func insert[S ~[]E, E any](a *Arena, s S, i int, v E) S {
	if i == len(s) {
		return Append(a, s, v)
	}

	if cap(s) > len(s)+1 {
		s = s[:len(s)+1]
		copy(s[i+1:], s[i:])
		s[i] = v
		return s
	} else {
		s2 := NewSlice[S](a, len(s)+1, (len(s)+1)*2)
		copy(s2, s[:i])
		s2[i] = v
		copy(s2[i+1:], s[i:])
		return s2
	}
}

type Arena struct {
	byteSlabs  []slab
	typedSlabs map[reflect.Type][]slab
}

const slabSize = 1024 * 1024

func (a *Arena) alloc(typ reflect.Type, num int) unsafe.Pointer {
	type iface struct {
		_    unsafe.Pointer
		rtyp *struct {
			size      int
			ptrPrefix int
			_         uint32
			_         uint8
			align     uint8
		}
	}

	// FIXME(dh): guard against types that are larger than a whole slab

	rtyp := (*iface)(unsafe.Pointer(&typ)).rtyp
	// rtyp.size already includes padding
	totalSize := num * rtyp.size
	if rtyp.ptrPrefix == 0 {
		// OPT(dh): skip full slabs
		for i := range a.byteSlabs {
			sl := &a.byteSlabs[i]
			off := align(sl.offset, rtyp.align)
			if sl.size-off >= totalSize {
				// Found slab
				sl.offset = off + totalSize
				ptr := unsafe.Add(sl.data, off)
				// Return zeroed memory
				clear(unsafe.Slice((*byte)(ptr), totalSize))
				return ptr
			}
		}
		// Need a new slab
		a.byteSlabs = append(a.byteSlabs, slab{
			data: unsafe.Pointer(unsafe.SliceData(make([]byte, slabSize))),
			size: slabSize,
		})
		sl := &a.byteSlabs[len(a.byteSlabs)-1]
		sl.offset = totalSize
		return sl.data
	} else {
		slabs := a.typedSlabs[typ]
		// OPT(dh): skip full slabs
		for i := range slabs {
			sl := &slabs[i]
			off := align(sl.offset, rtyp.align)
			if sl.size-off >= totalSize {
				// Found slab
				sl.offset = off + totalSize
				// No need to zero memory here, we do it for typed slabs when
				// resetting the arena.
				return unsafe.Add(sl.data, off)
			}
		}
		// Need a new slab
		sz := slabSize / rtyp.size
		ptr := reflect.MakeSlice(reflect.SliceOf(typ), sz, sz).UnsafePointer()
		sl := slab{
			data:   ptr,
			size:   sz,
			offset: totalSize,
		}
		a.typedSlabs[typ] = append(a.typedSlabs[typ], sl)
		return ptr
	}
}

// to has to be a power of two.
func align(v int, to uint8) int {
	return v + (-v & (int(to) - 1))
}

func (a *Arena) Reset() {
	if a.typedSlabs == nil {
		a.typedSlabs = make(map[reflect.Type][]slab)
	}
	for i := range a.byteSlabs {
		a.byteSlabs[i].offset = 0
	}
	for _, slabs := range a.typedSlabs {
		for i := range slabs {
			slab := &slabs[i]
			// Clear memory so it doesn't keep Go pointers alive
			clear(unsafe.Slice((*byte)(slab.data), slab.offset))
			slab.offset = 0
		}
	}
}

type slab struct {
	data   unsafe.Pointer
	size   int
	offset int
}

func hasPtrs(typ reflect.Type) bool {
	type iface struct {
		_    unsafe.Pointer
		rtyp *struct {
			size      uintptr
			ptrPrefix uintptr
		}
	}

	return (*iface)(unsafe.Pointer(&typ)).rtyp.ptrPrefix != 0
}
