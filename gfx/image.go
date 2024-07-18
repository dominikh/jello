package gfx

import "image"

type Image struct {
	Image  image.Image
	Extend Extend
}
