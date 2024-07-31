// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package gfx

import "image"

type Image struct {
	Image  image.Image
	Extend Extend
}
