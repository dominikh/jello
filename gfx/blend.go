//go:generate stringer -type=Mix
//go:generate stringer -type=Compose

package gfx

import "fmt"

// The values of constants in this package do not match those in peniko, because
// we want the zero values to be sane defaults. Our Vello port will have the
// same changes.
//
// Specifically, Clear and SrcOver have been swapped.

// / Defines the color mixing function for a [blend operation](BlendMode).
type Mix uint8

const (
	// Default attribute which specifies no blending. The blending formula
	// simply selects the source color.
	MixNormal Mix = 0
	// Source color is multiplied by the destination color and replaces the
	// destination.
	MixMultiply Mix = 1
	// Multiplies the complements of the backdrop and source color values, then
	// complements the result.
	MixScreen Mix = 2
	// Multiplies or screens the colors, depending on the backdrop color value.
	MixOverlay Mix = 3
	// Selects the darker of the backdrop and source colors.
	MixDarken Mix = 4
	// Selects the lighter of the backdrop and source colors.
	MixLighten Mix = 5
	// Brightens the backdrop color to reflect the source color. Painting with
	// black produces no change.
	MixColorDodge Mix = 6
	// Darkens the backdrop color to reflect the source color. Painting with
	// white produces no change.
	MixColorBurn Mix = 7
	// Multiplies or screens the colors, depending on the source color value.
	// The effect is similar to shining a harsh spotlight on the backdrop.
	MixHardLight Mix = 8
	// Darkens or lightens the colors, depending on the source color value. The
	// effect is similar to shining a diffused spotlight on the backdrop.
	MixSoftLight Mix = 9
	// Subtracts the darker of the two constituent colors from the lighter
	// color.
	MixDifference Mix = 10
	// Produces an effect similar to that of the Difference mode but lower in
	// contrast. Painting with white inverts the backdrop color; painting with
	// black produces no change.
	MixExclusion Mix = 11
	// Creates a color with the hue of the source color and the saturation and
	// luminosity of the backdrop color.
	MixHue Mix = 12
	// Creates a color with the saturation of the source color and the hue and
	// luminosity of the backdrop color. Painting with this mode in an area of
	// the backdrop that is a pure gray (no saturation) produces no change.
	MixSaturation Mix = 13
	// Creates a color with the hue and saturation of the source color and the
	// luminosity of the backdrop color. This preserves the gray levels of the
	// backdrop and is useful for coloring monochrome images or tinting color
	// images.
	MixColor Mix = 14
	// Creates a color with the luminosity of the source color and the hue and
	// saturation of the backdrop color. This produces an inverse effect to that
	// of the Color mode.
	MixLuminosity Mix = 15
	// Clip is the same as normal, but the latter always creates an isolated
	// blend group and the former can optimize that out.
	MixClip Mix = 128
)

// / Defines the layer composition function for a [blend operation](BlendMode).
type Compose uint8

const (
	// The source is placed over the destination.
	ComposeSrcOver Compose = 0
	// Only the source will be present.
	ComposeCopy Compose = 1
	// Only the destination will be present.
	ComposeDest Compose = 2
	// No regions are enabled.
	ComposeClear Compose = 3
	// The destination is placed over the source.
	ComposeDestOver Compose = 4
	// The parts of the source that overlap with the destination are placed.
	ComposeSrcIn Compose = 5
	// The parts of the destination that overlap with the source are placed.
	ComposeDestIn Compose = 6
	// The parts of the source that fall outside of the destination are placed.
	ComposeSrcOut Compose = 7
	// The parts of the destination that fall outside of the source are placed.
	ComposeDestOut Compose = 8
	// The parts of the source which overlap the destination replace the
	// destination. The destination is placed everywhere else.
	ComposeSrcAtop Compose = 9
	// The parts of the destination which overlaps the source replace the
	// source. The source is placed everywhere else.
	ComposeDestAtop Compose = 10
	// The non-overlapping regions of source and destination are combined.
	ComposeXor Compose = 11
	// The sum of the source image and destination image is displayed.
	ComposePlus Compose = 12
	// Allows two elements to cross fade by changing their opacities from 0 to 1
	// on one element and 1 to 0 on the other element.
	ComposePlusLighter Compose = 13
)

// / Blend mode consisting of [color mixing](Mix) and [composition functions](Compose).
type BlendMode struct {
	// The color mixing function
	Mix Mix
	// The layer composition function
	Compose Compose
}

func (bm BlendMode) String() string {
	return fmt.Sprintf("(%s, %s)", bm.Mix, bm.Compose)
}
