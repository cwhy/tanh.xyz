/**
 * Color utilities using OKLCH color space
 * OKLCH provides perceptually uniform colors, ideal for data visualization
 */

export interface OklchColor {
    l: number  // Lightness: 0-1
    c: number  // Chroma: 0-0.4 (saturation)
    h: number  // Hue: 0-360
}

/**
 * Convert OKLCH to CSS oklch() string
 */
export function oklchToCss(color: OklchColor): string {
    return `oklch(${color.l} ${color.c} ${color.h})`
}

/**
 * Generate a sequence of distinguishable colors using golden angle
 * The golden angle (~137.5°) produces maximum separation between consecutive hues
 */
export function generateDistinguishableColors(count: number, options?: {
    lightness?: number
    chroma?: number
    startHue?: number
}): OklchColor[] {
    const { lightness = 0.7, chroma = 0.15, startHue = 0 } = options ?? {}
    const goldenAngle = 137.507764  // degrees

    const colors: OklchColor[] = []
    for (let i = 0; i < count; i++) {
        const hue = (startHue + i * goldenAngle) % 360
        colors.push({ l: lightness, c: chroma, h: hue })
    }
    return colors
}

/**
 * Color generator that yields new distinguishable colors on demand
 * Useful for streaming/online scenarios where cluster count is unknown
 */
export function createColorGenerator(options?: {
    lightness?: number
    chroma?: number
    startHue?: number
}): () => OklchColor {
    const { lightness = 0.7, chroma = 0.15, startHue = 0 } = options ?? {}
    const goldenAngle = 137.507764

    let index = 0
    return () => {
        const hue = (startHue + index * goldenAngle) % 360
        index++
        return { l: lightness, c: chroma, h: hue }
    }
}

/**
 * Get a color for a specific index (deterministic)
 * Useful when you need the same color for the same cluster index
 */
export function getColorForIndex(index: number, options?: {
    lightness?: number
    chroma?: number
    startHue?: number
}): OklchColor {
    const { lightness = 0.7, chroma = 0.15, startHue = 0 } = options ?? {}
    const goldenAngle = 137.507764
    const hue = (startHue + index * goldenAngle) % 360
    return { l: lightness, c: chroma, h: hue }
}

/**
 * Create a darker/lighter variant of a color (for medoid highlighting)
 */
export function adjustLightness(color: OklchColor, delta: number): OklchColor {
    return {
        ...color,
        l: Math.max(0, Math.min(1, color.l + delta))
    }
}

/**
 * Create a more/less saturated variant of a color
 */
export function adjustChroma(color: OklchColor, delta: number): OklchColor {
    return {
        ...color,
        c: Math.max(0, Math.min(0.4, color.c + delta))
    }
}
