import { createEffect, type JSX } from 'solid-js'

interface MnistImageProps {
    pixels: Float32Array
    /** Pixel size of the rendered image (width = height since MNIST is square) */
    size?: number
    class?: string
}

/**
 * Renders a single 28×28 MNIST digit image onto a canvas.
 * `pixels` should be a Float32Array of length 784 with values in [0, 1].
 */
export function MnistImage(props: MnistImageProps): JSX.Element {
    let canvas!: HTMLCanvasElement

    createEffect(() => {
        const pixels = props.pixels
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const imageData = ctx.createImageData(28, 28)
        for (let i = 0; i < 784; i++) {
            const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)))
            imageData.data[i * 4 + 0] = v
            imageData.data[i * 4 + 1] = v
            imageData.data[i * 4 + 2] = v
            imageData.data[i * 4 + 3] = 255
        }
        ctx.putImageData(imageData, 0, 0)
    })

    const displaySize = () => props.size ?? 28

    return (
        <canvas
            ref={canvas}
            width={28}
            height={28}
            class={props.class}
            style={{
                width: `${displaySize()}px`,
                height: `${displaySize()}px`,
                'image-rendering': 'pixelated',
            }}
        />
    )
}
