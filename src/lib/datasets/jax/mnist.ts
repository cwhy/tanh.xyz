import type { MnistRawDataset } from '../../datasources/mnist'

export interface MnistMlpSplit {
    images: Float32Array
    labels: Int32Array
    oneHot: Float32Array
    count: number
    width: number
    height: number
}

export interface MnistMlpDataset {
    train: MnistMlpSplit
    test: MnistMlpSplit
}

function toInt32Array(data: Int32Array | Float32Array): Int32Array {
    if (data instanceof Int32Array) return data
    const out = new Int32Array(data.length)
    for (let i = 0; i < data.length; i++) {
        out[i] = Math.round(data[i])
    }
    return out
}

function formatSplit(
    imagesRaw: Int32Array | Float32Array,
    imageShape: number[],
    labelsRaw: Int32Array | Float32Array
): MnistMlpSplit {
    if (imageShape.length !== 3) {
        throw new Error(`Expected image shape rank 3, got rank ${imageShape.length}`)
    }

    const [count, height, width] = imageShape
    const pixelsPerImage = width * height
    if (imagesRaw.length !== count * pixelsPerImage) {
        throw new Error('MNIST image buffer size does not match shape metadata')
    }

    const labels = toInt32Array(labelsRaw)
    if (labels.length !== count) {
        throw new Error('MNIST labels length does not match image count')
    }

    const images = new Float32Array(imagesRaw.length)
    for (let i = 0; i < imagesRaw.length; i++) {
        images[i] = imagesRaw[i] / 255
    }

    const oneHot = new Float32Array(count * 10)
    for (let i = 0; i < count; i++) {
        const label = labels[i]
        if (label >= 0 && label < 10) {
            oneHot[i * 10 + label] = 1
        }
    }

    return {
        images,
        labels,
        oneHot,
        count,
        width,
        height,
    }
}

export function formatMnistForMlp(raw: MnistRawDataset): MnistMlpDataset {
    return {
        train: formatSplit(raw.train.images.data, raw.train.images.shape, raw.train.labels.data),
        test: formatSplit(raw.test.images.data, raw.test.images.shape, raw.test.labels.data),
    }
}

export function getImagePixels(
    split: MnistMlpSplit,
    index: number
): Float32Array {
    const pixelsPerImage = split.width * split.height
    const start = index * pixelsPerImage
    const end = start + pixelsPerImage
    return split.images.subarray(start, end)
}

export function getImageTensor2D(
    split: MnistMlpSplit,
    index: number
): number[][] {
    const pixels = getImagePixels(split, index)
    const rows: number[][] = []
    for (let r = 0; r < split.height; r++) {
        const row: number[] = []
        for (let c = 0; c < split.width; c++) {
            row.push(pixels[r * split.width + c])
        }
        rows.push(row)
    }
    return rows
}
