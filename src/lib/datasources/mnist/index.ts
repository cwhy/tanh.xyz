export interface MnistIdxFile {
    shape: number[]
    data: Int32Array | Float32Array
}

export interface MnistRawDataset {
    train: {
        images: MnistIdxFile
        labels: MnistIdxFile
    }
    test: {
        images: MnistIdxFile
        labels: MnistIdxFile
    }
}

export interface MnistFetchProgress {
    stage: 'check-cache' | 'downloading' | 'decompressing' | 'parsing' | 'ready'
    file: string
    completedFiles: number
    totalFiles: number
}

interface FetchOptions {
    onProgress?: (progress: MnistFetchProgress) => void
    forceRefresh?: boolean
}

const CACHE_NAME = 'mnist-datasource-v1'

const mnistLinks = {
    train: {
        images: 'https://cdn.jsdelivr.net/gh/fgnt/mnist/train-images-idx3-ubyte.gz',
        labels: 'https://cdn.jsdelivr.net/gh/fgnt/mnist/train-labels-idx1-ubyte.gz',
    },
    test: {
        images: 'https://cdn.jsdelivr.net/gh/fgnt/mnist/t10k-images-idx3-ubyte.gz',
        labels: 'https://cdn.jsdelivr.net/gh/fgnt/mnist/t10k-labels-idx1-ubyte.gz',
    },
} as const

function fileNameFromUrl(url: string): string {
    const parts = url.split('/')
    return parts[parts.length - 1] ?? url
}

async function fetchGzipBytes(
    url: string,
    options: {
        fileIndex: number
        totalFiles: number
        onProgress?: (progress: MnistFetchProgress) => void
        forceRefresh?: boolean
    }
): Promise<ArrayBuffer> {
    const { fileIndex, totalFiles, onProgress, forceRefresh = false } = options
    const file = fileNameFromUrl(url)

    onProgress?.({
        stage: 'check-cache',
        file,
        completedFiles: fileIndex,
        totalFiles,
    })

    let response: Response | undefined
    const canUseCache = typeof caches !== 'undefined'

    if (canUseCache && !forceRefresh) {
        const cache = await caches.open(CACHE_NAME)
        response = await cache.match(url) ?? undefined
        if (!response) {
            onProgress?.({
                stage: 'downloading',
                file,
                completedFiles: fileIndex,
                totalFiles,
            })
            response = await fetch(url)
            if (!response.ok) {
                throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`)
            }
            await cache.put(url, response.clone())
        }
    } else {
        onProgress?.({
            stage: 'downloading',
            file,
            completedFiles: fileIndex,
            totalFiles,
        })
        response = await fetch(url)
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`)
        }
    }

    return response.arrayBuffer()
}

async function parseIdxFromGzip(
    url: string,
    options: {
        fileIndex: number
        totalFiles: number
        onProgress?: (progress: MnistFetchProgress) => void
        forceRefresh?: boolean
    }
): Promise<MnistIdxFile> {
    const { fileIndex, totalFiles, onProgress } = options
    const file = fileNameFromUrl(url)

    const gzipBytes = await fetchGzipBytes(url, options)

    onProgress?.({
        stage: 'decompressing',
        file,
        completedFiles: fileIndex,
        totalFiles,
    })

    const stream = new Blob([gzipBytes]).stream().pipeThrough(new DecompressionStream('gzip'))
    const buffer = await new Response(stream).arrayBuffer()
    const view = new DataView(buffer)

    onProgress?.({
        stage: 'parsing',
        file,
        completedFiles: fileIndex,
        totalFiles,
    })

    const dataType = view.getUint8(2)
    const rank = view.getUint8(3)

    const shape: number[] = []
    for (let i = 0; i < rank; i++) {
        shape.push(view.getUint32(4 + i * 4, false))
    }
    const size = shape.reduce((acc, value) => acc * value, 1)
    const dataView = new DataView(buffer, 4 + rank * 4)

    let data: Int32Array | Float32Array
    switch (dataType) {
        case 0x08:
            data = new Int32Array(size)
            for (let i = 0; i < size; i++) data[i] = dataView.getUint8(i)
            break
        case 0x09:
            data = new Int32Array(size)
            for (let i = 0; i < size; i++) data[i] = dataView.getInt8(i)
            break
        case 0x0b:
            data = new Int32Array(size)
            for (let i = 0; i < size; i++) data[i] = dataView.getInt16(i * 2, false)
            break
        case 0x0c:
            data = new Int32Array(size)
            for (let i = 0; i < size; i++) data[i] = dataView.getInt32(i * 4, false)
            break
        case 0x0d:
            data = new Float32Array(size)
            for (let i = 0; i < size; i++) data[i] = dataView.getFloat32(i * 4, false)
            break
        case 0x0e:
            data = new Float32Array(size)
            for (let i = 0; i < size; i++) data[i] = dataView.getFloat64(i * 8, false)
            break
        default:
            throw new Error(`Unsupported IDX data type: ${dataType}`)
    }

    onProgress?.({
        stage: 'ready',
        file,
        completedFiles: fileIndex + 1,
        totalFiles,
    })

    return { shape, data }
}

export async function fetchMnistRaw(options: FetchOptions = {}): Promise<MnistRawDataset> {
    const { onProgress, forceRefresh = false } = options

    const urls = [
        mnistLinks.train.images,
        mnistLinks.train.labels,
        mnistLinks.test.images,
        mnistLinks.test.labels,
    ]

    const results = await Promise.all(
        urls.map((url, index) =>
            parseIdxFromGzip(url, {
                fileIndex: index,
                totalFiles: urls.length,
                onProgress,
                forceRefresh,
            })
        )
    )

    return {
        train: {
            images: results[0],
            labels: results[1],
        },
        test: {
            images: results[2],
            labels: results[3],
        },
    }
}
