import { numpy as np } from '@jax-js/jax'

/**
 * Sparse distance cache for high-dimensional data.
 * Lazily computes and caches pairwise distances between indexed vectors.
 */
export class DistanceCache {
    private cache = new Map<number, number>()
    private totalPoints: number

    constructor(totalPoints: number) {
        this.totalPoints = totalPoints
    }

    private key(i: number, j: number): number {
        // Canonical key with smaller index first
        return i < j ? i * this.totalPoints + j : j * this.totalPoints + i
    }

    get(i: number, j: number): number | undefined {
        if (i === j) return 0
        return this.cache.get(this.key(i, j))
    }

    set(i: number, j: number, dist: number): void {
        this.cache.set(this.key(i, j), dist)
    }

    /**
     * Return cached distance, or compute synchronously from imageData and cache it.
     */
    compute(i: number, j: number, imageData: Float32Array, dim: number = 784): number {
        if (i === j) return 0
        const cached = this.get(i, j)
        if (cached !== undefined) return cached

        let sum = 0
        const iOff = i * dim
        const jOff = j * dim
        for (let k = 0; k < dim; k++) {
            const d = imageData[iOff + k] - imageData[jOff + k]
            sum += d * d
        }
        const dist = Math.sqrt(sum)
        this.set(i, j, dist)
        return dist
    }

    get size(): number {
        return this.cache.size
    }

    clear(): void {
        this.cache.clear()
    }
}

/**
 * Batch compute Euclidean distances from one image to many target images using JAX-JS.
 * Falls back to plain JS for small batches to avoid GPU dispatch overhead.
 */
export async function computeBatchDistancesJax(
    imageData: Float32Array,
    queryIdx: number,
    targetIndices: number[],
    dim: number = 784
): Promise<Float32Array> {
    const m = targetIndices.length
    if (m === 0) return new Float32Array(0)

    // For very small batches, plain JS is faster than GPU dispatch overhead
    if (m <= 4) {
        const result = new Float32Array(m)
        const iOff = queryIdx * dim
        for (let t = 0; t < m; t++) {
            let sum = 0
            const jOff = targetIndices[t] * dim
            for (let k = 0; k < dim; k++) {
                const d = imageData[iOff + k] - imageData[jOff + k]
                sum += d * d
            }
            result[t] = Math.sqrt(sum)
        }
        return result
    }

    // Build target matrix [m, dim]
    const targetMatrix = new Float32Array(m * dim)
    for (let i = 0; i < m; i++) {
        targetMatrix.set(
            imageData.subarray(targetIndices[i] * dim, (targetIndices[i] + 1) * dim),
            i * dim
        )
    }
    const queryPixels = imageData.subarray(queryIdx * dim, (queryIdx + 1) * dim)

    // JAX-JS: broadcast subtract [m, dim] - [1, dim], square, sum axis 1, sqrt
    const q = np.array(queryPixels, { shape: [1, dim] })
    const M = np.array(targetMatrix, { shape: [m, dim] })
    const diff = np.subtract(M, q)
    const sq = np.multiply(diff, diff)
    const sumSq = np.sum(sq, 1)
    const dists = np.sqrt(sumSq)

    return await dists.jsAsync() as Float32Array
}
