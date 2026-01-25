/**
 * Two-moon dataset generation for binary classification
 * Creates two interleaving half-circles of points
 */

export interface DataPoint {
    x: number
    y: number
    label: 0 | 1
}

export interface TwoMoonDataset {
    points: DataPoint[]
    X: number[][] // [n_samples, 2]
    y: number[]   // [n_samples]
}

/**
 * Generate the classic two-moon dataset
 * @param nSamples - Number of samples per moon (total = 2 * nSamples)
 * @param noise - Standard deviation of Gaussian noise
 * @param separation - Vertical separation between moons
 */
export function generateTwoMoons(
    nSamples: number = 100,
    noise: number = 0.1,
    separation: number = 0.5
): TwoMoonDataset {
    const points: DataPoint[] = []
    const X: number[][] = []
    const y: number[] = []

    // Generate first moon (upper, label 0)
    for (let i = 0; i < nSamples; i++) {
        const angle = (Math.PI * i) / nSamples
        const x = Math.cos(angle) + randomGaussian() * noise
        const yCoord = Math.sin(angle) + randomGaussian() * noise

        points.push({ x, y: yCoord, label: 0 })
        X.push([x, yCoord])
        y.push(0)
    }

    // Generate second moon (lower, label 1)
    for (let i = 0; i < nSamples; i++) {
        const angle = (Math.PI * i) / nSamples
        const x = 1 - Math.cos(angle) + randomGaussian() * noise
        const yCoord = 1 - Math.sin(angle) - separation + randomGaussian() * noise

        points.push({ x, y: yCoord, label: 1 })
        X.push([x, yCoord])
        y.push(1)
    }

    return { points, X, y }
}

/**
 * Box-Muller transform for Gaussian random numbers
 */
function randomGaussian(): number {
    const u1 = Math.random()
    const u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/**
 * Normalize dataset to [0, 1] range for visualization
 */
export function normalizeForVisualization(dataset: TwoMoonDataset): TwoMoonDataset {
    const xs = dataset.points.map(p => p.x)
    const ys = dataset.points.map(p => p.y)

    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)

    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1
    const padding = 0.1

    const normalizedPoints = dataset.points.map(p => ({
        x: padding + (p.x - minX) / rangeX * (1 - 2 * padding),
        y: padding + (p.y - minY) / rangeY * (1 - 2 * padding),
        label: p.label
    }))

    const normalizedX = dataset.X.map(([x, y]) => [
        (x - minX) / rangeX,
        (y - minY) / rangeY
    ])

    return {
        points: normalizedPoints,
        X: normalizedX,
        y: dataset.y
    }
}
