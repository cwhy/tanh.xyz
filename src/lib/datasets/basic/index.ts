export interface Point2D {
    x: number
    y: number
}

export type BasicDatasetDistribution =
    | 'uniform'
    | 'clusters'
    | 'two-moons'
    | 'spiral'
    | 'checkerboard'

interface TwoMoonsOptions {
    noise?: number
    separation?: number
    padding?: number
}

interface SpiralOptions {
    turns?: number
    radialJitter?: number
    tangentialJitter?: number
    padding?: number
}

interface CheckerboardOptions {
    squaresPerAxis?: number
    padding?: number
}

function randomGaussian(): number {
    const u1 = Math.random()
    const u2 = Math.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
}

export function normalizePoints(points: Point2D[], padding: number = 0.1): Point2D[] {
    if (points.length === 0) {
        return []
    }

    const xs = points.map(p => p.x)
    const ys = points.map(p => p.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)

    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1

    return points.map(p => ({
        x: padding + ((p.x - minX) / rangeX) * (1 - 2 * padding),
        y: padding + ((p.y - minY) / rangeY) * (1 - 2 * padding),
    }))
}

export function generateUniformPoints(count: number, padding: number = 0.05): Point2D[] {
    return Array.from({ length: count }, () => ({
        x: padding + Math.random() * (1 - 2 * padding),
        y: padding + Math.random() * (1 - 2 * padding),
    }))
}

export function generateClusteredPoints(count: number): Point2D[] {
    const points: Point2D[] = []
    const numClusters = 3 + Math.floor(Math.random() * 3) // 3-5 clusters

    const centers = Array.from({ length: numClusters }, () => ({
        x: 0.15 + Math.random() * 0.7,
        y: 0.15 + Math.random() * 0.7,
    }))

    for (let i = 0; i < count; i++) {
        const center = centers[Math.floor(Math.random() * numClusters)]
        const r = Math.sqrt(-2 * Math.log(Math.random())) * 0.08
        const theta = Math.random() * 2 * Math.PI

        points.push({
            x: clamp(center.x + r * Math.cos(theta), 0.05, 0.95),
            y: clamp(center.y + r * Math.sin(theta), 0.05, 0.95),
        })
    }

    return points
}

export function generateTwoMoonsPoints(count: number, options: TwoMoonsOptions = {}): Point2D[] {
    const { noise = 0.1, separation = 0.5, padding = 0.1 } = options
    const points: Point2D[] = []

    if (count <= 0) {
        return points
    }

    const firstCount = Math.floor(count / 2)
    const secondCount = count - firstCount

    for (let i = 0; i < firstCount; i++) {
        const angle = (Math.PI * i) / Math.max(firstCount - 1, 1)
        const x = Math.cos(angle) + randomGaussian() * noise
        const y = Math.sin(angle) + randomGaussian() * noise
        points.push({ x, y })
    }

    for (let i = 0; i < secondCount; i++) {
        const angle = (Math.PI * i) / Math.max(secondCount - 1, 1)
        const x = 1 - Math.cos(angle) + randomGaussian() * noise
        const y = 1 - Math.sin(angle) - separation + randomGaussian() * noise
        points.push({ x, y })
    }

    return normalizePoints(points, padding)
}

export function generateSpiralPoints(count: number, options: SpiralOptions = {}): Point2D[] {
    const { turns = 2.75, radialJitter = 0.012, tangentialJitter = 0.008, padding = 0.08 } = options
    const points: Point2D[] = []

    if (count <= 0) {
        return points
    }

    const firstArmCount = Math.ceil(count / 2)
    const secondArmCount = count - firstArmCount

    const generateArm = (armCount: number, phase: number) => {
        for (let i = 0; i < armCount; i++) {
            const t = armCount <= 1 ? 0 : i / (armCount - 1)
            const angle = turns * 2 * Math.PI * t + phase
            const radius = 0.14 + 0.9 * t

            points.push({
                x: (radius + randomGaussian() * radialJitter) * Math.cos(angle + randomGaussian() * tangentialJitter),
                y: (radius + randomGaussian() * radialJitter) * Math.sin(angle + randomGaussian() * tangentialJitter),
            })
        }
    }

    generateArm(firstArmCount, 0)
    generateArm(secondArmCount, Math.PI)

    return normalizePoints(points, padding)
}

export function generateCheckerboardPoints(count: number, options: CheckerboardOptions = {}): Point2D[] {
    const { squaresPerAxis = 6, padding = 0.05 } = options

    if (count <= 0) {
        return []
    }

    const activeCells: Array<{ col: number; row: number }> = []
    for (let row = 0; row < squaresPerAxis; row++) {
        for (let col = 0; col < squaresPerAxis; col++) {
            if ((row + col) % 2 === 0) {
                activeCells.push({ row, col })
            }
        }
    }

    return Array.from({ length: count }, () => {
        const cell = activeCells[Math.floor(Math.random() * activeCells.length)]
        const unitX = (cell.col + Math.random()) / squaresPerAxis
        const unitY = (cell.row + Math.random()) / squaresPerAxis

        return {
            x: padding + unitX * (1 - 2 * padding),
            y: padding + unitY * (1 - 2 * padding),
        }
    })
}

export function generateBasicDatasetPoints(
    count: number,
    distribution: BasicDatasetDistribution
): Point2D[] {
    switch (distribution) {
        case 'two-moons':
            return generateTwoMoonsPoints(count)
        case 'spiral':
            return generateSpiralPoints(count)
        case 'checkerboard':
            return generateCheckerboardPoints(count)
        case 'clusters':
            return generateClusteredPoints(count)
        case 'uniform':
            return generateUniformPoints(count)
    }
}
