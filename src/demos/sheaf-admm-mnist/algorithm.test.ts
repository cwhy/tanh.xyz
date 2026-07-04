import { describe, expect, test } from 'vitest'
import {
    AGENT_COUNT,
    DIGIT_COUNT,
    MNIST_SIZE,
    buildMnistPrototypes,
    createSheafAdmmSession,
    runSheafAdmm,
    stepSheafAdmm,
    type PatchAgent,
    type SheafAdmmSnapshot,
} from './algorithm'

function blankPixels(): Float32Array {
    return new Float32Array(MNIST_SIZE * MNIST_SIZE)
}

function drawSquare(pixels: Float32Array, value: number, top: number, left: number, size: number): Float32Array {
    for (let r = top; r < top + size; r++) {
        for (let c = left; c < left + size; c++) {
            pixels[r * MNIST_SIZE + c] = value
        }
    }
    return pixels
}

describe('Sheaf-ADMM MNIST math', () => {
    test('builds per-class MNIST prototypes by averaging labeled images', () => {
        const images = new Float32Array(3 * MNIST_SIZE * MNIST_SIZE)
        drawSquare(images.subarray(0, MNIST_SIZE * MNIST_SIZE), 1, 0, 0, 4)
        drawSquare(images.subarray(MNIST_SIZE * MNIST_SIZE, 2 * MNIST_SIZE * MNIST_SIZE), 0.5, 0, 0, 4)
        drawSquare(images.subarray(2 * MNIST_SIZE * MNIST_SIZE), 0.25, 12, 12, 4)

        const prototypes = buildMnistPrototypes(images, new Int32Array([3, 3, 8]))
        const digit3Offset = 3 * MNIST_SIZE * MNIST_SIZE
        const digit8Offset = 8 * MNIST_SIZE * MNIST_SIZE

        expect(prototypes.counts[3]).toBe(2)
        expect(prototypes.counts[8]).toBe(1)
        expect(prototypes.means[digit3Offset]).toBeCloseTo(0.75)
        expect(prototypes.means[digit8Offset + 12 * MNIST_SIZE + 12]).toBeCloseTo(0.25)
    })

    test('creates a 7x7 local-agent MNIST session with grid neighbor edges', () => {
        const images = new Float32Array(10 * MNIST_SIZE * MNIST_SIZE)
        for (let digit = 0; digit < 10; digit++) {
            drawSquare(
                images.subarray(digit * MNIST_SIZE * MNIST_SIZE, (digit + 1) * MNIST_SIZE * MNIST_SIZE),
                digit / 9,
                digit % 5,
                digit % 4,
                4
            )
        }
        const prototypes = buildMnistPrototypes(images, new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        const session = createSheafAdmmSession(drawSquare(blankPixels(), 1, 0, 0, 4), prototypes)

        expect(session.agents).toHaveLength(AGENT_COUNT)
        expect(session.edges).toHaveLength(84)
        expect(session.initial.probabilities).toHaveLength(DIGIT_COUNT)
    })

    test('diffusion lowers projected sheaf disagreement on a controlled chain', () => {
        const agents: PatchAgent[] = Array.from({ length: AGENT_COUNT }, (_, id) => ({
            id,
            row: Math.floor(id / 7),
            col: id % 7,
            ink: 1,
            target: Array.from({ length: DIGIT_COUNT }, (_, digit) => (id === 0 && digit === 2 ? 4 : 0)),
            reliability: Array.from({ length: DIGIT_COUNT }, () => 1),
        }))
        agents[1].target = Array.from({ length: DIGIT_COUNT }, (_, digit) => (digit === 2 ? -4 : 0))

        const snapshot: SheafAdmmSnapshot = {
            iteration: 0,
            x: agents.map(agent => [...agent.target]),
            z: agents.map(agent => [...agent.target]),
            u: Array.from({ length: AGENT_COUNT }, () => Array.from({ length: DIGIT_COUNT }, () => 0)),
            prediction: 2,
            localPrediction: 2,
            probabilities: Array.from({ length: DIGIT_COUNT }, (_, digit) => (digit === 2 ? 1 : 0)),
            primalResidual: 0,
            consensusResidual: 0,
            dualEnergy: 0,
            edges: [{ from: 0, to: 1, weight: 1, mismatch: 0 }],
        }

        const next = stepSheafAdmm(agents, snapshot, { rho: 1, gamma: 0.1, diffusionSteps: 8 })

        expect(next.consensusResidual).toBeLessThan(1.1)
        expect(Math.abs(next.z[0][2] - next.z[1][2])).toBeLessThan(Math.abs(snapshot.z[0][2] - snapshot.z[1][2]))
    })

    test('unrolled iterations keep residuals finite', () => {
        const images = new Float32Array(10 * MNIST_SIZE * MNIST_SIZE)
        for (let digit = 0; digit < 10; digit++) {
            drawSquare(
                images.subarray(digit * MNIST_SIZE * MNIST_SIZE, (digit + 1) * MNIST_SIZE * MNIST_SIZE),
                1,
                digit,
                digit,
                3
            )
        }
        const prototypes = buildMnistPrototypes(images, new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        const session = createSheafAdmmSession(drawSquare(blankPixels(), 1, 4, 4, 8), prototypes)
        const history = runSheafAdmm(session, { rho: 1.2, gamma: 0.07, diffusionSteps: 4 }, 6)

        expect(history).toHaveLength(7)
        for (const snapshot of history) {
            expect(Number.isFinite(snapshot.primalResidual)).toBe(true)
            expect(Number.isFinite(snapshot.consensusResidual)).toBe(true)
            expect(snapshot.prediction).toBeGreaterThanOrEqual(0)
            expect(snapshot.prediction).toBeLessThan(10)
        }
    })
})

