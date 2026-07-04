import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { afterEach, describe, expect, test, vi } from 'vitest'
import {
    loadTrainedSheafAdmmMnistModel,
    runTrainedSheafAdmmMnist,
} from './trainedModel'

const PUBLIC_ROOT = join(process.cwd(), 'public')

function installPublicFetch() {
    const originalFetch = globalThis.fetch
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(String(input), 'http://local')
        const file = await readFile(join(PUBLIC_ROOT, url.pathname))
        return new Response(file)
    }) as typeof fetch
    return () => {
        globalThis.fetch = originalFetch
    }
}

describe('trained Sheaf-ADMM MNIST browser model', () => {
    let restoreFetch: (() => void) | null = null

    afterEach(() => {
        restoreFetch?.()
        restoreFetch = null
        vi.restoreAllMocks()
    })

    test('loads the exported checkpoint artifact and runs finite ADMM inference', async () => {
        restoreFetch = installPublicFetch()
        const model = await loadTrainedSheafAdmmMnistModel('http://local/data/demos/sheaf-admm-mnist/params/v1/manifest.json')
        const pixels = new Float32Array(28 * 28)
        for (let i = 0; i < 28; i++) {
            pixels[i * 28 + i] = 1
            pixels[i * 28 + Math.max(0, 27 - i)] = 0.6
        }

        const run = runTrainedSheafAdmmMnist(model, pixels, 3)
        const latest = run.history[run.history.length - 1]

        expect(model.config.numAgents).toBe(81)
        expect(model.edges).toHaveLength(272)
        expect(run.session.agents).toHaveLength(81)
        expect(run.history).toHaveLength(4)
        expect(latest.x).toHaveLength(81)
        expect(latest.z).toHaveLength(81)
        expect(latest.edges).toHaveLength(272)
        expect(latest.prediction).toBeGreaterThanOrEqual(0)
        expect(latest.prediction).toBeLessThan(10)
        expect(latest.probabilities.reduce((acc, value) => acc + value, 0)).toBeCloseTo(1, 5)
        expect(Number.isFinite(latest.primalResidual)).toBe(true)
        expect(Number.isFinite(latest.consensusResidual)).toBe(true)
        expect(Number.isFinite(latest.dualEnergy)).toBe(true)
        expect(latest.probabilities.every(Number.isFinite)).toBe(true)
    })
})
