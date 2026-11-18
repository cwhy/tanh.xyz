import { expect, it, vi } from 'vitest'
import { createRoot, flush } from 'solid-js'
import { defaultDevice } from '@jax-js/jax'
import { createTrainingStore } from './training-store'

vi.mock('../../lib/deeplearning/runtime', () => ({
    initJax: async () => {
        defaultDevice('cpu')
        return 'cpu'
    },
}))

it('trains with real JAX objects and regenerates data without proxying model parameters', async () => {
    let dispose!: () => void
    const store = createRoot(cleanup => {
        dispose = cleanup
        return createTrainingStore({ hiddenSize: 4, gridResolution: 4 })
    })
    try {
        store.setNSamples(10)
        store.setEpochsPerRound(2)
        store.setBatchSize(5)
        flush()
        await store.initialize()
        flush()
        await store.runTraining()
        flush()
        expect(store.state.epoch).toBe(2)
        expect(Number.isFinite(store.state.loss)).toBe(true)
        expect(store.state.decisionGrid).toHaveLength(4)
        expect(store.state.isTraining).toBe(false)
        store.generateData()
        flush()
        expect(store.state.epoch).toBe(0)
        expect(store.state.decisionGrid).toBeNull()
    } finally {
        dispose()
    }
})
