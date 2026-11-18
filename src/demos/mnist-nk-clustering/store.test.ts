import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createRoot, createSignal, flush } from 'solid-js'
import { computeBatchDistancesJax } from '../../lib/clustering/distance'
import { createMnistNKStore } from './store'

vi.mock('../../lib/deeplearning/runtime', () => ({ initJax: async () => 'cpu' }))
vi.mock('../../lib/datasources/mnist', () => ({ fetchMnistRaw: async () => ({}) }))
vi.mock('../../lib/datasets/jax/mnist', () => ({
    formatMnistForMlp: () => ({
        test: {
            count: 6,
            images: Float32Array.from({ length: 6 * 784 }, (_, i) => (Math.floor(i / 784) + 1) / 10),
        },
    }),
}))
vi.mock('../../lib/clustering/distance', async importOriginal => ({
    ...await importOriginal<typeof import('../../lib/clustering/distance')>(),
    computeBatchDistancesJax: vi.fn(),
}))

let cleanup: () => void
let store: ReturnType<typeof createMnistNKStore>
beforeEach(async () => {
    vi.mocked(computeBatchDistancesJax).mockReset().mockResolvedValue(new Float32Array([0.1]))
    store = createRoot(dispose => {
        cleanup = dispose
        return createMnistNKStore()
    })
    await store.loadAndGenerate()
    flush()
})
afterEach(() => {
    store.dispose()
    cleanup()
    vi.useRealTimers()
})

function deferredDistances() {
    let resolve!: (value: Float32Array) => void
    const promise = new Promise<Float32Array>(done => { resolve = done })
    vi.mocked(computeBatchDistancesJax).mockReturnValueOnce(promise)
    return () => resolve(new Float32Array([0.1]))
}

it('publishes completed steps and their history together', async () => {
    expect(store.totalCount()).toBe(6)
    await store.stepOnce()
    flush()
    expect(store.processedCount()).toBe(1)
    expect(store.evennessHistory().at(-1)?.step).toBe(1)
})

it('coalesces overlapping steps while GPU distances are pending', async () => {
    await store.stepOnce()
    const resolve = deferredDistances()
    const first = store.stepOnce()
    const second = store.stepOnce()
    expect(computeBatchDistancesJax).toHaveBeenCalledTimes(1)
    resolve()
    const [a, b] = await Promise.all([first, second])
    flush()
    expect(a?.point.id).toBe(b?.point.id)
    expect(store.processedCount()).toBe(2)
    expect(store.evennessHistory()).toHaveLength(2)
})

it('discards a GPU result from a run that was reset', async () => {
    await store.stepOnce()
    const resolve = deferredDistances()
    const pending = store.stepOnce()
    store.reset()
    resolve()
    expect(await pending).toBeNull()
    flush()
    expect(store.processedCount()).toBe(0)
    expect(store.clusters()).toEqual([])
    expect(store.evennessHistory()).toEqual([])
    expect(await store.stepOnce()).not.toBeNull()
})

it('discards pending work and clears timers when the page is disposed', async () => {
    vi.useFakeTimers()
    await store.stepOnce()
    const resolve = deferredDistances()
    const pending = store.stepOnce()
    store.startStreaming()
    store.dispose()
    resolve()
    expect(await pending).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
    flush()
    expect(store.processedCount()).toBe(1)
})

it('validates staged parameters without flushing unrelated writes', () => {
    let dispose!: () => void
    const [value, setValue] = createRoot(cleanup => {
        dispose = cleanup
        return createSignal(0)
    })
    try {
        setValue(1)
        store.setN(2)
        store.setK(5)
        expect(value()).toBe(0)
        flush()
        expect(store.n()).toBe(6)
        expect(store.k()).toBe(5)
    } finally {
        dispose()
    }
})
