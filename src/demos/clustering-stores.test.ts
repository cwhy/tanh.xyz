import { afterEach, describe, expect, it, vi } from 'vitest'
import { createRoot, flush } from 'solid-js'
import { createOnlineKMeansStore } from './online-k-means/store'
import { createNKClusteringStore } from './nk-clustering/store'
import { createRivalKStore } from './rival-k/store'

// Exercise the browser scheduler: the server build commits writes immediately
// and would hide the read-after-write regressions this migration must prevent.
const disposers: Array<() => void> = []
function owned<T>(create: () => T): T {
    return createRoot(dispose => {
        disposers.push(dispose)
        return create()
    })
}
afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
    vi.useRealTimers()
})

describe('clustering actions with batched writes', () => {
    it('uses the new K immediately when rebuilding the algorithm', () => {
        const store = owned(createOnlineKMeansStore)
        store.setK(2)
        store.addSinglePoint(0, 0)
        store.addSinglePoint(1, 1)
        store.addSinglePoint(0.1, 0.1)
        flush()
        expect(store.k()).toBe(2)
        expect(store.clusters()).toHaveLength(2)
        expect(store.points()).toHaveLength(3)
    })

    for (const [name, create] of [
        ['online K-means', createOnlineKMeansStore],
        ['NK', createNKClusteringStore],
        ['Rival-K', createRivalKStore],
    ] as const) {
        it(`${name}: generates and consumes consecutive points before a UI flush`, () => {
            const store = owned(() => create())
            store.generateRandomPoints(5, 'uniform')
            const first = store.stepOnce()
            const second = store.stepOnce()
            expect(first).not.toBeNull()
            expect(second).not.toBeNull()
            expect(second?.point.id).not.toBe(first?.point.id)
            flush()
            expect(store.currentPointIndex()).toBe(2)
            store.reset()
            flush()
            expect(store.points()).toHaveLength(0)
            expect(store.isStreaming()).toBe(false)
        })

        it(`${name}: starts immediately after generation and stops its timer`, () => {
            vi.useFakeTimers()
            const store = owned(() => create())
            store.generateRandomPoints(5, 'uniform')
            store.startStreaming()
            flush()
            expect(store.isStreaming()).toBe(true)
            vi.advanceTimersByTime(1000)
            flush()
            expect(store.currentPointIndex()).toBeGreaterThan(0)
            store.stopStreaming()
            flush()
            const count = store.currentPointIndex()
            vi.advanceTimersByTime(2000)
            flush()
            expect(store.currentPointIndex()).toBe(count)
            expect(vi.getTimerCount()).toBe(0)
        })
    }

    it('preserves the NK constraint when N and K change in the same event', () => {
        const store = owned(createNKClusteringStore)
        store.setN(2)
        store.setK(5)
        flush()
        expect(store.n()).toBe(6)
        expect(store.k()).toBe(5)
    })
})

for (const [name, create] of [
    ['NK', createNKClusteringStore],
    ['Rival-K', createRivalKStore],
] as const) {
    it(`${name}: processes deferred evictions across replay rounds`, () => {
        let seed = 17
        vi.spyOn(Math, 'random').mockImplementation(() => {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
            return seed / 2 ** 32
        })
        try {
            const store = owned(() => create())
            store.setCancelReplay(true)
            store.generateRandomPoints(50, 'uniform')
            for (let i = 0; i < 50; i++) expect(store.stepOnce()).not.toBeNull()
            flush()
            expect(store.replayQueueLength()).toBeGreaterThan(0)
            // Continue through queue promotion, including same-event step calls.
            for (let i = 0; i < 100; i++) {
                const result = store.stepOnce()
                if (!result) break
                expect(Number.isFinite(result.point.x)).toBe(true)
                expect(Number.isFinite(result.point.y)).toBe(true)
            }
            flush()
            expect(store.currentPointIndex()).toBe(50)
            expect(store.points().every(p => Number.isFinite(p.point.x))).toBe(true)
        } finally {
            vi.restoreAllMocks()
        }
    })
}
