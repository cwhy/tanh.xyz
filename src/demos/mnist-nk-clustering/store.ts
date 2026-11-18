/**
 * Store for MNIST NK Clustering demo.
 * Manages MNIST data loading, streaming, and NK clustering state.
 *
 * Note: If deferEviction is true, evictions are deferred and stored in a replay queue.
 */

import { createStore } from 'solid-js'
import { initJax } from '../../lib/deeplearning/runtime'
import { fetchMnistRaw, type MnistFetchProgress } from '../../lib/datasources/mnist'
import { formatMnistForMlp } from '../../lib/datasets/jax/mnist'
import { DistanceCache, computeBatchDistancesJax } from '../../lib/clustering/distance'
import { getColorForIndex, oklchToCss, type OklchColor } from '../../lib/visualization/color'
import {
    createMnistNKState,
    addMnistPoint,
    getClusterMode,
    getClusterDisplayData,
    type MnistAddPointResult,
} from './algorithm'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EvennessSnapshot {
    step: number
    median: number
    p75: number
    p25: number
    min: number
}

export type LoadStatus = 'idle' | 'loading' | 'ready'

export interface ClusterDisplayInfo {
    id: number
    color: string
    evenness: number
    maxSize: number
    pointCount: number
    isTopHalf: boolean
    isLastOne: boolean
    isSaturated: boolean
    medoidDataIndex: number | null
    kNNDataIndex: number | null
    pointsSortedByDist: Array<{ dataIndex: number; distToMode: number; isMedoid: boolean; isKNN: boolean }>
}

export interface MnistNKStore {
    // Parameters
    n: () => number
    setN: (n: number) => void
    k: () => number
    setK: (k: number) => void
    subsetSize: () => number
    setSubsetSize: (size: number) => void
    top1Enabled: () => boolean
    setTop1Enabled: (enabled: boolean) => void
    last1ShrinkEnabled: () => boolean
    setLast1ShrinkEnabled: (enabled: boolean) => void
    deferEviction: () => boolean
    setDeferEviction: (enabled: boolean) => void
    streamSpeed: () => number
    setStreamSpeed: (speed: number) => void

    // Loading state
    loadStatus: () => LoadStatus
    loadProgress: () => MnistFetchProgress | null
    jaxDevice: () => string | null

    // Clustering state
    clusters: () => ClusterDisplayInfo[]
    isStreaming: () => boolean
    processedCount: () => number
    totalCount: () => number
    lastAddedDataIndex: () => number | null
    lastEvictedDataIndex: () => number | null
    replayQueueLength: () => number

    // Evenness history
    evennessHistory: () => EvennessSnapshot[]
    streamFinished: () => boolean

    // Actions
    loadAndGenerate: () => Promise<void>
    startStreaming: () => void
    stopStreaming: () => void
    stepOnce: () => Promise<MnistAddPointResult | null>
    reset: () => void
    dispose: () => void

    // Data access
    getImagePixels: (dataIndex: number) => Float32Array
}

// ---------------------------------------------------------------------------
// Store factory
// ---------------------------------------------------------------------------

export function createMnistNKStore(): MnistNKStore {
    const defaults = { n: 15, k: 3, top1Enabled: false, last1ShrinkEnabled: false }
    const [state, setState] = createStore({
        ...defaults,
        subsetSize: 500,
        deferEviction: false,
        streamSpeed: 200,
        jaxDevice: null as string | null,
        loadStatus: 'idle' as LoadStatus,
        loadProgress: null as MnistFetchProgress | null,
        isStreaming: false,
        currentIdx: 0,
        totalCount: 0,
        clusters: [] as ClusterDisplayInfo[],
        lastAddedDataIndex: null as number | null,
        lastEvictedDataIndex: null as number | null,
        replayQueue: [] as number[],
        nextReplayQueue: [] as number[],
        evennessHistory: [] as EvennessSnapshot[],
    })
    type State = typeof state

    // Buffers, caches, and the algorithm remain ordinary objects owned by this instance.
    let allImageData: Float32Array = new Float32Array(0)
    let totalImages = 0
    let shuffledIndices: number[] = []
    let distanceCache = new DistanceCache(0)
    let clusteringState = createMnistNKState(defaults.n, defaults.k, defaults.top1Enabled, defaults.last1ShrinkEnabled)
    let streamingInterval: ReturnType<typeof setInterval> | null = null
    let activeStep: Promise<MnistAddPointResult | null> | null = null
    let generation = 0
    let loading = false
    let disposed = false

    function clearTimer() {
        if (streamingInterval !== null) clearInterval(streamingInterval)
        streamingInterval = null
    }

    function restart(s: State) {
        generation++
        activeStep = null
        clusteringState = createMnistNKState(s.n, s.k, s.top1Enabled, s.last1ShrinkEnabled)
        distanceCache = new DistanceCache(totalImages)
        s.isStreaming = false
        s.clusters = []
        s.currentIdx = 0
        s.lastAddedDataIndex = null
        s.lastEvictedDataIndex = null
        s.replayQueue = []
        s.nextReplayQueue = []
        s.evennessHistory = []
    }

    function configure(update: (draft: State) => void) {
        clearTimer()
        setState(s => {
            update(s)
            restart(s)
        })
    }

    function shuffleSubset(s: State) {
        const indices = Array.from({ length: totalImages }, (_, i) => i)
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1))
            ;[indices[i], indices[j]] = [indices[j], indices[i]]
        }
        shuffledIndices = indices.slice(0, s.subsetSize)
        s.totalCount = shuffledIndices.length
    }

    function getClusterColor(clusterId: number): OklchColor {
        return getColorForIndex(clusterId, { lightness: 0.65, chroma: 0.18 })
    }

    function syncState(s: State) {
        const clusters = clusteringState.clusters
        const k = clusteringState.k

        // Top-half: top ceil(n/2) clusters by evenness
        const sorted = [...clusters].sort((a, b) => b.evenness - a.evenness)
        const halfCount = Math.ceil(sorted.length / 2)
        const topHalfIds = new Set(sorted.slice(0, halfCount).map(c => c.id))

        // Last-one: single least-even cluster
        const lastOneId = sorted.length > 0 ? sorted[sorted.length - 1].id : null

        const displays: ClusterDisplayInfo[] = clusters.map(cluster => {
            const mode = getClusterMode(cluster)
            const { pointsSortedByDist, kNNPoint } = getClusterDisplayData(
                cluster, k, distanceCache, allImageData
            )
            const color = oklchToCss(getClusterColor(cluster.id))
            return {
                id: cluster.id,
                color,
                evenness: cluster.evenness,
                maxSize: cluster.maxSize,
                pointCount: cluster.points.length,
                isTopHalf: topHalfIds.has(cluster.id),
                isLastOne: cluster.id === lastOneId,
                isSaturated: cluster.maxSize <= k + 2,
                medoidDataIndex: mode?.dataIndex ?? null,
                kNNDataIndex: kNNPoint?.dataIndex ?? null,
                pointsSortedByDist: pointsSortedByDist.map(({ point, distToMode }) => ({
                    dataIndex: point.dataIndex,
                    distToMode,
                    isMedoid: point.id === cluster.medoidId,
                    isKNN: kNNPoint !== null && point.id === kNNPoint.id,
                })),
            }
        })

        s.clusters = displays

        // Record evenness history snapshot
        if (clusters.length > 0) {
            const evennessValues = clusters.map(c => c.evenness).sort((a, b) => a - b)
            const percentile = (arr: number[], p: number) => {
                const idx = (p / 100) * (arr.length - 1)
                const lo = Math.floor(idx)
                const hi = Math.ceil(idx)
                return lo === hi ? arr[lo] : arr[lo] * (hi - idx) + arr[hi] * (idx - lo)
            }
            s.evennessHistory.push({
                step: s.currentIdx,
                median: percentile(evennessValues, 50),
                p75: percentile(evennessValues, 75),
                p25: percentile(evennessValues, 25),
                min: evennessValues[0],
            })
        }
    }

    async function loadAndGenerate() {
        if (loading || disposed) return
        loading = true
        clearTimer()
        setState(s => {
            restart(s)
            s.loadStatus = 'loading'
            s.loadProgress = null
        })
        try {
            const device = await initJax()
            if (disposed) return
            setState(s => { s.jaxDevice = device })
            const raw = await fetchMnistRaw({
                onProgress: progress => {
                    if (!disposed) setState(s => { s.loadProgress = progress })
                },
            })
            if (disposed) return
            const { test } = formatMnistForMlp(raw)
            totalImages = test.count
            allImageData = test.images
            setState(s => {
                shuffleSubset(s)
                restart(s)
                s.loadStatus = 'ready'
                s.loadProgress = null
            })
        } catch (error) {
            if (disposed) return
            console.error('MNIST NK: failed to load data', error)
            setState(s => {
                s.loadStatus = 'idle'
                s.loadProgress = null
            })
        } finally {
            loading = false
        }
    }

    function advanceReplayRound(s: State) {
        if (s.currentIdx < shuffledIndices.length || s.replayQueue.length > 0 || s.nextReplayQueue.length === 0) return
        s.replayQueue = [...s.nextReplayQueue]
        s.nextReplayQueue = []
    }

    async function performStep(): Promise<MnistAddPointResult | null> {
        if (disposed) return null
        let work: { dataIndex: number; hasPending: boolean; deferEviction: boolean } | undefined
        setState(s => {
            if (s.loadStatus !== 'ready') return
            advanceReplayRound(s)
            const hasPending = s.currentIdx < shuffledIndices.length
            const dataIndex = hasPending ? shuffledIndices[s.currentIdx] : s.replayQueue[0]
            if (dataIndex === undefined) {
                clearTimer()
                s.isStreaming = false
                return
            }
            work = { dataIndex, hasPending, deferEviction: s.deferEviction }
        })
        if (!work) return null

        // Capture this run's resources before awaiting. A reset invalidates the result.
        const { dataIndex, hasPending, deferEviction } = work
        const run = generation
        const model = clusteringState
        const cache = distanceCache
        const images = allImageData
        const modeDataIndices = model.clusters
            .map(c => getClusterMode(c)?.dataIndex ?? null)
            .filter((index): index is number => index !== null)

        if (modeDataIndices.length > 0) {
            try {
                const distances = await computeBatchDistancesJax(images, dataIndex, modeDataIndices)
                if (run !== generation || disposed) return null
                modeDataIndices.forEach((index, i) => cache.set(dataIndex, index, distances[i]))
            } catch {
                // The CPU distance cache remains a valid fallback when GPU prewarming fails.
            }
        }
        if (run !== generation || disposed) return null
        const result = addMnistPoint(model, dataIndex, cache, images, deferEviction)
        setState(s => {
            if (hasPending) s.currentIdx++
            else s.replayQueue.shift()
            if (deferEviction && result.evicted) {
                const queue = hasPending ? s.replayQueue : s.nextReplayQueue
                queue.push(result.evicted.dataIndex)
            }
            s.lastAddedDataIndex = result.point.dataIndex
            s.lastEvictedDataIndex = result.evicted?.dataIndex ?? null
            advanceReplayRound(s)
            syncState(s)
        })
        return result
    }

    async function stepOnce(): Promise<MnistAddPointResult | null> {
        // Timer ticks and manual steps share one in-flight operation.
        if (activeStep) return activeStep
        const task = performStep()
        activeStep = task
        try {
            return await task
        } finally {
            if (activeStep === task) activeStep = null
        }
    }

    function startStreaming() {
        if (streamingInterval !== null || disposed) return
        setState(s => {
            if (s.loadStatus !== 'ready') return
            if (s.currentIdx >= shuffledIndices.length && s.replayQueue.length === 0 && s.nextReplayQueue.length === 0) return
            const run = generation
            s.isStreaming = true
            streamingInterval = setInterval(() => {
                void stepOnce().catch(error => {
                    if (run !== generation || disposed) return
                    stopStreaming()
                    console.error('MNIST NK: step failed', error)
                })
            }, s.streamSpeed)
        })
    }

    function stopStreaming() {
        clearTimer()
        setState(s => { s.isStreaming = false })
    }

    function reset() {
        clearTimer()
        setState(s => {
            shuffleSubset(s)
            restart(s)
        })
    }

    function dispose() {
        disposed = true
        generation++
        clearTimer()
    }

    function getImagePixelsStore(dataIndex: number): Float32Array {
        if (allImageData.length === 0) return new Float32Array(784)
        return allImageData.subarray(dataIndex * 784, (dataIndex + 1) * 784)
    }

    return {
        n: () => state.n,
        setN: value => configure(s => {
            s.n = value
            if (s.k >= value) s.k = Math.max(1, value - 1)
        }),
        k: () => state.k,
        setK: value => configure(s => {
            s.k = value
            if (s.n <= value) s.n = value + 1
        }),
        subsetSize: () => state.subsetSize,
        setSubsetSize: value => { setState(s => { s.subsetSize = value }) },
        top1Enabled: () => state.top1Enabled,
        setTop1Enabled: value => configure(s => { s.top1Enabled = value }),
        last1ShrinkEnabled: () => state.last1ShrinkEnabled,
        setLast1ShrinkEnabled: value => configure(s => { s.last1ShrinkEnabled = value }),
        deferEviction: () => state.deferEviction,
        setDeferEviction: value => { setState(s => { s.deferEviction = value }) },
        streamSpeed: () => state.streamSpeed,
        setStreamSpeed: value => { setState(s => { s.streamSpeed = value }) },
        loadStatus: () => state.loadStatus,
        loadProgress: () => state.loadProgress,
        jaxDevice: () => state.jaxDevice,
        clusters: () => state.clusters,
        isStreaming: () => state.isStreaming,
        processedCount: () => state.currentIdx,
        totalCount: () => state.totalCount,
        lastAddedDataIndex: () => state.lastAddedDataIndex,
        lastEvictedDataIndex: () => state.lastEvictedDataIndex,
        replayQueueLength: () => state.replayQueue.length + state.nextReplayQueue.length,
        evennessHistory: () => state.evennessHistory,
        streamFinished: () => state.loadStatus === 'ready' && state.currentIdx >= state.totalCount && state.replayQueue.length === 0 && state.nextReplayQueue.length === 0,
        loadAndGenerate, startStreaming, stopStreaming, stepOnce, reset, dispose,
        getImagePixels: getImagePixelsStore,
    }
}
