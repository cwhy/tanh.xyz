/**
 * Store for MNIST NK Clustering demo.
 * Manages MNIST data loading, streaming, and NK clustering state.
 *
 * Note: evictions are handled synchronously inside addMnistPoint (deferEviction=false),
 * so no replay queue is needed here.
 */

import { createSignal } from 'solid-js'
import { createStore } from 'solid-js/store'
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
    type MnistNKState,
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

    // Evenness history
    evennessHistory: () => EvennessSnapshot[]
    streamFinished: () => boolean

    // Actions
    loadAndGenerate: () => Promise<void>
    startStreaming: () => void
    stopStreaming: () => void
    stepOnce: () => Promise<MnistAddPointResult | null>
    reset: () => void

    // Data access
    getImagePixels: (dataIndex: number) => Float32Array
}

// ---------------------------------------------------------------------------
// Store factory
// ---------------------------------------------------------------------------

export function createMnistNKStore(): MnistNKStore {
    // Parameters
    const [n, setNInternal] = createSignal(15)
    const [k, setKInternal] = createSignal(3)
    const [subsetSize, setSubsetSize] = createSignal(500)
    const [top1Enabled, setTop1Enabled] = createSignal(false)
    const [last1ShrinkEnabled, setLast1ShrinkEnabled] = createSignal(false)
    const [streamSpeed, setStreamSpeed] = createSignal(200)
    const [jaxDevice, setJaxDevice] = createSignal<string | null>(null)

    const setN = (newN: number) => {
        if (newN <= k()) setKInternal(Math.max(1, newN - 1))
        setNInternal(newN)
        reinitClustering()
    }

    const setK = (newK: number) => {
        if (newK >= n()) setNInternal(newK + 1)
        setKInternal(newK)
        reinitClustering()
    }

    // Raw data (loaded once)
    let allImageData: Float32Array = new Float32Array(0)
    let totalImages = 0

    // Per-session data (reshuffled on generate/reset)
    let shuffledIndices: number[] = []
    let distanceCache: DistanceCache = new DistanceCache(0)
    // eslint-disable-next-line solid/reactivity
    let clusteringState: MnistNKState = createMnistNKState(n(), k(), top1Enabled(), last1ShrinkEnabled())

    const [state, setState] = createStore({
        loadStatus: 'idle' as LoadStatus,
        loadProgress: null as MnistFetchProgress | null,
        isStreaming: false,
        currentIdx: 0,
        clusters: [] as ClusterDisplayInfo[],
        lastAddedDataIndex: null as number | null,
        lastEvictedDataIndex: null as number | null,
    })

    let streamingInterval: ReturnType<typeof setInterval> | null = null

    // Evenness history
    let evennessHistoryArr: EvennessSnapshot[] = []
    const [historyVersion, setHistoryVersion] = createSignal(0)

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    function getClusterColor(clusterId: number): OklchColor {
        return getColorForIndex(clusterId, { lightness: 0.65, chroma: 0.18 })
    }

    function syncState() {
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

        setState({ clusters: displays })

        // Record evenness history snapshot
        if (clusters.length > 0) {
            const evennessValues = clusters.map(c => c.evenness).sort((a, b) => a - b)
            const percentile = (arr: number[], p: number) => {
                const idx = (p / 100) * (arr.length - 1)
                const lo = Math.floor(idx)
                const hi = Math.ceil(idx)
                return lo === hi ? arr[lo] : arr[lo] * (hi - idx) + arr[hi] * (idx - lo)
            }
            evennessHistoryArr.push({
                step: state.currentIdx,
                median: percentile(evennessValues, 50),
                p75: percentile(evennessValues, 75),
                p25: percentile(evennessValues, 25),
                min: evennessValues[0],
            })
            setHistoryVersion(v => v + 1)
        }
    }

    function reinitClustering() {
        stopStreaming()
        clusteringState = createMnistNKState(n(), k(), top1Enabled(), last1ShrinkEnabled())
        evennessHistoryArr = []
        setHistoryVersion(0)
        setState({
            clusters: [],
            currentIdx: 0,
            lastAddedDataIndex: null,
            lastEvictedDataIndex: null,
        })
        if (allImageData.length > 0) {
            distanceCache = new DistanceCache(totalImages)
        }
    }

    // ---------------------------------------------------------------------------
    // MNIST loading
    // ---------------------------------------------------------------------------

    async function loadAndGenerate() {
        if (state.loadStatus === 'loading') return

        setState({ loadStatus: 'loading', loadProgress: null })

        try {
            const device = await initJax()
            setJaxDevice(device)

            const raw = await fetchMnistRaw({
                onProgress: (p) => setState({ loadProgress: p }),
            })
            const dataset = formatMnistForMlp(raw)

            // Use test split (10k images), trimmed to subsetSize
            const split = dataset.test
            totalImages = split.count
            allImageData = split.images  // [count * 784] Float32Array, normalized [0,1]

            distanceCache = new DistanceCache(totalImages)

            // Pick a random shuffled subset as the streaming order
            const indices = Array.from({ length: totalImages }, (_, i) => i)
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1))
                    ;[indices[i], indices[j]] = [indices[j], indices[i]]
            }
            shuffledIndices = indices.slice(0, subsetSize())

            clusteringState = createMnistNKState(n(), k(), top1Enabled(), last1ShrinkEnabled())
            setState({
                loadStatus: 'ready',
                loadProgress: null,
                clusters: [],
                currentIdx: 0,
                lastAddedDataIndex: null,
                lastEvictedDataIndex: null,
            })
        } catch (err) {
            console.error('MNIST NK: failed to load data', err)
            setState({ loadStatus: 'idle', loadProgress: null })
        }
    }

    // ---------------------------------------------------------------------------
    // Streaming
    // ---------------------------------------------------------------------------

    async function stepOnce(): Promise<MnistAddPointResult | null> {
        if (state.currentIdx >= shuffledIndices.length) {
            stopStreaming()
            return null
        }

        const dataIndex = shuffledIndices[state.currentIdx]

        // Pre-warm the distance cache: batch-compute distances from this image
        // to all current cluster modes in one JAX-JS call
        const modeDataIndices = clusteringState.clusters
            .map(c => getClusterMode(c)?.dataIndex ?? null)
            .filter((idx): idx is number => idx !== null)

        if (modeDataIndices.length > 0) {
            try {
                const batchDists = await computeBatchDistancesJax(allImageData, dataIndex, modeDataIndices)
                for (let i = 0; i < modeDataIndices.length; i++) {
                    distanceCache.set(dataIndex, modeDataIndices[i], batchDists[i])
                }
            } catch (_) {
                // Fallback: cache will compute on demand
            }
        }

        // Evictions are handled synchronously inside addMnistPoint
        const result = addMnistPoint(clusteringState, dataIndex, distanceCache, allImageData)

        setState({
            currentIdx: state.currentIdx + 1,
            lastAddedDataIndex: result.point.dataIndex,
            lastEvictedDataIndex: result.evicted?.dataIndex ?? null,
        })

        syncState()
        return result
    }

    function startStreaming() {
        if (streamingInterval) return
        if (state.loadStatus !== 'ready') return
        if (state.currentIdx >= shuffledIndices.length) return

        setState({ isStreaming: true })
        streamingInterval = setInterval(async () => {
            const result = await stepOnce()
            if (!result) stopStreaming()
        }, streamSpeed())
    }

    function stopStreaming() {
        if (streamingInterval) {
            clearInterval(streamingInterval)
            streamingInterval = null
        }
        setState({ isStreaming: false })
    }

    function reset() {
        stopStreaming()
        evennessHistoryArr = []
        setHistoryVersion(0)
        if (allImageData.length > 0) {
            const indices = Array.from({ length: totalImages }, (_, i) => i)
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1))
                    ;[indices[i], indices[j]] = [indices[j], indices[i]]
            }
            shuffledIndices = indices.slice(0, subsetSize())
            distanceCache = new DistanceCache(totalImages)
        }
        clusteringState = createMnistNKState(n(), k(), top1Enabled(), last1ShrinkEnabled())
        setState({
            clusters: [],
            currentIdx: 0,
            lastAddedDataIndex: null,
            lastEvictedDataIndex: null,
        })
    }

    function getImagePixelsStore(dataIndex: number): Float32Array {
        if (allImageData.length === 0) return new Float32Array(784)
        const start = dataIndex * 784
        return allImageData.subarray(start, start + 784)
    }

    return {
        n,
        setN,
        k,
        setK,
        subsetSize,
        setSubsetSize,
        top1Enabled,
        setTop1Enabled,
        last1ShrinkEnabled,
        setLast1ShrinkEnabled,
        streamSpeed,
        setStreamSpeed,
        loadStatus: () => state.loadStatus,
        loadProgress: () => state.loadProgress,
        jaxDevice,
        clusters: () => state.clusters,
        isStreaming: () => state.isStreaming,
        processedCount: () => state.currentIdx,
        totalCount: () => shuffledIndices.length,
        lastAddedDataIndex: () => state.lastAddedDataIndex,
        lastEvictedDataIndex: () => state.lastEvictedDataIndex,
        evennessHistory: () => { historyVersion(); return evennessHistoryArr },
        streamFinished: () => state.loadStatus === 'ready' && state.currentIdx >= shuffledIndices.length,
        loadAndGenerate,
        startStreaming,
        stopStreaming,
        stepOnce,
        reset,
        getImagePixels: getImagePixelsStore,
    }
}
