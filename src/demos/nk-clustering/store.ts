/**
 * Store for NK Clustering demo
 * Manages streaming points and clustering state
 */

import { createStore } from 'solid-js'
import {
    createNKClusteringState,
    addPoint,
    type Point,
    type AddPointResult,
    type CenterType,
} from './algorithm'
import { getColorForIndex, oklchToCss, adjustLightness, type OklchColor } from '../../lib/visualization/color'
import {
    generateBasicDatasetPoints,
    type BasicDatasetDistribution,
} from '../../lib/datasets/basic'

export interface PointWithCluster {
    point: Point
    clusterId: number
    isMedoid: boolean
    color: string
    medoidColor: string
}

export interface ClusterInfo {
    id: number
    size: number
    maxSize: number
    medoidId: number | null
    cMin: number
    evenness: number
    isTopHalf: boolean
    isLastOne: boolean
    color: string
}

export interface NKClusteringStore {
    // Parameters
    n: () => number
    setN: (n: number) => void
    k: () => number
    setK: (k: number) => void
    centerType: () => CenterType
    setCenterType: (type: CenterType) => void
    cancelReplay: () => boolean
    setCancelReplay: (on: boolean) => void
    top1Enabled: () => boolean
    setTop1Enabled: (enabled: boolean) => void
    last1ShrinkEnabled: () => boolean
    setLast1ShrinkEnabled: (enabled: boolean) => void

    // State
    points: () => PointWithCluster[]
    clusters: () => ClusterInfo[]
    isStreaming: () => boolean
    streamSpeed: () => number
    setStreamSpeed: (speed: number) => void
    pendingPoints: () => Array<{ x: number; y: number }>
    currentPointIndex: () => number
    lastAddedPointId: () => number | null
    lastEvictedPointId: () => number | null
    replayQueueLength: () => number

    // Actions
    generateRandomPoints: (count: number, distribution: BasicDatasetDistribution) => void
    startStreaming: () => void
    stopStreaming: () => void
    stepOnce: () => AddPointResult | null
    reset: () => void
    addSinglePoint: (x: number, y: number) => AddPointResult
}

export function createNKClusteringStore(): NKClusteringStore {
    const defaults = {
        n: 10,
        k: 3,
        centerType: 'medoid' as CenterType,
        cancelReplay: false,
        top1Enabled: false,
        last1ShrinkEnabled: false,
    }
    // The algorithm owns its mutable internals; only configuration and display data are reactive.
    let clusteringState = createNKClusteringState(defaults.n, defaults.k, defaults.centerType, defaults.top1Enabled, defaults.last1ShrinkEnabled)
    const [state, setState] = createStore({
        ...defaults,
        streamSpeed: 500,
        points: [] as PointWithCluster[],
        clusters: [] as ClusterInfo[],
        pendingPoints: [] as Array<{ x: number; y: number }>,
        currentPointIndex: 0,
        isStreaming: false,
        lastAddedPointId: null as number | null,
        lastEvictedPointId: null as number | null,
        replayQueue: [] as Array<{ x: number; y: number }>,
        nextReplayQueue: [] as Array<{ x: number; y: number }>,
    })
    type State = typeof state
    let streamingInterval: ReturnType<typeof setInterval> | null = null

    function clearTimer() {
        if (streamingInterval !== null) clearInterval(streamingInterval)
        streamingInterval = null
    }

    function restart(s: State) {
        clusteringState = createNKClusteringState(s.n, s.k, s.centerType, s.top1Enabled, s.last1ShrinkEnabled)
        s.points = []
        s.clusters = []
        s.currentPointIndex = 0
        s.isStreaming = false
        s.lastAddedPointId = null
        s.lastEvictedPointId = null
        s.replayQueue = []
        s.nextReplayQueue = []
    }

    function configure(update: (draft: State) => void) {
        clearTimer()
        setState(s => {
            update(s)
            restart(s)
        })
    }

    function getClusterColor(clusterId: number): OklchColor {
        return getColorForIndex(clusterId, { lightness: 0.65, chroma: 0.18 })
    }

    /**
     * Sync internal state to reactive store
     */
    function syncState(s: State) {
        const pointsWithCluster: PointWithCluster[] = []

        for (const cluster of clusteringState.clusters) {
            const color = getClusterColor(cluster.id)
            const colorCss = oklchToCss(color)
            const medoidColor = oklchToCss(adjustLightness(color, -0.2))

            for (const point of cluster.points) {
                pointsWithCluster.push({
                    point,
                    clusterId: cluster.id,
                    isMedoid: point.id === cluster.medoidId,
                    color: colorCss,
                    medoidColor,
                })
            }
        }

        const sortedByEvenness = [...clusteringState.clusters].sort((a, b) => b.evenness - a.evenness)
        const halfCount = Math.ceil(sortedByEvenness.length / 2)
        const topHalfIds = new Set(sortedByEvenness.slice(0, halfCount).map(c => c.id))
        const lastOneId = sortedByEvenness.length > 0 ? sortedByEvenness[sortedByEvenness.length - 1].id : null

        const clusterInfos: ClusterInfo[] = clusteringState.clusters.map(c => ({
            id: c.id,
            size: c.points.length,
            maxSize: c.maxSize,
            medoidId: c.medoidId,
            cMin: c.cMin,
            evenness: c.evenness,
            isTopHalf: topHalfIds.has(c.id),
            isLastOne: c.id === lastOneId,
            color: oklchToCss(getClusterColor(c.id)),
        }))

        s.points = pointsWithCluster
        s.clusters = clusterInfos
    }

    function generateRandomPoints(count: number, distribution: BasicDatasetDistribution) {
        const points = generateBasicDatasetPoints(count, distribution)
        for (let i = points.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1))
            ;[points[i], points[j]] = [points[j], points[i]]
        }
        clearTimer()
        setState(s => {
            restart(s)
            s.pendingPoints = points
        })
    }

    function advanceReplayRound(s: State) {
        if (s.currentPointIndex < s.pendingPoints.length || s.replayQueue.length > 0 || s.nextReplayQueue.length === 0) return
        s.replayQueue = [...s.nextReplayQueue]
        s.nextReplayQueue = []
    }

    function stepOnce(): AddPointResult | null {
        let result: AddPointResult | null = null
        // A draft sees preceding writes in this event without flushing unrelated UI work.
        setState(s => {
            advanceReplayRound(s)
            const hasPending = s.currentPointIndex < s.pendingPoints.length
            const point = hasPending ? s.pendingPoints[s.currentPointIndex] : s.replayQueue[0]
            if (!point) {
                clearTimer()
                s.isStreaming = false
                return
            }
            const added = addPoint(clusteringState, point.x, point.y, s.cancelReplay)
            result = added
            s.lastAddedPointId = added.point.id
            s.lastEvictedPointId = added.evicted?.id ?? null
            if (hasPending) s.currentPointIndex++
            else {
                s.replayQueue.shift()
            }
            if (s.cancelReplay && added.evicted) {
                const queue = hasPending ? s.replayQueue : s.nextReplayQueue
                queue.push({ x: added.evicted.x, y: added.evicted.y })
            }
            advanceReplayRound(s)
            syncState(s)
        })
        return result
    }

    function addSinglePoint(x: number, y: number): AddPointResult {
        const result = addPoint(clusteringState, x, y)
        setState(s => {
            s.lastAddedPointId = result.point.id
            s.lastEvictedPointId = result.evicted?.id ?? null
            syncState(s)
        })
        return result
    }

    function startStreaming() {
        if (streamingInterval !== null) return
        setState(s => {
            if (s.currentPointIndex >= s.pendingPoints.length && s.replayQueue.length === 0 && s.nextReplayQueue.length === 0) return
            s.isStreaming = true
            streamingInterval = setInterval(stepOnce, s.streamSpeed)
        })
    }

    function stopStreaming() {
        clearTimer()
        setState(s => { s.isStreaming = false })
    }

    function reset() {
        clearTimer()
        setState(s => {
            restart(s)
            s.pendingPoints = []
        })
    }

    return {
        n: () => state.n,
        setN: (value: number) => configure(s => {
            s.n = value
            if (s.k >= value) s.k = Math.max(1, value - 1)
        }),
        k: () => state.k,
        setK: (value: number) => configure(s => {
            s.k = value
            if (s.n <= value) s.n = value + 1
        }),
        centerType: () => state.centerType,
        setCenterType: (value: CenterType) => configure(s => { s.centerType = value }),
        top1Enabled: () => state.top1Enabled,
        setTop1Enabled: (value: boolean) => configure(s => { s.top1Enabled = value }),
        last1ShrinkEnabled: () => state.last1ShrinkEnabled,
        setLast1ShrinkEnabled: (value: boolean) => configure(s => { s.last1ShrinkEnabled = value }),
        cancelReplay: () => state.cancelReplay,
        setCancelReplay: (value: boolean) => { setState(s => { s.cancelReplay = value }) },
        lastEvictedPointId: () => state.lastEvictedPointId,
        replayQueueLength: () => state.replayQueue.length + state.nextReplayQueue.length,
        points: () => state.points,
        clusters: () => state.clusters,
        isStreaming: () => state.isStreaming,
        streamSpeed: () => state.streamSpeed,
        setStreamSpeed: (value: number) => { setState(s => { s.streamSpeed = value }) },
        pendingPoints: () => state.pendingPoints,
        currentPointIndex: () => state.currentPointIndex,
        lastAddedPointId: () => state.lastAddedPointId,
        generateRandomPoints, startStreaming, stopStreaming, stepOnce, reset, addSinglePoint,
    }
}
