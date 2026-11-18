/**
 * Store for NK Clustering demo
 * Manages streaming points and clustering state
 */

import { createSignal, flush, untrack, createStore } from 'solid-js'
import {
    createNKClusteringState,
    addPoint,
    resetState,
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
    // Parameters with signals
    const [n, setNInternal] = createSignal(10)
    const [k, setKInternal] = createSignal(3)
    const [centerType, setCenterTypeInternal] = createSignal<CenterType>('medoid')
    const [cancelReplay, setCancelReplay] = createSignal(false)
    const [top1Enabled, setTop1Enabled] = createSignal(false)
    const [last1ShrinkEnabled, setLast1ShrinkEnabled] = createSignal(false)

    // Validate K < N when setting
    // These imperative algorithms consume committed UI inputs. Flush at action boundaries
    // before reading a value changed by an earlier action in the same event.
    const setN = (newN: number) => {
        flush()
        if (newN <= k()) {
            setKInternal(Math.max(1, newN - 1))
        }
        setNInternal(newN)
        reinitState()
    }

    const setK = (newK: number) => {
        flush()
        if (newK >= n()) {
            setNInternal(newK + 1)
        }
        setKInternal(newK)
        reinitState()
    }

    const setCenterType = (type: CenterType) => {
        flush()
        setCenterTypeInternal(type)
        reinitState()
    }

    // Clustering state
    let clusteringState = untrack(() => createNKClusteringState(n(), k(), centerType(), top1Enabled(), last1ShrinkEnabled()))

    // Reinitialize state when params change
    function reinitState() {
        flush()
        clusteringState = createNKClusteringState(
            n(),
            k(),
            centerType(),
            top1Enabled(),
            last1ShrinkEnabled()
        )
        setState(s => {
            s.points = []
            s.clusters = []
            s.currentPointIndex = 0
            s.lastAddedPointId = null
            s.lastEvictedPointId = null
            s.replayQueue = []
            s.nextReplayQueue = []
        })
    }

    // Reactive store for UI
    const [state, setState] = createStore({
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

    const [streamSpeed, setStreamSpeed] = createSignal(500)  // ms between points

    let streamingInterval: ReturnType<typeof setInterval> | null = null

    /**
     * Get color for a cluster
     */
    function getClusterColor(clusterId: number): OklchColor {
        return getColorForIndex(clusterId, { lightness: 0.65, chroma: 0.18 })
    }

    /**
     * Sync internal state to reactive store
     */
    function syncState() {
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

        setState(s => {
            s.points = pointsWithCluster
            s.clusters = clusterInfos
        })
    }

    /**
     * Generate random points for streaming
     */
    function generateRandomPoints(count: number, distribution: BasicDatasetDistribution) {
        flush()
        stopStreaming()
        resetState(clusteringState)
        clusteringState = createNKClusteringState(
            n(),
            k(),
            centerType(),
            top1Enabled(),
            last1ShrinkEnabled()
        )

        const points = generateBasicDatasetPoints(count, distribution)

        // Shuffle points to simulate streaming order
        for (let i = points.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1))
                ;[points[i], points[j]] = [points[j], points[i]]
        }

        setState(s => {
            s.pendingPoints = points
            s.currentPointIndex = 0
            s.points = []
            s.clusters = []
            s.lastAddedPointId = null
            s.lastEvictedPointId = null
            s.replayQueue = []
            s.nextReplayQueue = []
        })
    }

    /**
     * If we're done with current replay round, advance to the next one.
     */
    function advanceReplayRoundIfNeeded() {
        flush()
        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        if (!hasMorePending && state.replayQueue.length === 0 && state.nextReplayQueue.length > 0) {
            setState(s => {
                s.replayQueue = [...s.nextReplayQueue]
                s.nextReplayQueue = []
            })
        }
    }

    /**
     * Add a single point from pending queue or replay queue
     */
    function stepOnce(): AddPointResult | null {
        flush()
        advanceReplayRoundIfNeeded()
        flush()

        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        const hasReplayItems = state.replayQueue.length > 0
        const hasNextReplayItems = state.nextReplayQueue.length > 0

        // If no more pending points and no replay items (current or next round), we're done
        if (!hasMorePending && !hasReplayItems && !hasNextReplayItems) {
            stopStreaming()
            return null
        }

        let result: AddPointResult

        if (hasMorePending) {
            // Process pending points first
            const { x, y } = state.pendingPoints[state.currentPointIndex]
            result = addPoint(clusteringState, x, y, cancelReplay())

            setState(s => {
                s.currentPointIndex++
                s.lastAddedPointId = result.point.id
                s.lastEvictedPointId = result.evicted?.id ?? null
                // Deferred evictions from round 1 become round 2 replay candidates.
                if (cancelReplay() && result.evicted) {
                    s.replayQueue.push({ x: result.evicted.x, y: result.evicted.y })
                }
            })
        } else {
            // Process replay queue (after all original points are done)
            const replayPoint = state.replayQueue[0]
            result = addPoint(clusteringState, replayPoint.x, replayPoint.y, cancelReplay())

            setState(s => {
                s.replayQueue.shift()  // Remove processed replay item
                s.lastAddedPointId = result.point.id
                s.lastEvictedPointId = result.evicted?.id ?? null
                // Deferred evictions from replay round r are processed in round r+1.
                if (cancelReplay() && result.evicted) {
                    s.nextReplayQueue.push({ x: result.evicted.x, y: result.evicted.y })
                }
            })
        }

        // Move to the next replay round once current replay queue is exhausted.
        advanceReplayRoundIfNeeded()
        syncState()
        return result
    }

    /**
     * Add a point directly (for click-to-add)
     */
    function addSinglePoint(x: number, y: number): AddPointResult {
        flush()
        const result = addPoint(clusteringState, x, y)
        setState(s => {
            s.lastAddedPointId = result.point.id
            s.lastEvictedPointId = result.evicted?.id ?? null
        })
        syncState()
        return result
    }

    /**
     * Start streaming points
     */
    function startStreaming() {
        flush()
        if (streamingInterval) return
        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        const hasReplayItems = state.replayQueue.length > 0 || state.nextReplayQueue.length > 0
        if (!hasMorePending && !hasReplayItems) return

        setState(s => {
            s.isStreaming = true
        })
        streamingInterval = setInterval(() => {
            const result = stepOnce()
            if (!result) {
                stopStreaming()
            }
        }, streamSpeed())
    }

    /**
     * Stop streaming
     */
    function stopStreaming() {
        if (streamingInterval) {
            clearInterval(streamingInterval)
            streamingInterval = null
        }
        setState(s => {
            s.isStreaming = false
        })
    }

    /**
     * Reset everything
     */
    function reset() {
        flush()
        stopStreaming()
        clusteringState = createNKClusteringState(
            n(),
            k(),
            centerType(),
            top1Enabled(),
            last1ShrinkEnabled()
        )
        setState(s => {
            s.points = []
            s.clusters = []
            s.pendingPoints = []
            s.currentPointIndex = 0
            s.lastAddedPointId = null
            s.lastEvictedPointId = null
            s.replayQueue = []
            s.nextReplayQueue = []
        })
    }

    return {
        n,
        setN,
        k,
        setK,
        centerType,
        setCenterType,
        cancelReplay,
        setCancelReplay,
        top1Enabled,
        setTop1Enabled: (enabled: boolean) => { setTop1Enabled(enabled); reinitState() },
        last1ShrinkEnabled,
        setLast1ShrinkEnabled: (enabled: boolean) => { setLast1ShrinkEnabled(enabled); reinitState() },
        points: () => state.points,
        clusters: () => state.clusters,
        isStreaming: () => state.isStreaming,
        streamSpeed,
        setStreamSpeed,
        pendingPoints: () => state.pendingPoints,
        currentPointIndex: () => state.currentPointIndex,
        lastAddedPointId: () => state.lastAddedPointId,
        lastEvictedPointId: () => state.lastEvictedPointId,
        replayQueueLength: () => state.replayQueue.length + state.nextReplayQueue.length,
        generateRandomPoints,
        startStreaming,
        stopStreaming,
        stepOnce,
        reset,
        addSinglePoint,
    }
}
