/**
 * Store for NK Clustering demo
 * Manages streaming points and clustering state
 */

import { createSignal } from 'solid-js'
import { createStore, produce } from 'solid-js/store'
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
    medoidId: number | null
    cMin: number
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

    // Validate K < N when setting
    const setN = (newN: number) => {
        if (newN <= k()) {
            setKInternal(Math.max(1, newN - 1))
        }
        setNInternal(newN)
        reinitState()
    }

    const setK = (newK: number) => {
        if (newK >= n()) {
            setNInternal(newK + 1)
        }
        setKInternal(newK)
        reinitState()
    }

    const setCenterType = (type: CenterType) => {
        setCenterTypeInternal(type)
        reinitState()
    }

    // Clustering state
    // eslint-disable-next-line solid/reactivity
    let clusteringState = createNKClusteringState(n(), k(), centerType())

    // Reinitialize state when params change
    function reinitState() {
        clusteringState = createNKClusteringState(n(), k(), centerType())
        setState({
            points: [],
            clusters: [],
            currentPointIndex: 0,
            lastAddedPointId: null,
            lastEvictedPointId: null,
            replayQueue: [],
            nextReplayQueue: [],
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

        const clusterInfos: ClusterInfo[] = clusteringState.clusters.map(c => ({
            id: c.id,
            size: c.points.length,
            medoidId: c.medoidId,
            cMin: c.cMin,
            color: oklchToCss(getClusterColor(c.id)),
        }))

        setState({
            points: pointsWithCluster,
            clusters: clusterInfos,
        })
    }

    /**
     * Generate random points for streaming
     */
    function generateRandomPoints(count: number, distribution: BasicDatasetDistribution) {
        stopStreaming()
        resetState(clusteringState)
        clusteringState = createNKClusteringState(n(), k(), centerType())

        const points = generateBasicDatasetPoints(count, distribution)

        // Shuffle points to simulate streaming order
        for (let i = points.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1))
                ;[points[i], points[j]] = [points[j], points[i]]
        }

        setState({
            pendingPoints: points,
            currentPointIndex: 0,
            points: [],
            clusters: [],
            lastAddedPointId: null,
            lastEvictedPointId: null,
            replayQueue: [],
            nextReplayQueue: [],
        })
    }

    /**
     * If we're done with current replay round, advance to the next one.
     */
    function advanceReplayRoundIfNeeded() {
        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        if (!hasMorePending && state.replayQueue.length === 0 && state.nextReplayQueue.length > 0) {
            setState(produce(s => {
                s.replayQueue = [...s.nextReplayQueue]
                s.nextReplayQueue = []
            }))
        }
    }

    /**
     * Add a single point from pending queue or replay queue
     */
    function stepOnce(): AddPointResult | null {
        advanceReplayRoundIfNeeded()

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

            setState(produce(s => {
                s.currentPointIndex++
                s.lastAddedPointId = result.point.id
                s.lastEvictedPointId = result.evicted?.id ?? null
                // Deferred evictions from round 1 become round 2 replay candidates.
                if (cancelReplay() && result.evicted) {
                    s.replayQueue.push({ x: result.evicted.x, y: result.evicted.y })
                }
            }))
        } else {
            // Process replay queue (after all original points are done)
            const replayPoint = state.replayQueue[0]
            result = addPoint(clusteringState, replayPoint.x, replayPoint.y, cancelReplay())

            setState(produce(s => {
                s.replayQueue.shift()  // Remove processed replay item
                s.lastAddedPointId = result.point.id
                s.lastEvictedPointId = result.evicted?.id ?? null
                // Deferred evictions from replay round r are processed in round r+1.
                if (cancelReplay() && result.evicted) {
                    s.nextReplayQueue.push({ x: result.evicted.x, y: result.evicted.y })
                }
            }))
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
        const result = addPoint(clusteringState, x, y)
        setState({
            lastAddedPointId: result.point.id,
            lastEvictedPointId: result.evicted?.id ?? null,
        })
        syncState()
        return result
    }

    /**
     * Start streaming points
     */
    function startStreaming() {
        if (streamingInterval) return
        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        const hasReplayItems = state.replayQueue.length > 0 || state.nextReplayQueue.length > 0
        if (!hasMorePending && !hasReplayItems) return

        setState({ isStreaming: true })
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
        setState({ isStreaming: false })
    }

    /**
     * Reset everything
     */
    function reset() {
        stopStreaming()
        clusteringState = createNKClusteringState(n(), k(), centerType())
        setState({
            points: [],
            clusters: [],
            pendingPoints: [],
            currentPointIndex: 0,
            lastAddedPointId: null,
            lastEvictedPointId: null,
            replayQueue: [],
            nextReplayQueue: [],
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
