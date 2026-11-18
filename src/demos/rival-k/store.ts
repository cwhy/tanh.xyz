/**
 * Store for Rival-K Clustering demo
 * Manages streaming points and clustering state
 */

import { createSignal, flush, untrack, createStore } from 'solid-js'
import {
    createRivalKState,
    addPoint,
    resetRivalKState,
    getMaturity,
    type RivalKAddPointResult,
    type ClusterMaturity,
} from './algorithm'
import { getColorForIndex, oklchToCss, adjustLightness, type OklchColor } from '../../lib/visualization/color'
import {
    generateBasicDatasetPoints,
    type BasicDatasetDistribution,
} from '../../lib/datasets/basic'

export interface PointWithCluster {
    point: { x: number; y: number; id: number }
    clusterId: number
    isMedoid: boolean
    color: string
    medoidColor: string
}

export interface ClusterInfo {
    id: number
    size: number
    medoidId: number | null
    radius: number
    kNNDist: number
    evenness: number
    maturity: ClusterMaturity
    color: string
}

export function createRivalKStore() {
    // Parameters
    const [k, setKInternal] = createSignal(3)
    const [cancelReplay, setCancelReplay] = createSignal(false)

    // These imperative algorithms consume committed UI inputs. Flush at action boundaries
    // before reading a value changed by an earlier action in the same event.
    const setK = (newK: number) => {
        flush()
        setKInternal(newK)
        reinitState()
    }

    // Clustering state
    let clusteringState = untrack(() => createRivalKState(k()))

    function reinitState() {
        flush()
        clusteringState = createRivalKState(k())
        setState(s => {
            s.points = []
            s.clusters = []
            s.currentPointIndex = 0
            s.lastAddedPointId = null
            s.lastEvictedPointId = null
            s.replayQueue = []
            s.nextReplayQueue = []
            s.replayRound = 0
            s.replayRoundTotal = 0
            s.replayRoundProcessed = 0
        })
    }

    // Reactive store
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
        replayRound: 0,
        replayRoundTotal: 0,
        replayRoundProcessed: 0,
    })

    const [streamSpeed, setStreamSpeed] = createSignal(200)

    let streamingInterval: ReturnType<typeof setInterval> | null = null

    function getClusterColor(clusterId: number): OklchColor {
        return getColorForIndex(clusterId, { lightness: 0.65, chroma: 0.18 })
    }

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
            radius: c.radius,
            kNNDist: c.kNNDist,
            evenness: c.evenness,
            maturity: getMaturity(c, clusteringState.k),
            color: oklchToCss(getClusterColor(c.id)),
        }))

        setState(s => {
            s.points = pointsWithCluster
            s.clusters = clusterInfos
        })
    }

    function generateRandomPoints(count: number, distribution: BasicDatasetDistribution) {
        flush()
        stopStreaming()
        resetRivalKState(clusteringState)
        clusteringState = createRivalKState(k())

        const points = generateBasicDatasetPoints(count, distribution)

        // Shuffle
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
            s.replayRound = 0
            s.replayRoundTotal = 0
            s.replayRoundProcessed = 0
        })
    }

    function advanceReplayRoundIfNeeded() {
        flush()
        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        if (!hasMorePending && state.replayQueue.length === 0 && state.nextReplayQueue.length > 0) {
            const nextRound = state.replayRound + 1
            const nextTotal = state.nextReplayQueue.length
            setState(s => {
                s.replayQueue = [...s.nextReplayQueue]
                s.nextReplayQueue = []
                s.replayRound = nextRound
                s.replayRoundTotal = nextTotal
                s.replayRoundProcessed = 0
            })
        }
    }

    function stepOnce(): RivalKAddPointResult | null {
        flush()
        advanceReplayRoundIfNeeded()
        flush()

        const hasMorePending = state.currentPointIndex < state.pendingPoints.length
        const hasReplayItems = state.replayQueue.length > 0
        const hasNextReplayItems = state.nextReplayQueue.length > 0

        if (!hasMorePending && !hasReplayItems && !hasNextReplayItems) {
            stopStreaming()
            return null
        }

        let result: RivalKAddPointResult

        if (hasMorePending) {
            const { x, y } = state.pendingPoints[state.currentPointIndex]
            result = addPoint(clusteringState, x, y, cancelReplay())

            setState(s => {
                s.currentPointIndex++
                s.lastAddedPointId = result.point.id
                s.lastEvictedPointId = result.evicted?.id ?? null
                if (cancelReplay() && result.evicted) {
                    s.replayQueue.push({ x: result.evicted.x, y: result.evicted.y })
                    s.replayRoundTotal = s.replayQueue.length
                }
            })
        } else {
            const replayPoint = state.replayQueue[0]
            result = addPoint(clusteringState, replayPoint.x, replayPoint.y, cancelReplay())

            setState(s => {
                s.replayQueue.shift()
                s.replayRoundProcessed++
                s.lastAddedPointId = result.point.id
                s.lastEvictedPointId = result.evicted?.id ?? null
                if (cancelReplay() && result.evicted) {
                    s.nextReplayQueue.push({ x: result.evicted.x, y: result.evicted.y })
                }
            })
        }

        advanceReplayRoundIfNeeded()
        syncState()
        return result
    }

    function addSinglePoint(x: number, y: number): RivalKAddPointResult {
        flush()
        const result = addPoint(clusteringState, x, y)
        setState(s => {
            s.lastAddedPointId = result.point.id
            s.lastEvictedPointId = result.evicted?.id ?? null
        })
        syncState()
        return result
    }

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
            if (!result) stopStreaming()
        }, streamSpeed())
    }

    function stopStreaming() {
        if (streamingInterval) {
            clearInterval(streamingInterval)
            streamingInterval = null
        }
        setState(s => {
            s.isStreaming = false
        })
    }

    function reset() {
        flush()
        stopStreaming()
        clusteringState = createRivalKState(k())
        setState(s => {
            s.points = []
            s.clusters = []
            s.pendingPoints = []
            s.currentPointIndex = 0
            s.lastAddedPointId = null
            s.lastEvictedPointId = null
            s.replayQueue = []
            s.nextReplayQueue = []
            s.replayRound = 0
            s.replayRoundTotal = 0
            s.replayRoundProcessed = 0
        })
    }

    return {
        k,
        setK,
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
        replayQueue: () => state.replayQueue,
        replayQueueLength: () => state.replayQueue.length + state.nextReplayQueue.length,
        replayRound: () => state.replayRound,
        replayRoundTotal: () => state.replayRoundTotal,
        replayRoundProcessed: () => state.replayRoundProcessed,
        generateRandomPoints,
        startStreaming,
        stopStreaming,
        stepOnce,
        reset,
        addSinglePoint,
    }
}
