/**
 * Store for Rival-K Clustering demo
 * Manages streaming points and clustering state
 */

import { createStore } from 'solid-js'
import {
    createRivalKState,
    addPoint,
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
    const defaults = {
        k: 3,
        cancelReplay: false,
    }
    // The algorithm owns its mutable internals; only configuration and display data are reactive.
    let clusteringState = createRivalKState(defaults.k)
    const [state, setState] = createStore({
        ...defaults,
        streamSpeed: 200,
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
    type State = typeof state
    let streamingInterval: ReturnType<typeof setInterval> | null = null

    function clearTimer() {
        if (streamingInterval !== null) clearInterval(streamingInterval)
        streamingInterval = null
    }

    function restart(s: State) {
        clusteringState = createRivalKState(s.k)
        s.points = []
        s.clusters = []
        s.currentPointIndex = 0
        s.isStreaming = false
        s.lastAddedPointId = null
        s.lastEvictedPointId = null
        s.replayQueue = []
        s.nextReplayQueue = []
        s.replayRound = 0
        s.replayRoundTotal = 0
        s.replayRoundProcessed = 0
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
        s.replayRound++
        s.replayRoundTotal = s.replayQueue.length
        s.replayRoundProcessed = 0
    }

    function stepOnce(): RivalKAddPointResult | null {
        let result: RivalKAddPointResult | null = null
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
                s.replayRoundProcessed++
            }
            if (s.cancelReplay && added.evicted) {
                const queue = hasPending ? s.replayQueue : s.nextReplayQueue
                queue.push({ x: added.evicted.x, y: added.evicted.y })
                if (hasPending) s.replayRoundTotal = s.replayQueue.length
            }
            advanceReplayRound(s)
            syncState(s)
        })
        return result
    }

    function addSinglePoint(x: number, y: number): RivalKAddPointResult {
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
        k: () => state.k,
        setK: (value: number) => configure(s => { s.k = value }),
        cancelReplay: () => state.cancelReplay,
        setCancelReplay: (value: boolean) => { setState(s => { s.cancelReplay = value }) },
        lastEvictedPointId: () => state.lastEvictedPointId,
        replayQueueLength: () => state.replayQueue.length + state.nextReplayQueue.length,
        replayQueue: () => state.replayQueue,
        replayRound: () => state.replayRound,
        replayRoundTotal: () => state.replayRoundTotal,
        replayRoundProcessed: () => state.replayRoundProcessed,
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
