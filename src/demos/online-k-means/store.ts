/**
 * Store for Online Clustering demo — K-Means / Hard WTA / SoftHebb × Euclidean / Angular
 */

import { createStore } from 'solid-js'
import {
    createOnlineKMeansState,
    addPoint,
    clusterRadius,
    centroidAngle,
    computeAngularBoundaries,
    type AlgorithmType,
    type DistanceMetric,
    type OnlineKMeansAddPointResult,
} from './algorithm'
import { getColorForIndex, oklchToCss, adjustLightness, type OklchColor } from '../../lib/visualization/color'
import {
    generateBasicDatasetPoints,
    type BasicDatasetDistribution,
} from '../../lib/datasets/basic'

export interface PointWithCluster {
    point: { x: number; y: number; id: number }
    clusterId: number
    color: string
}

export interface ClusterInfo {
    id: number
    size: number
    centroidX: number
    centroidY: number
    radius: number
    angle: number
    color: string
    centroidColor: string
    bias: number
}

export function createOnlineKMeansStore() {
    const defaults = {
        k: 5,
        algorithm: 'online-k-means' as AlgorithmType,
        distanceMetric: 'euclidean' as DistanceMetric,
        learningRate: 0.01,
        softmaxBase: Math.E,
    }
    // The algorithm owns its mutable internals; only configuration and display data are reactive.
    let clusteringState = createOnlineKMeansState(defaults.k, defaults.algorithm, defaults.distanceMetric, defaults.learningRate, defaults.softmaxBase)
    const [state, setState] = createStore({
        ...defaults,
        streamSpeed: 200,
        points: [] as PointWithCluster[],
        clusters: [] as ClusterInfo[],
        pendingPoints: [] as Array<{ x: number; y: number }>,
        currentPointIndex: 0,
        isStreaming: false,
        lastAddedPointId: null as number | null,
        angularBoundaries: [] as number[],
    })
    type State = typeof state
    let streamingInterval: ReturnType<typeof setInterval> | null = null

    function clearTimer() {
        if (streamingInterval !== null) clearInterval(streamingInterval)
        streamingInterval = null
    }

    function restart(s: State) {
        clusteringState = createOnlineKMeansState(s.k, s.algorithm, s.distanceMetric, s.learningRate, s.softmaxBase)
        s.points = []
        s.clusters = []
        s.currentPointIndex = 0
        s.isStreaming = false
        s.lastAddedPointId = null
        s.angularBoundaries = []
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
            for (const point of cluster.points) {
                pointsWithCluster.push({ point, clusterId: cluster.id, color: colorCss })
            }
        }

        const isAngular = clusteringState.distanceMetric === 'angular'
        const clusterInfos: ClusterInfo[] = clusteringState.clusters.map(c => {
            const color = getClusterColor(c.id)
            return {
                id: c.id,
                size: c.points.length,
                centroidX: c.centroidX,
                centroidY: c.centroidY,
                radius: isAngular ? 0 : clusterRadius(c),
                angle: centroidAngle(c),
                color: oklchToCss(color),
                centroidColor: oklchToCss(adjustLightness(color, -0.2)),
                bias: c.bias,
            }
        })

        const boundaries = isAngular ? computeAngularBoundaries(clusteringState) : []
        s.points = pointsWithCluster
        s.clusters = clusterInfos
        s.angularBoundaries = boundaries
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

    function stepOnce(): OnlineKMeansAddPointResult | null {
        let result: OnlineKMeansAddPointResult | null = null
        // A draft sees preceding writes in this event without flushing unrelated UI work.
        setState(s => {
            const hasPending = s.currentPointIndex < s.pendingPoints.length
            const point = hasPending ? s.pendingPoints[s.currentPointIndex] : undefined
            if (!point) {
                clearTimer()
                s.isStreaming = false
                return
            }
            const added = addPoint(clusteringState, point.x, point.y)
            result = added
            s.lastAddedPointId = added.point.id
            s.currentPointIndex++
            syncState(s)
        })
        return result
    }

    function addSinglePoint(x: number, y: number): OnlineKMeansAddPointResult {
        const result = addPoint(clusteringState, x, y)
        setState(s => {
            s.lastAddedPointId = result.point.id
            syncState(s)
        })
        return result
    }

    function startStreaming() {
        if (streamingInterval !== null) return
        setState(s => {
            if (s.currentPointIndex >= s.pendingPoints.length) return
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
        algorithm: () => state.algorithm,
        setAlgorithm: (value: AlgorithmType) => configure(s => { s.algorithm = value }),
        distanceMetric: () => state.distanceMetric,
        setDistanceMetric: (value: DistanceMetric) => configure(s => { s.distanceMetric = value }),
        learningRate: () => state.learningRate,
        setLearningRate: (value: number) => {
            clusteringState.learningRate = value
            setState(s => { s.learningRate = value })
        },
        softmaxBase: () => state.softmaxBase,
        setSoftmaxBase: (value: number) => {
            clusteringState.softmaxBase = value
            setState(s => { s.softmaxBase = value })
        },
        angularBoundaries: () => state.angularBoundaries,
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
