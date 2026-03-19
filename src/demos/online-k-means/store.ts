/**
 * Store for Online Clustering demo — K-Means / Hard WTA / SoftHebb × Euclidean / Angular
 */

import { createSignal } from 'solid-js'
import { createStore, produce } from 'solid-js/store'
import {
    createOnlineKMeansState,
    addPoint,
    resetOnlineKMeansState,
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
    const [k, setKInternal] = createSignal(5)
    const [algorithm, setAlgorithmInternal] = createSignal<AlgorithmType>('online-k-means')
    const [distanceMetric, setDistanceMetricInternal] = createSignal<DistanceMetric>('euclidean')
    const [learningRate, setLearningRateInternal] = createSignal(0.01)
    const [softmaxBase, setSoftmaxBaseInternal] = createSignal(Math.E)

    const setK = (newK: number) => { setKInternal(newK); reinitState() }
    const setAlgorithm = (v: AlgorithmType) => { setAlgorithmInternal(v); reinitState() }
    const setDistanceMetric = (v: DistanceMetric) => { setDistanceMetricInternal(v); reinitState() }

    const setLearningRate = (v: number) => {
        setLearningRateInternal(v)
        if (clusteringState) clusteringState.learningRate = v
    }
    const setSoftmaxBase = (v: number) => {
        setSoftmaxBaseInternal(v)
        if (clusteringState) clusteringState.softmaxBase = v
    }

    // eslint-disable-next-line solid/reactivity
    let clusteringState = createOnlineKMeansState(k(), algorithm(), distanceMetric(), learningRate(), softmaxBase())

    function reinitState() {
        clusteringState = createOnlineKMeansState(k(), algorithm(), distanceMetric(), learningRate(), softmaxBase())
        setState({ points: [], clusters: [], angularBoundaries: [], currentPointIndex: 0, lastAddedPointId: null })
    }

    const [state, setState] = createStore({
        points: [] as PointWithCluster[],
        clusters: [] as ClusterInfo[],
        angularBoundaries: [] as number[],
        pendingPoints: [] as Array<{ x: number; y: number }>,
        currentPointIndex: 0,
        isStreaming: false,
        lastAddedPointId: null as number | null,
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
        setState({ points: pointsWithCluster, clusters: clusterInfos, angularBoundaries: boundaries })
    }

    function generateRandomPoints(count: number, distribution: BasicDatasetDistribution) {
        stopStreaming()
        clusteringState = createOnlineKMeansState(k(), algorithm(), distanceMetric(), learningRate(), softmaxBase())
        const points = generateBasicDatasetPoints(count, distribution)
        for (let i = points.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1))
                ;[points[i], points[j]] = [points[j], points[i]]
        }
        setState({ pendingPoints: points, currentPointIndex: 0, points: [], clusters: [], angularBoundaries: [], lastAddedPointId: null })
    }

    function stepOnce(): OnlineKMeansAddPointResult | null {
        if (state.currentPointIndex >= state.pendingPoints.length) { stopStreaming(); return null }
        const { x, y } = state.pendingPoints[state.currentPointIndex]
        const result = addPoint(clusteringState, x, y)
        setState(produce(s => { s.currentPointIndex++; s.lastAddedPointId = result.point.id }))
        syncState()
        return result
    }

    function addSinglePoint(x: number, y: number): OnlineKMeansAddPointResult {
        const result = addPoint(clusteringState, x, y)
        setState({ lastAddedPointId: result.point.id })
        syncState()
        return result
    }

    function startStreaming() {
        if (streamingInterval) return
        if (state.currentPointIndex >= state.pendingPoints.length) return
        setState({ isStreaming: true })
        streamingInterval = setInterval(() => { if (!stepOnce()) stopStreaming() }, streamSpeed())
    }

    function stopStreaming() {
        if (streamingInterval) { clearInterval(streamingInterval); streamingInterval = null }
        setState({ isStreaming: false })
    }

    function reset() {
        stopStreaming()
        clusteringState = createOnlineKMeansState(k(), algorithm(), distanceMetric(), learningRate(), softmaxBase())
        setState({ points: [], clusters: [], angularBoundaries: [], pendingPoints: [], currentPointIndex: 0, lastAddedPointId: null })
    }

    return {
        k, setK,
        algorithm, setAlgorithm,
        distanceMetric, setDistanceMetric,
        learningRate, setLearningRate,
        softmaxBase, setSoftmaxBase,
        points: () => state.points,
        clusters: () => state.clusters,
        angularBoundaries: () => state.angularBoundaries,
        isStreaming: () => state.isStreaming,
        streamSpeed, setStreamSpeed,
        pendingPoints: () => state.pendingPoints,
        currentPointIndex: () => state.currentPointIndex,
        lastAddedPointId: () => state.lastAddedPointId,
        generateRandomPoints, startStreaming, stopStreaming, stepOnce, reset, addSinglePoint,
    }
}
