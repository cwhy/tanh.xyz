/**
 * NK Clustering Algorithm for MNIST
 *
 * Adapts the NK online clustering algorithm to high-dimensional image data.
 * Key differences from the 2D version:
 * - Points are identified by MNIST dataset indices; pixel distance is computed via DistanceCache
 * - Evenness = d(mode, k-th NN) / d(mode, furthest point in cluster)
 * - Top-1 mechanism: most even cluster grows instead of evicting when full
 */

import type { DistanceCache } from '../../lib/clustering/distance'

export interface MnistPoint {
    /** Local sequential ID assigned during clustering */
    id: number
    /** Index into the MNIST dataset (used to look up pixel data) */
    dataIndex: number
}

export interface MnistCluster {
    id: number
    points: MnistPoint[]
    /** Local ID of the mode (medoid) point */
    medoidId: number | null
    /** Max distance from mode to any point = admission radius */
    radius: number
    /** K-th nearest neighbor distance from mode = evenness numerator */
    kNNDist: number
    /** evenness = kNNDist / radius; 1 if cluster has ≤ k points */
    evenness: number
    /** Normally equals global N, but increases permanently under top-1 mechanism */
    maxSize: number
}

export interface MnistNKState {
    clusters: MnistCluster[]
    n: number
    k: number
    top1Enabled: boolean
    last1ShrinkEnabled: boolean
    nextPointId: number
    nextClusterId: number
}

export interface MnistAddPointResult {
    point: MnistPoint
    clusterId: number
    evicted: MnistPoint | null
    newClusterCreated: boolean
    /** True when the top-1 cluster expanded its capacity instead of evicting */
    top1Expanded: boolean
    /** True when the evicting cluster's maxSize was reduced by 1 (evicted point is still reclustered) */
    last1Shrunk: boolean
}

// ---------------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------------

export function createMnistNKState(
    n: number,
    k: number,
    top1Enabled: boolean = false,
    last1ShrinkEnabled: boolean = false
): MnistNKState {
    if (k >= n) throw new Error(`K (${k}) must be less than N (${n})`)
    return { clusters: [], n, k, top1Enabled, last1ShrinkEnabled, nextPointId: 0, nextClusterId: 0 }
}

export function resetMnistNKState(state: MnistNKState): void {
    state.clusters = []
    state.nextPointId = 0
    state.nextClusterId = 0
}

// ---------------------------------------------------------------------------
// Distance helpers (use cache)
// ---------------------------------------------------------------------------

function dist(a: MnistPoint, b: MnistPoint, cache: DistanceCache, imageData: Float32Array): number {
    return cache.compute(a.dataIndex, b.dataIndex, imageData)
}

// ---------------------------------------------------------------------------
// Mode (medoid) computation
// ---------------------------------------------------------------------------

/**
 * Find the K-th nearest neighbor distance for a point within a set of points.
 */
function kthNNDist(point: MnistPoint, points: MnistPoint[], k: number, cache: DistanceCache, imageData: Float32Array): number {
    const others = points.filter(p => p.id !== point.id)
    if (others.length < k) return Infinity
    const distances = others.map(p => dist(point, p, cache, imageData)).sort((a, b) => a - b)
    return distances[k - 1]
}

/**
 * Find the mode (medoid) – the point with the lowest k-th NN distance within the cluster.
 * Falls back to the point minimizing total distance when the cluster has ≤ k points.
 */
export function calculateMode(
    points: MnistPoint[],
    k: number,
    cache: DistanceCache,
    imageData: Float32Array
): MnistPoint | null {
    if (points.length === 0) return null
    if (points.length === 1) return points[0]

    if (points.length <= k) {
        // Too few points for k-NN: find point minimizing sum of distances to all others
        let best = points[0]
        let bestSum = Infinity
        for (const candidate of points) {
            const sum = points
                .filter(p => p.id !== candidate.id)
                .reduce((acc, p) => acc + dist(candidate, p, cache, imageData), 0)
            if (sum < bestSum) {
                bestSum = sum
                best = candidate
            }
        }
        return best
    }

    let best = points[0]
    let bestDist = Infinity
    for (const candidate of points) {
        const d = kthNNDist(candidate, points, k, cache, imageData)
        if (d < bestDist) {
            bestDist = d
            best = candidate
        }
    }
    return best
}

/**
 * Get the current mode point of a cluster.
 */
export function getClusterMode(cluster: MnistCluster): MnistPoint | null {
    if (cluster.medoidId === null) return null
    return cluster.points.find(p => p.id === cluster.medoidId) ?? null
}

// ---------------------------------------------------------------------------
// Evenness
// ---------------------------------------------------------------------------

/**
 * Compute evenness = d(mode, k-th NN) / d(mode, furthest).
 * Returns 1 when there are ≤ k other points (not enough to measure looseness).
 */
export function calculateEvenness(
    mode: MnistPoint,
    points: MnistPoint[],
    k: number,
    radius: number,
    cache: DistanceCache,
    imageData: Float32Array
): number {
    if (points.length <= 1 || radius === 0) return 1
    const others = points.filter(p => p.id !== mode.id)
    if (others.length < k) return 1

    const distances = others.map(p => dist(mode, p, cache, imageData)).sort((a, b) => a - b)
    const kNNDist = distances[k - 1]
    return kNNDist / radius
}

/**
 * Get the k-th nearest neighbour of the mode within the cluster (by position, 1-indexed).
 * Returns null when the cluster has fewer than k+1 points.
 */
export function getKNNPoint(
    mode: MnistPoint,
    points: MnistPoint[],
    k: number,
    cache: DistanceCache,
    imageData: Float32Array
): MnistPoint | null {
    const others = points.filter(p => p.id !== mode.id)
    if (others.length < k) return null
    const sorted = others
        .map(p => ({ point: p, d: dist(mode, p, cache, imageData) }))
        .sort((a, b) => a.d - b.d)
    return sorted[k - 1].point
}

// ---------------------------------------------------------------------------
// Cluster metrics update
// ---------------------------------------------------------------------------

function updateClusterMetrics(
    cluster: MnistCluster,
    k: number,
    cache: DistanceCache,
    imageData: Float32Array
): void {
    const mode = calculateMode(cluster.points, k, cache, imageData)
    cluster.medoidId = mode?.id ?? null
    if (!mode) {
        cluster.radius = 0
        cluster.kNNDist = 0
        cluster.evenness = 1
        return
    }

    cluster.radius = Math.max(0, ...cluster.points.map(p => dist(mode, p, cache, imageData)))

    if (cluster.points.length <= k) {
        cluster.kNNDist = cluster.radius
        cluster.evenness = 1
    } else {
        const others = cluster.points.filter(p => p.id !== mode.id)
        const sorted = others.map(p => dist(mode, p, cache, imageData)).sort((a, b) => a - b)
        cluster.kNNDist = sorted[k - 1]
        cluster.evenness = cluster.radius > 0 ? cluster.kNNDist / cluster.radius : 1
    }
}

// ---------------------------------------------------------------------------
// Top-1 / Last-1 cluster selection helpers
// ---------------------------------------------------------------------------

/**
 * Returns the IDs of clusters in the top half by evenness (ceil(n/2) of n clusters).
 * Returns empty set when fewer than 2 clusters exist.
 */
function getTopHalfClusterIds(clusters: MnistCluster[]): Set<number> {
    if (clusters.length < 2) return new Set()
    const sorted = [...clusters].sort((a, b) => b.evenness - a.evenness)
    const halfCount = Math.ceil(sorted.length / 2)
    return new Set(sorted.slice(0, halfCount).map(c => c.id))
}

/**
 * Find the least-even cluster that can still shrink (maxSize > k+1), excluding a given id.
 * Used by Last-1 Shrink to pick which cluster to shrink when top-1 grows.
 */
function getLeastEvenShrinkable(clusters: MnistCluster[], k: number, excludeId: number): MnistCluster | null {
    let worst: MnistCluster | null = null
    for (const c of clusters) {
        if (c.id === excludeId || c.maxSize <= k + 1) continue
        if (worst === null || c.evenness < worst.evenness) worst = c
    }
    return worst
}

// ---------------------------------------------------------------------------
// Core insertion logic
// ---------------------------------------------------------------------------

function tryPlacePoint(
    state: MnistNKState,
    point: MnistPoint,
    triedClusters: Set<number>,
    cache: DistanceCache,
    imageData: Float32Array,
    deferEviction: boolean = false
): MnistAddPointResult {
    const topHalfIds = state.top1Enabled ? getTopHalfClusterIds(state.clusters) : new Set<number>()

    const clustersWithDist = state.clusters
        .filter(c => !triedClusters.has(c.id))
        .map(c => {
            const mode = getClusterMode(c)
            const d = mode ? dist(point, mode, cache, imageData) : Infinity
            return { cluster: c, d }
        })
        .sort((a, b) => a.d - b.d)

    for (const { cluster, d } of clustersWithDist) {
        // Case 1: cluster is not full – add directly
        if (cluster.points.length < cluster.maxSize) {
            cluster.points.push(point)
            updateClusterMetrics(cluster, state.k, cache, imageData)
            return { point, clusterId: cluster.id, evicted: null, newClusterCreated: false, top1Expanded: false, last1Shrunk: false }
        }

        // Case 2: cluster is full – can the point displace the furthest?
        const mode = getClusterMode(cluster)
        if (mode && d < cluster.radius) {
            // Top-half mechanism: grow instead of evicting if this cluster is in the top half by evenness
            if (topHalfIds.has(cluster.id)) {
                cluster.maxSize++
                cluster.points.push(point)
                updateClusterMetrics(cluster, state.k, cache, imageData)

                // Last-1 shrink: counterbalance by shrinking the least-even shrinkable cluster
                let last1Shrunk = false
                if (state.last1ShrinkEnabled) {
                    const target = getLeastEvenShrinkable(state.clusters, state.k, cluster.id)
                    if (target) {
                        target.maxSize--
                        // If now over capacity, evict furthest and recluster
                        if (target.points.length > target.maxSize) {
                            const targetMode = getClusterMode(target)
                            if (targetMode) {
                                let shrinkEvicted = target.points[0]
                                let shrinkMaxD = 0
                                for (const p of target.points) {
                                    const pd = dist(p, targetMode, cache, imageData)
                                    if (pd > shrinkMaxD) { shrinkMaxD = pd; shrinkEvicted = p }
                                }
                                target.points = target.points.filter(p => p.id !== shrinkEvicted.id)
                                updateClusterMetrics(target, state.k, cache, imageData)
                                tryPlacePoint(state, shrinkEvicted, new Set([target.id]), cache, imageData, false)
                            }
                        } else {
                            updateClusterMetrics(target, state.k, cache, imageData)
                        }
                        last1Shrunk = true
                    }
                }

                return { point, clusterId: cluster.id, evicted: null, newClusterCreated: false, top1Expanded: true, last1Shrunk }
            }

            // Find furthest point to evict
            let evicted = cluster.points[0]
            let maxD = 0
            for (const p of cluster.points) {
                const pd = dist(p, mode, cache, imageData)
                if (pd > maxD) { maxD = pd; evicted = p }
            }

            cluster.points = cluster.points.filter(p => p.id !== evicted.id)
            cluster.points.push(point)
            updateClusterMetrics(cluster, state.k, cache, imageData)

            if (deferEviction) {
                return { point, clusterId: cluster.id, evicted, newClusterCreated: false, top1Expanded: false, last1Shrunk: false }
            }

            triedClusters.add(cluster.id)
            const evictedResult = tryPlacePoint(state, evicted, triedClusters, cache, imageData, false)
            return {
                point,
                clusterId: cluster.id,
                evicted,
                newClusterCreated: evictedResult.newClusterCreated,
                top1Expanded: false,
                last1Shrunk: false,
            }
        }

        triedClusters.add(cluster.id)
    }

    // No cluster accepted the point – create a new one
    const newCluster: MnistCluster = {
        id: state.nextClusterId++,
        points: [point],
        medoidId: point.id,
        radius: 0,
        kNNDist: 0,
        evenness: 1,
        maxSize: state.n,
    }
    state.clusters.push(newCluster)
    return { point, clusterId: newCluster.id, evicted: null, newClusterCreated: true, top1Expanded: false, last1Shrunk: false }
}

/**
 * Add a new MNIST image to the clustering by its dataset index.
 */
export function addMnistPoint(
    state: MnistNKState,
    dataIndex: number,
    cache: DistanceCache,
    imageData: Float32Array,
    deferEviction: boolean = false
): MnistAddPointResult {
    const point: MnistPoint = { id: state.nextPointId++, dataIndex }

    if (state.clusters.length === 0) {
        const cluster: MnistCluster = {
            id: state.nextClusterId++,
            points: [point],
            medoidId: point.id,
            radius: 0,
            kNNDist: 0,
            evenness: 1,
            maxSize: state.n,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, evicted: null, newClusterCreated: true, top1Expanded: false, last1Shrunk: false }
    }

    return tryPlacePoint(state, point, new Set(), cache, imageData, deferEviction)
}

/**
 * Re-insert an existing point (e.g., from replay queue) by its MnistPoint.
 */
export function reinsertMnistPoint(
    state: MnistNKState,
    point: MnistPoint,
    cache: DistanceCache,
    imageData: Float32Array,
    deferEviction: boolean = false
): MnistAddPointResult {
    if (state.clusters.length === 0) {
        const cluster: MnistCluster = {
            id: state.nextClusterId++,
            points: [point],
            medoidId: point.id,
            radius: 0,
            kNNDist: 0,
            evenness: 1,
            maxSize: state.n,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, evicted: null, newClusterCreated: true, top1Expanded: false, last1Shrunk: false }
    }

    return tryPlacePoint(state, point, new Set(), cache, imageData, deferEviction)
}

// ---------------------------------------------------------------------------
// Query helpers for UI
// ---------------------------------------------------------------------------

/**
 * For each cluster, return its points sorted by distance to the mode (closest first),
 * along with the mode index and k-th NN index within that sorted list.
 */
export function getClusterDisplayData(
    cluster: MnistCluster,
    k: number,
    cache: DistanceCache,
    imageData: Float32Array
): {
    pointsSortedByDist: Array<{ point: MnistPoint; distToMode: number }>
    kNNPoint: MnistPoint | null
} {
    const mode = getClusterMode(cluster)
    if (!mode) {
        return { pointsSortedByDist: cluster.points.map(p => ({ point: p, distToMode: 0 })), kNNPoint: null }
    }

    const sorted = cluster.points
        .map(p => ({ point: p, distToMode: dist(p, mode, cache, imageData) }))
        .sort((a, b) => a.distToMode - b.distToMode)

    const kNNPoint = getKNNPoint(mode, cluster.points, k, cache, imageData)
    return { pointsSortedByDist: sorted, kNNPoint }
}
