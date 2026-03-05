/**
 * NK Clustering Algorithm
 *
 * An online clustering algorithm with bounded cluster sizes.
 * - N: Maximum number of points per cluster
 * - K: Number of neighbors used to determine the medoid (K < N)
 *
 * Medoid Definition: The point with the lowest distance to its K-th nearest neighbor.
 * This finds the point in the densest region of the cluster, making it robust to outliers.
 */

export interface Point {
    x: number
    y: number
    id: number
}

export interface Cluster {
    id: number
    points: Point[]
    medoidId: number | null
    cMin: number  // Distance from center to furthest point
    kNNDist: number
    evenness: number
    maxSize: number
}

export type CenterType = 'medoid' | 'centroid'

export interface NKClusteringState {
    clusters: Cluster[]
    n: number  // Max cluster size
    k: number  // K-th neighbor for medoid calculation
    centerType: CenterType  // How to calculate cluster center
    top1Enabled: boolean
    last1ShrinkEnabled: boolean
    nextPointId: number
    nextClusterId: number
}

/**
 * Calculate Euclidean distance between two points
 */
export function distance(a: Point, b: Point): number {
    const dx = a.x - b.x
    const dy = a.y - b.y
    return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Find the K-th nearest neighbor distance for a point within a set of points
 */
function kthNearestNeighborDistance(point: Point, points: Point[], k: number): number {
    const distances = points
        .filter(p => p.id !== point.id)
        .map(p => distance(point, p))
        .sort((a, b) => a - b)

    // If we don't have K neighbors, return Infinity
    if (distances.length < k) {
        return Infinity
    }
    return distances[k - 1]  // 0-indexed, so k-1 gives K-th neighbor
}

/**
 * Calculate the medoid of a cluster
 * Medoid = point with lowest K-th nearest neighbor distance
 */
export function calculateMedoid(points: Point[], k: number): Point | null {
    if (points.length === 0) return null
    if (points.length === 1) return points[0]

    // If we have fewer than K+1 points, use the centroid-like approach
    // (point minimizing sum of distances to all others)
    if (points.length <= k) {
        let bestPoint = points[0]
        let bestSum = Infinity

        for (const candidate of points) {
            const sum = points
                .filter(p => p.id !== candidate.id)
                .reduce((acc, p) => acc + distance(candidate, p), 0)
            if (sum < bestSum) {
                bestSum = sum
                bestPoint = candidate
            }
        }
        return bestPoint
    }

    // Find point with lowest K-th NN distance
    let bestPoint = points[0]
    let bestDist = Infinity

    for (const candidate of points) {
        const kthDist = kthNearestNeighborDistance(candidate, points, k)
        if (kthDist < bestDist) {
            bestDist = kthDist
            bestPoint = candidate
        }
    }

    return bestPoint
}

/**
 * Calculate the centroid of a cluster
 * Returns the point closest to the geometric center
 */
export function calculateCentroid(points: Point[]): Point | null {
    if (points.length === 0) return null
    if (points.length === 1) return points[0]

    // Calculate geometric center
    const centerX = points.reduce((sum, p) => sum + p.x, 0) / points.length
    const centerY = points.reduce((sum, p) => sum + p.y, 0) / points.length
    const center: Point = { x: centerX, y: centerY, id: -1 }

    // Find the point closest to the geometric center
    let bestPoint = points[0]
    let bestDist = Infinity

    for (const candidate of points) {
        const dist = distance(candidate, center)
        if (dist < bestDist) {
            bestDist = dist
            bestPoint = candidate
        }
    }

    return bestPoint
}

/**
 * Calculate the cluster center based on the center type
 */
export function calculateCenter(points: Point[], k: number, centerType: CenterType): Point | null {
    if (centerType === 'centroid') {
        return calculateCentroid(points)
    }
    return calculateMedoid(points, k)
}

/**
 * Calculate cMin: distance from center to the furthest point
 */
export function calculateCMin(center: Point, points: Point[]): number {
    if (points.length === 0) return 0
    return Math.max(...points.map(p => distance(center, p)))
}

/**
 * Calculate evenness = d(center, k-th NN) / d(center, furthest).
 * Returns 1 for very small clusters or zero-radius clusters.
 */
export function calculateEvenness(center: Point, points: Point[], k: number, cMin: number): number {
    if (points.length <= 1 || cMin === 0) return 1
    if (points.length <= k) return 1

    const kDist = kthNearestNeighborDistance(center, points, k)
    if (!Number.isFinite(kDist)) return 1
    return kDist / cMin
}

/**
 * Update a cluster's medoid and cMin
 */
function updateClusterMetrics(cluster: Cluster, k: number, centerType: CenterType): void {
    const center = calculateCenter(cluster.points, k, centerType)
    cluster.medoidId = center?.id ?? null
    if (!center) {
        cluster.cMin = 0
        cluster.kNNDist = 0
        cluster.evenness = 1
        return
    }

    cluster.cMin = calculateCMin(center, cluster.points)
    if (cluster.points.length <= k || cluster.cMin === 0) {
        cluster.kNNDist = cluster.cMin
        cluster.evenness = 1
        return
    }

    const kDist = kthNearestNeighborDistance(center, cluster.points, k)
    cluster.kNNDist = Number.isFinite(kDist) ? kDist : cluster.cMin
    cluster.evenness = calculateEvenness(center, cluster.points, k, cluster.cMin)
}

/**
 * Get the medoid point of a cluster
 */
export function getClusterMedoid(cluster: Cluster): Point | null {
    if (cluster.medoidId === null) return null
    return cluster.points.find(p => p.id === cluster.medoidId) ?? null
}

/**
 * Returns the IDs of clusters in the top half by evenness.
 * Returns empty set when fewer than 2 clusters exist.
 */
function getTopHalfClusterIds(clusters: Cluster[]): Set<number> {
    if (clusters.length < 2) return new Set()
    const sorted = [...clusters].sort((a, b) => b.evenness - a.evenness)
    const halfCount = Math.ceil(sorted.length / 2)
    return new Set(sorted.slice(0, halfCount).map(c => c.id))
}

/**
 * Find the least-even cluster that can still shrink (maxSize > k+2), excluding a given id.
 */
function getLeastEvenShrinkable(clusters: Cluster[], k: number, excludeId: number): Cluster | null {
    let worst: Cluster | null = null
    for (const c of clusters) {
        if (c.id === excludeId || c.maxSize <= k + 2) continue
        if (worst === null || c.evenness < worst.evenness) worst = c
    }
    return worst
}

function getFurthestPointFromCenter(cluster: Cluster, center: Point): Point {
    let evictedPoint = cluster.points[0]
    let maxDist = -Infinity
    for (const p of cluster.points) {
        const d = distance(p, center)
        if (d > maxDist) {
            maxDist = d
            evictedPoint = p
        }
    }
    return evictedPoint
}

/**
 * Create initial state for NK clustering
 */
export function createNKClusteringState(
    n: number,
    k: number,
    centerType: CenterType = 'medoid',
    top1Enabled: boolean = false,
    last1ShrinkEnabled: boolean = false
): NKClusteringState {
    if (k >= n) {
        throw new Error(`K (${k}) must be less than N (${n})`)
    }
    return {
        clusters: [],
        n,
        k,
        centerType,
        top1Enabled,
        last1ShrinkEnabled,
        nextPointId: 0,
        nextClusterId: 0,
    }
}

/**
 * Result of adding a point
 */
export interface AddPointResult {
    point: Point
    clusterId: number
    evicted: Point | null  // Point that was kicked out (if any)
    newClusterCreated: boolean
}

/**
 * Add a new point to the clustering
 * Returns information about where the point ended up
 * @param deferEviction If true, evicted points are not immediately re-placed (returned for caller to handle)
 */
export function addPoint(
    state: NKClusteringState,
    x: number,
    y: number,
    deferEviction: boolean = false
): AddPointResult {
    const point: Point = { x, y, id: state.nextPointId++ }

    // If no clusters exist, create the first one
    if (state.clusters.length === 0) {
        const cluster: Cluster = {
            id: state.nextClusterId++,
            points: [point],
            medoidId: point.id,
            cMin: 0,
            kNNDist: 0,
            evenness: 1,
            maxSize: state.n,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, evicted: null, newClusterCreated: true }
    }

    // Try to place the point, with possible cascading evictions
    return tryPlacePoint(state, point, new Set(), deferEviction)
}

/**
 * Add an existing point (e.g., from replay queue) to the clustering
 */
export function addExistingPoint(
    state: NKClusteringState,
    point: Point,
    deferEviction: boolean = false
): AddPointResult {
    // If no clusters exist, create the first one
    if (state.clusters.length === 0) {
        const cluster: Cluster = {
            id: state.nextClusterId++,
            points: [point],
            medoidId: point.id,
            cMin: 0,
            kNNDist: 0,
            evenness: 1,
            maxSize: state.n,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, evicted: null, newClusterCreated: true }
    }

    // Try to place the point, with possible cascading evictions
    return tryPlacePoint(state, point, new Set(), deferEviction)
}

/**
 * Try to place a point in the best cluster, handling evictions
 */
function tryPlacePoint(
    state: NKClusteringState,
    point: Point,
    triedClusters: Set<number>,
    deferEviction: boolean = false
): AddPointResult {
    const topHalfIds = state.top1Enabled ? getTopHalfClusterIds(state.clusters) : new Set<number>()

    // Get clusters sorted by distance to their centers
    const clustersWithDist = state.clusters
        .filter(c => !triedClusters.has(c.id))
        .map(c => {
            const center = getClusterMedoid(c)
            const dist = center ? distance(point, center) : Infinity
            return { cluster: c, dist }
        })
        .sort((a, b) => a.dist - b.dist)

    // Try each cluster in order of proximity
    for (const { cluster, dist } of clustersWithDist) {
        // Case 1: Cluster is not full - just add
        if (cluster.points.length < cluster.maxSize) {
            cluster.points.push(point)
            updateClusterMetrics(cluster, state.k, state.centerType)
            return { point, clusterId: cluster.id, evicted: null, newClusterCreated: false }
        }

        // Case 2: Cluster is full - check if point can displace someone
        const center = getClusterMedoid(cluster)
        if (center && dist < cluster.cMin) {
            // Top-half mechanism: top 50% even clusters grow instead of evicting.
            if (topHalfIds.has(cluster.id)) {
                cluster.maxSize++
                cluster.points.push(point)
                updateClusterMetrics(cluster, state.k, state.centerType)

                // Last-1 shrink mechanism: shrink least-even shrinkable cluster.
                if (state.last1ShrinkEnabled) {
                    const target = getLeastEvenShrinkable(state.clusters, state.k, cluster.id)
                    if (target) {
                        target.maxSize--
                        if (target.points.length > target.maxSize) {
                            const targetCenter = getClusterMedoid(target)
                            if (targetCenter) {
                                const targetEvicted = getFurthestPointFromCenter(target, targetCenter)
                                target.points = target.points.filter(p => p.id !== targetEvicted.id)
                                updateClusterMetrics(target, state.k, state.centerType)
                                tryPlacePoint(state, targetEvicted, new Set([target.id]), false)
                            }
                        } else {
                            updateClusterMetrics(target, state.k, state.centerType)
                        }
                    }
                }

                return { point, clusterId: cluster.id, evicted: null, newClusterCreated: false }
            }

            // Point is closer than the furthest point - it can join

            // Find the point to evict (furthest from center)
            const evictedPoint = getFurthestPointFromCenter(cluster, center)

            // Swap: add new point, remove evicted
            cluster.points = cluster.points.filter(p => p.id !== evictedPoint.id)
            cluster.points.push(point)
            updateClusterMetrics(cluster, state.k, state.centerType)

            // If deferring eviction, return evicted point for caller to handle later
            if (deferEviction) {
                return {
                    point,
                    clusterId: cluster.id,
                    evicted: evictedPoint,
                    newClusterCreated: false,
                }
            }

            // Try to place the evicted point elsewhere
            triedClusters.add(cluster.id)
            const evictedResult = tryPlacePoint(state, evictedPoint, triedClusters, false)

            return {
                point,
                clusterId: cluster.id,
                evicted: evictedPoint,
                newClusterCreated: evictedResult.newClusterCreated,
            }
        }

        // Point couldn't fit in this cluster, try next
        triedClusters.add(cluster.id)
    }

    // No existing cluster could accommodate the point - create new cluster
    const newCluster: Cluster = {
        id: state.nextClusterId++,
        points: [point],
        medoidId: point.id,
        cMin: 0,
        kNNDist: 0,
        evenness: 1,
        maxSize: state.n,
    }
    state.clusters.push(newCluster)
    return { point, clusterId: newCluster.id, evicted: null, newClusterCreated: true }
}

/**
 * Get all points with their cluster assignments
 */
export function getAllPointsWithClusters(state: NKClusteringState): Array<{
    point: Point
    clusterId: number
    isMedoid: boolean
}> {
    const result: Array<{ point: Point; clusterId: number; isMedoid: boolean }> = []

    for (const cluster of state.clusters) {
        for (const point of cluster.points) {
            result.push({
                point,
                clusterId: cluster.id,
                isMedoid: point.id === cluster.medoidId,
            })
        }
    }

    return result
}

/**
 * Reset the clustering state (keep N and K)
 */
export function resetState(state: NKClusteringState): void {
    state.clusters = []
    state.nextPointId = 0
    state.nextClusterId = 0
}
