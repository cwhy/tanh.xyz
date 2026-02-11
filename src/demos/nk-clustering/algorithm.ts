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
    cMin: number  // Distance from medoid to furthest point
}

export type CenterType = 'medoid' | 'centroid'

export interface NKClusteringState {
    clusters: Cluster[]
    n: number  // Max cluster size
    k: number  // K-th neighbor for medoid calculation
    centerType: CenterType  // How to calculate cluster center
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
 * Calculate cMin: distance from medoid to the furthest point
 */
export function calculateCMin(medoid: Point, points: Point[]): number {
    if (points.length === 0) return 0
    return Math.max(...points.map(p => distance(medoid, p)))
}

/**
 * Update a cluster's medoid and cMin
 */
function updateClusterMetrics(cluster: Cluster, k: number, centerType: CenterType): void {
    const medoid = calculateCenter(cluster.points, k, centerType)
    cluster.medoidId = medoid?.id ?? null
    if (medoid) {
        cluster.cMin = calculateCMin(medoid, cluster.points)
    }
}

/**
 * Get the medoid point of a cluster
 */
export function getClusterMedoid(cluster: Cluster): Point | null {
    if (cluster.medoidId === null) return null
    return cluster.points.find(p => p.id === cluster.medoidId) ?? null
}

/**
 * Create initial state for NK clustering
 */
export function createNKClusteringState(n: number, k: number, centerType: CenterType = 'medoid'): NKClusteringState {
    if (k >= n) {
        throw new Error(`K (${k}) must be less than N (${n})`)
    }
    return {
        clusters: [],
        n,
        k,
        centerType,
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
    // Get clusters sorted by distance to their medoids
    const clustersWithDist = state.clusters
        .filter(c => !triedClusters.has(c.id))
        .map(c => {
            const medoid = getClusterMedoid(c)
            const dist = medoid ? distance(point, medoid) : Infinity
            return { cluster: c, dist }
        })
        .sort((a, b) => a.dist - b.dist)

    // Try each cluster in order of proximity
    for (const { cluster, dist } of clustersWithDist) {
        // Case 1: Cluster is not full - just add
        if (cluster.points.length < state.n) {
            cluster.points.push(point)
            updateClusterMetrics(cluster, state.k, state.centerType)
            return { point, clusterId: cluster.id, evicted: null, newClusterCreated: false }
        }

        // Case 2: Cluster is full - check if point can displace someone
        const medoid = getClusterMedoid(cluster)
        if (medoid && dist < cluster.cMin) {
            // Point is closer than the furthest point - it can join

            // Find the point to evict (furthest from medoid)
            let evictedPoint = cluster.points[0]
            let maxDist = 0
            for (const p of cluster.points) {
                const d = distance(p, medoid)
                if (d > maxDist) {
                    maxDist = d
                    evictedPoint = p
                }
            }

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
