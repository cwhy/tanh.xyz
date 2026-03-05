/**
 * Rival-K Clustering Algorithm
 *
 * An online clustering algorithm parameterised only by K.
 * Clusters have no hard size cap; instead, the cluster with the lowest
 * evenness is continuously trimmed so the system self-regulates.
 *
 * Terminology:
 *   Kid   – cluster with size < K+2  (evenness always 1)
 *   K2    – cluster with size == K+2
 *   Adult – cluster with size > K+2
 *
 * Insertion priority:
 *   1. Kid clusters where Dm < Dmf → pick smallest Dm.
 *   2. All kid clusters + the single adult with highest evenness
 *      where Dm < Dmf → pick smallest Dm.
 *   3. Create a new cluster.
 *
 * After every insertion the cluster with the LOWEST evenness
 * (excluding the one that just received a point) evicts its
 * furthest point.
 */

// ── Types ──────────────────────────────────────────────────────────

export interface Point {
    x: number
    y: number
    id: number
}

export interface RivalKCluster {
    id: number
    points: Point[]
    medoidId: number | null
    radius: number       // distance from medoid to furthest point
    kNNDist: number      // distance from medoid to K-th nearest neighbor
    evenness: number     // kNNDist / radius
}

export type ClusterMaturity = 'kid' | 'k2' | 'adult'

export interface RivalKState {
    clusters: RivalKCluster[]
    k: number
    nextPointId: number
    nextClusterId: number
}

export interface RivalKAddPointResult {
    point: Point
    clusterId: number
    evicted: Point | null
    newClusterCreated: boolean
}

// ── Helpers ────────────────────────────────────────────────────────

export function distance(a: Point, b: Point): number {
    const dx = a.x - b.x
    const dy = a.y - b.y
    return Math.sqrt(dx * dx + dy * dy)
}

function kthNearestNeighborDistance(point: Point, points: Point[], k: number): number {
    const distances = points
        .filter(p => p.id !== point.id)
        .map(p => distance(point, p))
        .sort((a, b) => a - b)
    if (distances.length < k) return Infinity
    return distances[k - 1]
}

export function getMaturity(cluster: RivalKCluster, k: number): ClusterMaturity {
    const threshold = k + 2
    if (cluster.points.length < threshold) return 'kid'
    if (cluster.points.length === threshold) return 'k2'
    return 'adult'
}

// ── Medoid ─────────────────────────────────────────────────────────

export function calculateMedoid(points: Point[], k: number): Point | null {
    if (points.length === 0) return null
    if (points.length === 1) return points[0]

    // Fewer than K+1 points → use point minimising total distance
    if (points.length <= k) {
        let best = points[0]
        let bestSum = Infinity
        for (const c of points) {
            const sum = points
                .filter(p => p.id !== c.id)
                .reduce((acc, p) => acc + distance(c, p), 0)
            if (sum < bestSum) { bestSum = sum; best = c }
        }
        return best
    }

    // Point with smallest K-th NN distance
    let best = points[0]
    let bestDist = Infinity
    for (const c of points) {
        const d = kthNearestNeighborDistance(c, points, k)
        if (d < bestDist) { bestDist = d; best = c }
    }
    return best
}

export function getClusterMedoid(cluster: RivalKCluster): Point | null {
    if (cluster.medoidId === null) return null
    return cluster.points.find(p => p.id === cluster.medoidId) ?? null
}

// ── Metric updates ────────────────────────────────────────────────

function updateClusterMetrics(cluster: RivalKCluster, k: number): void {
    const medoid = calculateMedoid(cluster.points, k)
    cluster.medoidId = medoid?.id ?? null
    if (!medoid) {
        cluster.radius = 0
        cluster.kNNDist = 0
        cluster.evenness = 1
        return
    }

    cluster.radius = cluster.points.length === 0
        ? 0
        : Math.max(...cluster.points.map(p => distance(medoid, p)))

    if (cluster.points.length <= k || cluster.radius === 0) {
        cluster.kNNDist = cluster.radius
        cluster.evenness = 1
        return
    }

    const kDist = kthNearestNeighborDistance(medoid, cluster.points, k)
    cluster.kNNDist = Number.isFinite(kDist) ? kDist : cluster.radius
    cluster.evenness = cluster.kNNDist / cluster.radius
}

// ── Eviction helpers ──────────────────────────────────────────────

function getFurthestPointFromMedoid(cluster: RivalKCluster, medoid: Point): Point {
    let worst = cluster.points[0]
    let maxD = -Infinity
    for (const p of cluster.points) {
        const d = distance(p, medoid)
        if (d > maxD) { maxD = d; worst = p }
    }
    return worst
}

/**
 * Find the cluster with the lowest evenness, excluding a given id.
 * Kids always have evenness 1, so they naturally lose this competition
 * only when every other cluster is also at evenness 1.
 */
function getLeastEven(clusters: RivalKCluster[], excludeId: number): RivalKCluster | null {
    let worst: RivalKCluster | null = null
    for (const c of clusters) {
        if (c.id === excludeId) continue
        if (c.points.length <= 1) continue  // can't evict from a singleton
        if (worst === null || c.evenness < worst.evenness) worst = c
    }
    return worst
}

// ── State management ──────────────────────────────────────────────

export function createRivalKState(k: number): RivalKState {
    return {
        clusters: [],
        k,
        nextPointId: 0,
        nextClusterId: 0,
    }
}

export function resetRivalKState(state: RivalKState): void {
    state.clusters = []
    state.nextPointId = 0
    state.nextClusterId = 0
}

// ── Core insertion ────────────────────────────────────────────────

/**
 * Add a brand-new point (assigns a fresh id).
 */
export function addPoint(
    state: RivalKState,
    x: number,
    y: number,
    deferEviction: boolean = false,
): RivalKAddPointResult {
    const point: Point = { x, y, id: state.nextPointId++ }
    return insertPoint(state, point, deferEviction)
}

/**
 * Re-insert an existing point (keeps its id).
 */
export function addExistingPoint(
    state: RivalKState,
    point: Point,
    deferEviction: boolean = false,
): RivalKAddPointResult {
    return insertPoint(state, point, deferEviction)
}

/**
 * Shared insertion logic: place point → evict from least-even cluster
 * ONLY if the recipient cluster is an adult (i.e. an adult just grew).
 * The cluster that just received the point is excluded from eviction.
 */
function insertPoint(
    state: RivalKState,
    point: Point,
    deferEviction: boolean,
): RivalKAddPointResult {
    const placeResult = placePoint(state, point)

    let evictResult: Point | null = null
    const recipient = state.clusters.find(c => c.id === placeResult.clusterId)!

    // Only evict if the cluster that just grew is an adult
    if (getMaturity(recipient, state.k) === 'adult') {
        evictResult = evictFromWorst(state, deferEviction, placeResult.clusterId)
    }

    return {
        point: placeResult.point,
        clusterId: placeResult.clusterId,
        evicted: evictResult,
        newClusterCreated: placeResult.newClusterCreated,
    }
}

/**
 * Place a point into the best cluster according to the priority rules:
 *
 * Let Dm = distance to medoid
 * Let DmK = distance to K-th nearest neighbor of the medoid (kNNDist)
 * Let DmF = distance to furthest point of the cluster (radius)
 *
 * 1. Kid cluster where Dm < DmK → pick smallest Dm.
 * 2. Any cluster where Dm < DmK → pick smallest Dm.
 * 3. Kid cluster where Dm < DmF → pick smallest Dm.
 * 4. Create a new cluster.
 */
function placePoint(
    state: RivalKState,
    point: Point,
): { point: Point; clusterId: number; newClusterCreated: boolean } {
    const { clusters, k } = state

    if (clusters.length === 0) {
        return createNewCluster(state, point)
    }

    type Candidate = { cluster: RivalKCluster; dm: number }
    const getCandidates = (filter: (c: RivalKCluster, dm: number) => boolean): Candidate[] => {
        const candidates: Candidate[] = []
        for (const c of clusters) {
            const medoid = getClusterMedoid(c)
            if (!medoid) continue
            const dm = distance(point, medoid)
            if (filter(c, dm)) {
                candidates.push({ cluster: c, dm })
            }
        }
        return candidates.sort((a, b) => a.dm - b.dm)
    }

    // ── Step 1: Kid cluster where Dm < DmK ──
    const step1 = getCandidates((c, dm) => getMaturity(c, k) === 'kid' && dm < c.kNNDist)
    if (step1.length > 0) {
        const chosen = step1[0].cluster
        chosen.points.push(point)
        updateClusterMetrics(chosen, k)
        return { point, clusterId: chosen.id, newClusterCreated: false }
    }

    // ── Step 2: All clusters compete for lowest Dm ──
    const step2 = getCandidates(() => true) // All clusters
    if (step2.length > 0) {
        const winner = step2[0]
        const mat = getMaturity(winner.cluster, k)

        const accept = mat === 'kid' ||
            (mat === 'k2' && winner.dm < winner.cluster.radius) ||
            (mat === 'adult' && winner.dm < winner.cluster.kNNDist)

        if (accept) {
            const chosen = winner.cluster
            chosen.points.push(point)
            updateClusterMetrics(chosen, k)
            return { point, clusterId: chosen.id, newClusterCreated: false }
        }
    }

    // ── Step 3: Create new cluster ──
    return createNewCluster(state, point)
}

function createNewCluster(
    state: RivalKState,
    point: Point,
): { point: Point; clusterId: number; newClusterCreated: boolean } {
    const cluster: RivalKCluster = {
        id: state.nextClusterId++,
        points: [point],
        medoidId: point.id,
        radius: 0,
        kNNDist: 0,
        evenness: 1,
    }
    state.clusters.push(cluster)
    return { point, clusterId: cluster.id, newClusterCreated: true }
}

/**
 * Evict the furthest point from the least-even cluster.
 * Returns the evicted point (or null if no qualifying cluster exists).
 *
 * @param excludeClusterId – the cluster that just received a point;
 *        excluded so it isn't immediately trimmed.
 */
function evictFromWorst(
    state: RivalKState,
    deferEviction: boolean,
    excludeClusterId: number,
): Point | null {
    // Only evict when there are at least 2 adult clusters
    let adultCount = 0
    for (const c of state.clusters) {
        if (getMaturity(c, state.k) === 'adult') adultCount++
    }
    if (adultCount < 2) return null

    const worst = getLeastEven(state.clusters, excludeClusterId)
    if (!worst) return null

    const medoid = getClusterMedoid(worst)
    if (!medoid) return null

    const evicted = getFurthestPointFromMedoid(worst, medoid)
    worst.points = worst.points.filter(p => p.id !== evicted.id)
    updateClusterMetrics(worst, state.k)

    if (!deferEviction) {
        // Re-place without triggering another eviction (only one eviction per original insertion)
        placePoint(state, evicted)
        return evicted
    }

    return evicted
}

// ── Query helpers ─────────────────────────────────────────────────

export function getAllPointsWithClusters(state: RivalKState): Array<{
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
