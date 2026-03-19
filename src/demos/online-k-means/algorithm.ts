/**
 * Online Clustering Algorithms × Distance Metrics
 *
 * Algorithms:
 *   1. Online K-Means — running mean, hard assignment
 *   2. Hard WTA — learning-rate update, hard assignment (winner only)
 *   3. SoftHebb — learning-rate update, soft assignment (all neurons)
 *
 * Distance Metrics:
 *   - Euclidean: centroids in data space, ||x − w||
 *   - Angular:   centroids on unit sphere, cosine similarity w·x̂
 *
 * The metric controls input preprocessing, similarity computation,
 * and the update rule variant. The algorithm controls competition
 * (hard vs soft) and which neurons update.
 */

// ── Types ──────────────────────────────────────────────────────────

export type AlgorithmType = 'online-k-means' | 'hard-wta' | 'softhebb'
export type DistanceMetric = 'euclidean' | 'angular'

export interface Point {
    x: number
    y: number
    id: number
}

export interface OnlineKMeansCluster {
    id: number
    points: Point[]
    centroidX: number
    centroidY: number
    /** Sum of input coords (raw for Euclidean, normalized for Angular) */
    sumX: number
    /** Sum of input coords (raw for Euclidean, normalized for Angular) */
    sumY: number
    /** Learnable bias for SoftHebb (log-prior for cluster frequency) */
    bias: number
}

export interface OnlineKMeansState {
    clusters: OnlineKMeansCluster[]
    k: number
    nextPointId: number
    nextClusterId: number
    algorithm: AlgorithmType
    distanceMetric: DistanceMetric
    /** Learning rate for WTA / SoftHebb weight updates */
    learningRate: number
    /** Softmax base for SoftHebb (b → ∞ = hard, b = e = standard softmax, b → 1 = uniform) */
    softmaxBase: number
}

export interface OnlineKMeansAddPointResult {
    point: Point
    clusterId: number
    newClusterCreated: boolean
    responsibilities?: number[]
}

// ── Helpers ────────────────────────────────────────────────────────

export function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    const dx = a.x - b.x
    const dy = a.y - b.y
    return Math.sqrt(dx * dx + dy * dy)
}

function distSq(ax: number, ay: number, bx: number, by: number): number {
    const dx = ax - bx
    const dy = ay - by
    return dx * dx + dy * dy
}

function normalize2D(x: number, y: number): [number, number] {
    const len = Math.sqrt(x * x + y * y)
    if (len < 1e-12) return [1, 0]
    return [x / len, y / len]
}

function dot2D(ax: number, ay: number, bx: number, by: number): number {
    return ax * bx + ay * by
}

// ── State management ──────────────────────────────────────────────

export function createOnlineKMeansState(
    k: number,
    algorithm: AlgorithmType = 'online-k-means',
    distanceMetric: DistanceMetric = 'euclidean',
    learningRate: number = 0.01,
    softmaxBase: number = Math.E,
): OnlineKMeansState {
    return {
        clusters: [],
        k: Math.max(1, k),
        nextPointId: 0,
        nextClusterId: 0,
        algorithm,
        distanceMetric,
        learningRate,
        softmaxBase,
    }
}

export function resetOnlineKMeansState(state: OnlineKMeansState): void {
    state.clusters = []
    state.nextPointId = 0
    state.nextClusterId = 0
}

// ── Core dispatch ────────────────────────────────────────────────

export function addPoint(
    state: OnlineKMeansState,
    x: number,
    y: number,
): OnlineKMeansAddPointResult {
    switch (state.algorithm) {
        case 'hard-wta':
            return addPointHardWTA(state, x, y)
        case 'softhebb':
            return addPointSoftHebb(state, x, y)
        case 'online-k-means':
        default:
            return addPointKMeans(state, x, y)
    }
}

// ── Online K-Means ────────────────────────────────────────────────
//
// Euclidean: assign by ||x−w||, update w = running mean of raw inputs
// Angular:   assign by cos(x,w), update w = normalize(running sum of x̂)

function addPointKMeans(
    state: OnlineKMeansState,
    x: number,
    y: number,
): OnlineKMeansAddPointResult {
    const angular = state.distanceMetric === 'angular'
    // Input for assignment & accumulation
    const [ix, iy] = angular ? normalize2D(x, y) : [x, y]
    const point: Point = { x, y, id: state.nextPointId++ }

    // Seeding
    if (state.clusters.length < state.k) {
        const [cx, cy] = angular ? normalize2D(ix, iy) : [ix, iy]
        const cluster: OnlineKMeansCluster = {
            id: state.nextClusterId++,
            points: [point],
            centroidX: cx,
            centroidY: cy,
            sumX: ix,
            sumY: iy,
            bias: 0,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, newClusterCreated: true }
    }

    // Assignment
    let bestIdx = 0
    if (angular) {
        let bestSim = dot2D(state.clusters[0].centroidX, state.clusters[0].centroidY, ix, iy)
        for (let i = 1; i < state.clusters.length; i++) {
            const s = dot2D(state.clusters[i].centroidX, state.clusters[i].centroidY, ix, iy)
            if (s > bestSim) { bestSim = s; bestIdx = i }
        }
    } else {
        let bestDist = distSq(state.clusters[0].centroidX, state.clusters[0].centroidY, x, y)
        for (let i = 1; i < state.clusters.length; i++) {
            const d = distSq(state.clusters[i].centroidX, state.clusters[i].centroidY, x, y)
            if (d < bestDist) { bestDist = d; bestIdx = i }
        }
    }

    const winner = state.clusters[bestIdx]
    winner.points.push(point)
    winner.sumX += ix
    winner.sumY += iy

    if (angular) {
        // Centroid = normalize(sum) — direction of the mean
        ;[winner.centroidX, winner.centroidY] = normalize2D(winner.sumX, winner.sumY)
    } else {
        // Centroid = arithmetic mean
        const n = winner.points.length
        winner.centroidX = winner.sumX / n
        winner.centroidY = winner.sumY / n
    }

    return { point, clusterId: winner.id, newClusterCreated: false }
}

// ── Hard WTA ──────────────────────────────────────────────────────
//
// Euclidean: assign by ||x−w||, Δw = η(x − w), winner only
// Angular:   assign by w·x̂,    Δw = η(x̂ − u·w), winner only (self-normalizing)

function addPointHardWTA(
    state: OnlineKMeansState,
    x: number,
    y: number,
): OnlineKMeansAddPointResult {
    const angular = state.distanceMetric === 'angular'
    const [ix, iy] = angular ? normalize2D(x, y) : [x, y]
    const point: Point = { x, y, id: state.nextPointId++ }

    // Seeding
    if (state.clusters.length < state.k) {
        const cluster: OnlineKMeansCluster = {
            id: state.nextClusterId++,
            points: [point],
            centroidX: ix,
            centroidY: iy,
            sumX: 0,
            sumY: 0,
            bias: 0,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, newClusterCreated: true }
    }

    // Assignment
    let bestIdx = 0
    if (angular) {
        let bestU = dot2D(state.clusters[0].centroidX, state.clusters[0].centroidY, ix, iy)
        for (let i = 1; i < state.clusters.length; i++) {
            const u = dot2D(state.clusters[i].centroidX, state.clusters[i].centroidY, ix, iy)
            if (u > bestU) { bestU = u; bestIdx = i }
        }
    } else {
        let bestDist = distSq(state.clusters[0].centroidX, state.clusters[0].centroidY, x, y)
        for (let i = 1; i < state.clusters.length; i++) {
            const d = distSq(state.clusters[i].centroidX, state.clusters[i].centroidY, x, y)
            if (d < bestDist) { bestDist = d; bestIdx = i }
        }
    }

    const winner = state.clusters[bestIdx]
    const η = state.learningRate

    if (angular) {
        // Self-normalizing Hebbian: Δw = η(x̂ − u·w)
        const u = dot2D(winner.centroidX, winner.centroidY, ix, iy)
        winner.centroidX += η * (ix - u * winner.centroidX)
        winner.centroidY += η * (iy - u * winner.centroidY)
    } else {
        // Simple gradient: Δw = η(x − w)
        winner.centroidX += η * (x - winner.centroidX)
        winner.centroidY += η * (y - winner.centroidY)
    }

    winner.points.push(point)
    return { point, clusterId: winner.id, newClusterCreated: false }
}

// ── SoftHebb ──────────────────────────────────────────────────────
//
// Euclidean: u_k = −||x − w_k||², Δw_k = η·y_k·(x − w_k)
// Angular:   u_k = w_k·x̂,        Δw_k = η·y_k·(x̂ − u_k·w_k)
// Both:      y_k = softmax_b(u_k + bias_k),
//            Δbias_k = η·e^(−bias_k)·(y_k − e^(bias_k))

function addPointSoftHebb(
    state: OnlineKMeansState,
    x: number,
    y: number,
): OnlineKMeansAddPointResult {
    const angular = state.distanceMetric === 'angular'
    const [ix, iy] = angular ? normalize2D(x, y) : [x, y]
    const point: Point = { x, y, id: state.nextPointId++ }

    // Seeding
    if (state.clusters.length < state.k) {
        const cluster: OnlineKMeansCluster = {
            id: state.nextClusterId++,
            points: [point],
            centroidX: ix,
            centroidY: iy,
            sumX: 0,
            sumY: 0,
            bias: 0,
        }
        state.clusters.push(cluster)
        return { point, clusterId: cluster.id, newClusterCreated: true }
    }

    const η = state.learningRate
    const logB = Math.log(state.softmaxBase)

    // Preactivation
    const preactivations: number[] = []
    for (const c of state.clusters) {
        if (angular) {
            // Dot product (≈ cosine sim since ||w|| ≈ 1)
            preactivations.push(dot2D(c.centroidX, c.centroidY, ix, iy))
        } else {
            // Negative squared Euclidean distance
            preactivations.push(-distSq(c.centroidX, c.centroidY, x, y))
        }
    }

    // Softmax with base b: y_k = b^(u_k + bias_k) / Σ b^(u_l + bias_l)
    const logits = preactivations.map((u, i) => logB * (u + state.clusters[i].bias))
    const maxLogit = Math.max(...logits)
    const expLogits = logits.map(l => Math.exp(l - maxLogit))
    const sumExp = expLogits.reduce((a, v) => a + v, 0)
    const responsibilities = expLogits.map(e => e / sumExp)

    // Find winner
    let bestIdx = 0
    let bestResp = responsibilities[0]
    for (let i = 1; i < responsibilities.length; i++) {
        if (responsibilities[i] > bestResp) { bestResp = responsibilities[i]; bestIdx = i }
    }

    // Update ALL neurons
    for (let i = 0; i < state.clusters.length; i++) {
        const c = state.clusters[i]
        const y_k = responsibilities[i]

        if (angular) {
            const u_k = preactivations[i]
            // Self-normalizing Hebbian: Δw = η·y_k·(x̂ − u_k·w)
            c.centroidX += η * y_k * (ix - u_k * c.centroidX)
            c.centroidY += η * y_k * (iy - u_k * c.centroidY)
        } else {
            // Gradient: Δw = η·y_k·(x − w)
            c.centroidX += η * y_k * (x - c.centroidX)
            c.centroidY += η * y_k * (y - c.centroidY)
        }

        // Bias update (same for both metrics)
        c.bias += η * Math.exp(-c.bias) * (y_k - Math.exp(c.bias))
    }

    state.clusters[bestIdx].points.push(point)

    return {
        point,
        clusterId: state.clusters[bestIdx].id,
        newClusterCreated: false,
        responsibilities,
    }
}

// ── Query helpers ─────────────────────────────────────────────────

export function getAllPointsWithClusters(state: OnlineKMeansState): Array<{
    point: Point
    clusterId: number
}> {
    const result: Array<{ point: Point; clusterId: number }> = []
    for (const cluster of state.clusters) {
        for (const point of cluster.points) {
            result.push({ point, clusterId: cluster.id })
        }
    }
    return result
}

/** Euclidean radius — only meaningful when metric is 'euclidean'. */
export function clusterRadius(cluster: OnlineKMeansCluster): number {
    if (cluster.points.length === 0) return 0
    return Math.max(
        ...cluster.points.map(p =>
            distance(p, { x: cluster.centroidX, y: cluster.centroidY })
        )
    )
}

/** Angle of centroid direction from origin (radians, [0, 2π)). */
export function centroidAngle(cluster: OnlineKMeansCluster): number {
    const angle = Math.atan2(cluster.centroidY, cluster.centroidX)
    return angle < 0 ? angle + 2 * Math.PI : angle
}

/** Angular bisector boundaries between adjacent centroid directions. */
export function computeAngularBoundaries(state: OnlineKMeansState): number[] {
    if (state.clusters.length < 2) return []

    const angles = state.clusters
        .map(c => centroidAngle(c))
        .sort((a, b) => a - b)

    const boundaries: number[] = []
    for (let i = 0; i < angles.length; i++) {
        const next = (i + 1) % angles.length
        let mid: number
        if (next === 0) {
            mid = (angles[i] + angles[next] + 2 * Math.PI) / 2
            if (mid >= 2 * Math.PI) mid -= 2 * Math.PI
        } else {
            mid = (angles[i] + angles[next]) / 2
        }
        boundaries.push(mid)
    }
    return boundaries
}
