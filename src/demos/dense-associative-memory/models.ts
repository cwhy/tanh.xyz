import { numpy as np, tree, valueAndGrad } from '@jax-js/jax'
import { adam, applyUpdates, type OptState } from '@jax-js/optax'
import { initJax } from '../../lib/deeplearning/runtime'

export { initJax, type OptState }

// ── Types ──────────────────────────────────────────────────────────────

export interface DamParams {
    xi: np.Array        // (Nv, Nh) weight matrix — columns are "basic memories"
    theta: np.Array     // scalar threshold
    [key: string]: np.Array
}

export interface SeededRng {
    next: () => number
    nextNormal: () => number
}

// ── RNG ────────────────────────────────────────────────────────────────

export function createSeededRng(seed: number): SeededRng {
    let state = (seed >>> 0) || 1
    const next = () => {
        state ^= state << 13
        state ^= state >>> 17
        state ^= state << 5
        return ((state >>> 0) + 1) / 4294967297
    }
    const nextNormal = () => {
        const u1 = Math.max(next(), 1e-12)
        const u2 = next()
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }
    return { next, nextNormal }
}

// ── Parameter initialization ───────────────────────────────────────────

export function initDamParams(seed: number, Nv: number, Nh: number): DamParams {
    const rng = createSeededRng(seed)
    // Xavier initialization for xi
    const scale = Math.sqrt(2 / (Nv + Nh))
    const xiData = new Float32Array(Nv * Nh)
    for (let i = 0; i < xiData.length; i++) {
        xiData[i] = rng.nextNormal() * scale
    }
    return {
        xi: np.array(xiData).reshape([Nv, Nh]),
        theta: np.array([0.5]),       // scalar, learnable
    }
}

// ── Sharp sigmoid (differentiable Heaviside) ───────────────────────────

/**
 * σ_β(x) = 1 / (1 + exp(-β * x))
 * Approximates the Heaviside step function Θ(x) as β → ∞.
 * We use β ≈ 20 during training.
 */
function sharpSigmoid(x: np.Array, beta: number): np.Array {
    // 1 / (1 + exp(-beta * x))
    return np.reciprocal(np.exp(x.mul(-beta)).add(1))
}

// ── Forward pass (for training) ────────────────────────────────────────

/**
 * Given visible memories V (M × Nv), compute the reconstruction.
 * 
 * h_raw = V · ξ / Nv           → (M, Nh)
 * h     = σ_β(h_raw - θ)       → (M, Nh)  — sharp sigmoid
 * v_hat = h · ξᵀ               → (M, Nv)  — reconstruction
 */
function reconstruct(params: DamParams, V: np.Array, beta: number): np.Array {
    const Nv = V.shape[1]
    // h_raw = V @ xi / Nv — xi consumed here, so callers must keep refs
    const hRaw = np.matmul(V, params.xi.ref).div(Nv)
    // h = sharpSigmoid(hRaw - theta)
    const h = sharpSigmoid(hRaw.sub(params.theta.ref), beta)
    // v_hat = h @ xi^T  — another consumption of xi
    return np.matmul(h, params.xi.ref.transpose())
}

// ── Loss function ──────────────────────────────────────────────────────

/**
 * L(ξ, θ) = (1/M) Σ_m ||v_m - ξ · σ_β(ξᵀ v_m / Nv - θ)||²
 */
function damLoss(params: DamParams, V: np.Array): np.Array {
    const beta = 20
    // V is used twice: once in reconstruct (matmul), once here in sub
    const vHat = reconstruct(params, V.ref, beta)
    return np.square(vHat.sub(V)).mean()
}

// ── Training step ──────────────────────────────────────────────────────

export async function trainStep(
    params: DamParams,
    optState: OptState,
    solver: ReturnType<typeof adam>,
    memories: np.Array,
): Promise<{ params: DamParams; optState: OptState; loss: number }> {
    const [lossVal, grads] = valueAndGrad(damLoss)(tree.ref(params), memories.ref) as [np.Array, DamParams]
    const loss = await lossVal.jsAsync() as number

    const [updates, newOptState] = solver.update(grads, optState)
    const newParams = applyUpdates(params, updates) as DamParams

    return { params: newParams, optState: newOptState, loss }
}

// ── Create optimizer ───────────────────────────────────────────────────

export function createOptimizer(
    params: DamParams,
    learningRate: number,
): { solver: ReturnType<typeof adam>; optState: OptState } {
    const solver = adam(learningRate)
    const optState = solver.init(tree.ref(params))
    return { solver, optState }
}

// ── Recall dynamics ────────────────────────────────────────────────────

/**
 * One iteration of the recall dynamics (discrete Euler step).
 * Matches paper Eq. (1):
 *   dh_μ/dt = -h_μ + (1/Nv) Σ_i ξ_μi v_i
 *   dv_i/dt = -v_i + Σ_μ ξ_iμ F(h_μ - θ)
 * 
 * F is either the hard Heaviside Θ (paper default) or a sharp sigmoid
 * σ_β (matching training), controlled by the beta parameter.
 * beta=0 → hard threshold; beta>0 → sharp sigmoid with that steepness.
 */
export function recallStep(
    xi: Float32Array,
    theta: number,
    v: Float32Array,
    Nv: number,
    Nh: number,
    tauRatio: number,
    beta: number = 0,
): { v: Float32Array; h: Float32Array; hBinary: Float32Array; hActivated: Float32Array } {
    const dtOverTauV = 1 / tauRatio

    // 1. Update hidden: h_new = (1/Nv) * ξᵀ · v
    const hNew = new Float32Array(Nh)
    for (let mu = 0; mu < Nh; mu++) {
        let sum = 0
        for (let i = 0; i < Nv; i++) {
            sum += xi[i * Nh + mu] * v[i]
        }
        hNew[mu] = sum / Nv
    }

    // 2. Apply activation: F(h - θ)
    const hBinary = new Float32Array(Nh)
    const hActivated = new Float32Array(Nh)
    for (let mu = 0; mu < Nh; mu++) {
        hBinary[mu] = hNew[mu] >= theta ? 1 : 0
        if (beta > 0) {
            // Sharp sigmoid: 1 / (1 + exp(-β(h-θ)))
            const x = beta * (hNew[mu] - theta)
            hActivated[mu] = 1 / (1 + Math.exp(-x))
        } else {
            hActivated[mu] = hBinary[mu]
        }
    }

    // 3. Update visible: v_target = ξ · F(h-θ)
    const vTarget = new Float32Array(Nv)
    for (let i = 0; i < Nv; i++) {
        let sum = 0
        for (let mu = 0; mu < Nh; mu++) {
            sum += xi[i * Nh + mu] * hActivated[mu]
        }
        vTarget[i] = sum
    }

    // v_new = (1 - dt/τv) * v + (dt/τv) * vTarget
    const vNew = new Float32Array(Nv)
    for (let i = 0; i < Nv; i++) {
        const raw = (1 - dtOverTauV) * v[i] + dtOverTauV * vTarget[i]
        vNew[i] = Math.max(0, Math.min(1, raw))
    }

    return { v: vNew, h: hNew, hBinary, hActivated }
}

// ── Extract basic memories ─────────────────────────────────────────────

/**
 * Extract the μ-th column of ξ as a visible-space image.
 * This is the "basic memory" that hidden neuron μ encodes.
 */
export function extractBasicMemory(
    xiData: Float32Array,
    Nv: number,
    Nh: number,
    mu: number
): Float32Array {
    const col = new Float32Array(Nv)
    for (let i = 0; i < Nv; i++) {
        col[i] = xiData[i * Nh + mu]
    }
    return col
}

// ── Utilities ──────────────────────────────────────────────────────────

/** Add Gaussian noise to a Float32Array (in-place copy) */
export function addNoise(
    data: Float32Array,
    sigma: number,
    rng: SeededRng
): Float32Array {
    const out = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) {
        out[i] = data[i] + sigma * rng.nextNormal()
    }
    return out
}

/** Compute mean squared error between two Float32Arrays */
export function mse(a: Float32Array, b: Float32Array): number {
    let sum = 0
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i]
        sum += diff * diff
    }
    return sum / a.length
}

/** Compute fraction of hidden neurons that match target */
export function hiddenAccuracy(
    predicted: Float32Array,
    target: Float32Array,
): number {
    let match = 0
    for (let i = 0; i < predicted.length; i++) {
        if (predicted[i] === target[i]) match++
    }
    return match / predicted.length
}
