import {
    DIGIT_COUNT,
    MNIST_SIZE,
    type PatchAgent,
    type SheafAdmmSession,
    type SheafAdmmSnapshot,
} from './algorithm'

interface TensorManifestEntry {
    shape: number[]
    dtype?: 'float32'
    file?: string
}

interface TrainedSheafAdmmMnistManifest {
    format: 'sheaf-admm-mnist-v1'
    config: Partial<TrainedSheafAdmmMnistConfig>
    baseUrl?: string
    tensors: Record<string, TensorManifestEntry>
}

export interface TrainedSheafAdmmMnistConfig {
    imageSize: number
    patchSize: number
    stride: number
    connectivity: 4 | 8
    numClasses: number
    numAgents: number
    numEdges: number
    dV: number
    dE: number
    hiddenDim: number
    numDirections: 8
    loraRank: number
    loraAlpha: number
    l1Weight: number
    rho: number
    cgIters: number
    tikhonovEps: number
    zMode: 'project'
    zInit: 'h' | 'zeros'
    evalIterations: number
    checkpointKind: string
    sourceConfig?: unknown
}

interface Tensor {
    shape: number[]
    values: Float32Array
}

interface EncoderParams {
    inputProjKernel: Tensor
    inputProjBias: Tensor
    blockNormScale: Tensor
    blockDense1Kernel: Tensor
    blockDense1Bias: Tensor
    blockDense2Kernel: Tensor
    blockDense2Bias: Tensor
    commKernel: Tensor
    commBias: Tensor
    commNormScale: Tensor
    commNormBias: Tensor
    qDiagKernel: Tensor
    qDiagBias: Tensor
    qKernel: Tensor
    qBias: Tensor
    loraPreLnScale: Tensor
    loraPreLnBias: Tensor
    loraAKernel: Tensor
    loraABias: Tensor
    loraBKernel: Tensor
    loraBBias: Tensor
}

interface TrainedParams {
    encoder: EncoderParams
    restrictionShared: Tensor
    decoderKernel: Tensor
    decoderBias: Tensor
}

interface EncodedAgents {
    h: Float32Array
    qDiag: Float32Array
    q: Float32Array
    loraA: Float32Array
    loraB: Float32Array
}

interface TrainedEdge {
    from: number
    to: number
    dirFromTo: number
    dirToFrom: number
}

interface TrainedState {
    x: Float32Array
    z: Float32Array
    y: Float32Array
    zPrev: Float32Array
}

export interface TrainedSheafAdmmMnistModel {
    config: TrainedSheafAdmmMnistConfig
    params: TrainedParams
    edges: TrainedEdge[]
    centers: Array<{ row: number; col: number; y: number; x: number }>
}

export interface TrainedSheafAdmmMnistRun {
    session: SheafAdmmSession
    history: SheafAdmmSnapshot[]
}

export const LOCAL_TRAINED_SHEAF_ADMM_MNIST_MANIFEST_URL = '/data/demos/sheaf-admm-mnist/params/v1/manifest.json'
export const DEFAULT_TRAINED_SHEAF_ADMM_MNIST_MANIFEST_URL =
    import.meta.env.VITE_SHEAF_ADMM_MNIST_MODEL_MANIFEST_URL ?? LOCAL_TRAINED_SHEAF_ADMM_MNIST_MANIFEST_URL

const CONFIG_DEFAULTS: TrainedSheafAdmmMnistConfig = {
    imageSize: 28,
    patchSize: 3,
    stride: 3,
    connectivity: 8,
    numClasses: 10,
    numAgents: 81,
    numEdges: 272,
    dV: 32,
    dE: 24,
    hiddenDim: 256,
    numDirections: 8,
    loraRank: 8,
    loraAlpha: 1,
    l1Weight: 0.006337180166370117,
    rho: 0.12,
    cgIters: 5,
    tikhonovEps: 1e-5,
    zMode: 'project',
    zInit: 'h',
    evalIterations: 20,
    checkpointKind: 'ema_params',
}

const TENSOR_NAMES = {
    inputProjKernel: 'encoder.inputProj.kernel',
    inputProjBias: 'encoder.inputProj.bias',
    blockNormScale: 'encoder.block0.norm.scale',
    blockDense1Kernel: 'encoder.block0.dense1.kernel',
    blockDense1Bias: 'encoder.block0.dense1.bias',
    blockDense2Kernel: 'encoder.block0.dense2.kernel',
    blockDense2Bias: 'encoder.block0.dense2.bias',
    commKernel: 'encoder.commHead.kernel',
    commBias: 'encoder.commHead.bias',
    commNormScale: 'encoder.commNorm.scale',
    commNormBias: 'encoder.commNorm.bias',
    qDiagKernel: 'encoder.qDiag.kernel',
    qDiagBias: 'encoder.qDiag.bias',
    qKernel: 'encoder.q.kernel',
    qBias: 'encoder.q.bias',
    loraPreLnScale: 'encoder.loraPreLn.scale',
    loraPreLnBias: 'encoder.loraPreLn.bias',
    loraAKernel: 'encoder.loraA.kernel',
    loraABias: 'encoder.loraA.bias',
    loraBKernel: 'encoder.loraB.kernel',
    loraBBias: 'encoder.loraB.bias',
} as const

export async function loadTrainedSheafAdmmMnistModel(
    manifestUrl: string = DEFAULT_TRAINED_SHEAF_ADMM_MNIST_MANIFEST_URL
): Promise<TrainedSheafAdmmMnistModel> {
    const manifestResponse = await fetch(manifestUrl)
    if (!manifestResponse.ok) {
        throw new Error(`Failed to fetch Sheaf-ADMM MNIST manifest: ${manifestResponse.status}`)
    }
    const manifest = await manifestResponse.json() as TrainedSheafAdmmMnistManifest
    if (manifest.format !== 'sheaf-admm-mnist-v1') {
        throw new Error(`Unsupported Sheaf-ADMM MNIST manifest format: ${manifest.format}`)
    }

    const config = { ...CONFIG_DEFAULTS, ...manifest.config }
    const manifestBase = new URL(manifestUrl, globalThis.location?.href)
    const baseUrl = manifest.baseUrl ? new URL(manifest.baseUrl, manifestBase).href : manifestBase.href.replace(/[^/]*$/, '')

    const load = async (name: string): Promise<Tensor> => {
        const entry = manifest.tensors[name]
        if (!entry) throw new Error(`Missing Sheaf-ADMM MNIST tensor "${name}"`)
        if (entry.dtype && entry.dtype !== 'float32') throw new Error(`Unsupported tensor dtype for "${name}": ${entry.dtype}`)
        const tensorUrl = new URL(entry.file ?? `${name}.f32.bin`, baseUrl).href
        const response = await fetch(tensorUrl)
        if (!response.ok) throw new Error(`Failed to fetch Sheaf-ADMM MNIST tensor "${name}": ${response.status}`)
        return { shape: entry.shape, values: new Float32Array(await response.arrayBuffer()) }
    }

    const encoder = Object.fromEntries(
        await Promise.all(
            Object.entries(TENSOR_NAMES).map(async ([key, name]) => [key, await load(name)])
        )
    ) as unknown as EncoderParams

    const model: TrainedSheafAdmmMnistModel = {
        config,
        params: {
            encoder,
            restrictionShared: await load('restriction.shared'),
            decoderKernel: await load('decoder.cls.kernel'),
            decoderBias: await load('decoder.cls.bias'),
        },
        centers: makeCenters(config),
        edges: [],
    }
    model.edges = makeTrainedEdges(model.centers, config)
    return model
}

export function runTrainedSheafAdmmMnist(
    model: TrainedSheafAdmmMnistModel,
    pixels: Float32Array,
    iterations: number
): TrainedSheafAdmmMnistRun {
    const patches = patchify(pixels, model.config)
    const encoded = encodeAgents(model, patches)
    const agents = makePatchAgents(model, pixels, encoded)
    const history: SheafAdmmSnapshot[] = []

    let state = initialState(model, encoded)
    history.push(makeSnapshot(model, agents, encoded, state, 0))

    const limit = Math.max(0, Math.round(iterations))
    for (let iteration = 1; iteration <= limit; iteration++) {
        state = stepState(model, encoded, state)
        history.push(makeSnapshot(model, agents, encoded, state, iteration))
    }

    return {
        session: {
            agents,
            edges: history[0].edges,
            initial: history[0],
        },
        history,
    }
}

function makeCenters(config: TrainedSheafAdmmMnistConfig): Array<{ row: number; col: number; y: number; x: number }> {
    const centers = []
    const start = Math.floor(config.patchSize / 2)
    let row = 0
    for (let y = start; y < config.imageSize; y += config.stride) {
        let col = 0
        for (let x = start; x < config.imageSize; x += config.stride) {
            centers.push({ row, col, y, x })
            col += 1
        }
        row += 1
    }
    return centers
}

function makeTrainedEdges(
    centers: Array<{ row: number; col: number; y: number; x: number }>,
    config: TrainedSheafAdmmMnistConfig
): TrainedEdge[] {
    const byGrid = new Map(centers.map((center, index) => [`${center.row}:${center.col}`, index]))
    const offsets = config.connectivity === 8
        ? [[0, 1], [1, 0], [1, 1], [1, -1]]
        : [[0, 1], [1, 0]]
    const edges: TrainedEdge[] = []
    for (const [from, center] of centers.entries()) {
        for (const [dr, dc] of offsets) {
            const to = byGrid.get(`${center.row + dr}:${center.col + dc}`)
            if (to === undefined) continue
            const other = centers[to]
            edges.push({
                from,
                to,
                dirFromTo: directionIndex(other.y - center.y, other.x - center.x),
                dirToFrom: directionIndex(center.y - other.y, center.x - other.x),
            })
        }
    }
    return edges
}

function directionIndex(dy: number, dx: number): number {
    const north = dy < 0
    const south = dy > 0
    const east = dx > 0
    const west = dx < 0
    if (north && east) return 1
    if (north && west) return 7
    if (north) return 0
    if (south && east) return 3
    if (south && west) return 5
    if (south) return 4
    return east ? 2 : 6
}

function patchify(pixels: Float32Array, config: TrainedSheafAdmmMnistConfig): Float32Array {
    const centers = makeCenters(config)
    const out = new Float32Array(centers.length * config.patchSize * config.patchSize)
    const pad = Math.floor(config.patchSize / 2)
    for (const [agent, center] of centers.entries()) {
        let cursor = agent * config.patchSize * config.patchSize
        for (let py = 0; py < config.patchSize; py++) {
            for (let px = 0; px < config.patchSize; px++) {
                const y = center.y + py - pad
                const x = center.x + px - pad
                out[cursor++] = y >= 0 && y < MNIST_SIZE && x >= 0 && x < MNIST_SIZE
                    ? pixels[y * MNIST_SIZE + x]
                    : 0
            }
        }
    }
    return out
}

function encodeAgents(model: TrainedSheafAdmmMnistModel, patches: Float32Array): EncodedAgents {
    const { config, params } = model
    const N = config.numAgents
    const dV = config.dV
    const dE = config.dE
    const rank = config.loraRank
    const K = config.numDirections
    const patchDim = config.patchSize * config.patchSize

    const h = new Float32Array(N * dV)
    const qDiag = new Float32Array(N * dV)
    const q = new Float32Array(N * dV)
    const loraA = new Float32Array(N * K * dE * rank)
    const loraB = new Float32Array(N * K * dV * rank)

    for (let agent = 0; agent < N; agent++) {
        const patch = patches.subarray(agent * patchDim, (agent + 1) * patchDim)
        let feats = denseVec(patch, params.encoder.inputProjKernel, params.encoder.inputProjBias)
        const blockInput = feats
        let y = rmsNorm(feats, params.encoder.blockNormScale)
        y = geluVec(denseVec(y, params.encoder.blockDense1Kernel, params.encoder.blockDense1Bias))
        y = denseVec(y, params.encoder.blockDense2Kernel, params.encoder.blockDense2Bias)
        feats = addVec(blockInput, y)

        const comm = layerNorm(
            denseVec(feats, params.encoder.commKernel, params.encoder.commBias),
            params.encoder.commNormScale,
            params.encoder.commNormBias
        )
        h.set(comm, agent * dV)

        const qd = denseVec(feats, params.encoder.qDiagKernel, params.encoder.qDiagBias)
        for (let i = 0; i < dV; i++) qDiag[agent * dV + i] = softplus(qd[i]) + 1e-4
        q.set(denseVec(feats, params.encoder.qKernel, params.encoder.qBias), agent * dV)

        const loraFeats = layerNorm(feats, params.encoder.loraPreLnScale, params.encoder.loraPreLnBias)
        loraA.set(denseVec(loraFeats, params.encoder.loraAKernel, params.encoder.loraABias), agent * K * dE * rank)
        loraB.set(denseVec(loraFeats, params.encoder.loraBKernel, params.encoder.loraBBias), agent * K * dV * rank)
    }

    return { h, qDiag, q, loraA, loraB }
}

function initialState(model: TrainedSheafAdmmMnistModel, encoded: EncodedAgents): TrainedState {
    const stateSize = model.config.numAgents * model.config.dV
    const z = model.config.zInit === 'h' ? new Float32Array(encoded.h) : new Float32Array(stateSize)
    return {
        x: new Float32Array(z),
        z,
        y: new Float32Array(stateSize),
        zPrev: new Float32Array(z),
    }
}

function stepState(
    model: TrainedSheafAdmmMnistModel,
    encoded: EncodedAgents,
    previous: TrainedState
): TrainedState {
    const { config } = model
    const stateSize = config.numAgents * config.dV
    const x = new Float32Array(stateSize)
    const rho = config.rho

    for (let i = 0; i < stateSize; i++) {
        const a = encoded.qDiag[i] + rho
        const t = (rho * (previous.z[i] - previous.y[i]) - encoded.q[i]) / a
        x[i] = softThreshold(t, config.l1Weight / a)
    }

    const zTarget = new Float32Array(stateSize)
    for (let i = 0; i < stateSize; i++) zTarget[i] = x[i] + previous.y[i]
    const z = projectConsensus(model, encoded, zTarget, previous.z)

    const y = new Float32Array(stateSize)
    for (let i = 0; i < stateSize; i++) y[i] = previous.y[i] + x[i] - z[i]
    return { x, z, y, zPrev: previous.z }
}

function projectConsensus(
    model: TrainedSheafAdmmMnistModel,
    encoded: EncodedAgents,
    zTarget: Float32Array,
    zPrev: Float32Array
): Float32Array {
    const b = laplacianApply(model, encoded, zTarget)
    const x = new Float32Array(zTarget.length)
    for (let i = 0; i < x.length; i++) x[i] = zTarget[i] - zPrev[i]
    const matvec = (values: Float32Array) => {
        const out = laplacianApply(model, encoded, values)
        for (let i = 0; i < out.length; i++) out[i] += model.config.tikhonovEps * values[i]
        return out
    }
    const w = conjugateGradient(matvec, b, x, model.config.cgIters)
    const z = new Float32Array(zTarget.length)
    for (let i = 0; i < z.length; i++) z[i] = zTarget[i] - w[i]
    return z
}

function conjugateGradient(
    matvec: (values: Float32Array) => Float32Array,
    b: Float32Array,
    x0: Float32Array,
    iterations: number
): Float32Array {
    const x = new Float32Array(x0)
    let ap = matvec(x)
    const r = new Float32Array(b.length)
    const p = new Float32Array(b.length)
    for (let i = 0; i < b.length; i++) {
        r[i] = b[i] - ap[i]
        p[i] = r[i]
    }
    let rTr = dot(r, r)

    for (let iter = 0; iter < iterations; iter++) {
        ap = matvec(p)
        const alpha = rTr / (dot(p, ap) + 1e-8)
        for (let i = 0; i < x.length; i++) {
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
        }
        const nextRTr = dot(r, r)
        const beta = nextRTr / (rTr + 1e-8)
        for (let i = 0; i < p.length; i++) p[i] = r[i] + beta * p[i]
        rTr = nextRTr
    }
    return x
}

function laplacianApply(
    model: TrainedSheafAdmmMnistModel,
    encoded: EncodedAgents,
    z: Float32Array
): Float32Array {
    const { config, params, edges } = model
    const out = new Float32Array(z.length)
    const residual = new Float32Array(config.dE)
    const contribU = new Float32Array(config.dV)
    const contribV = new Float32Array(config.dV)

    for (const edge of edges) {
        applyEndpoint(model, encoded, edge.from, edge.dirFromTo, z, residual, 1)
        applyEndpoint(model, encoded, edge.to, edge.dirToFrom, z, residual, -1)

        adjointEndpoint(model, encoded, edge.from, edge.dirFromTo, residual, contribU)
        adjointEndpoint(model, encoded, edge.to, edge.dirToFrom, residual, contribV)
        for (let d = 0; d < config.dV; d++) {
            out[edge.from * config.dV + d] += contribU[d]
            out[edge.to * config.dV + d] -= contribV[d]
        }
    }

    void params
    return out
}

function edgeResiduals(
    model: TrainedSheafAdmmMnistModel,
    encoded: EncodedAgents,
    z: Float32Array
): { mismatches: number[]; rms: number } {
    const residual = new Float32Array(model.config.dE)
    const mismatches: number[] = []
    let sum = 0
    for (const edge of model.edges) {
        applyEndpoint(model, encoded, edge.from, edge.dirFromTo, z, residual, 1)
        applyEndpoint(model, encoded, edge.to, edge.dirToFrom, z, residual, -1)
        let edgeSum = 0
        for (let i = 0; i < residual.length; i++) edgeSum += residual[i] * residual[i]
        mismatches.push(Math.sqrt(edgeSum / model.config.dE))
        sum += edgeSum
    }
    return { mismatches, rms: Math.sqrt(sum / Math.max(1, model.edges.length * model.config.dE)) }
}

function applyEndpoint(
    model: TrainedSheafAdmmMnistModel,
    encoded: EncodedAgents,
    agent: number,
    slot: number,
    z: Float32Array,
    out: Float32Array,
    sign: 1 | -1
) {
    const { config, params } = model
    if (sign === 1) out.fill(0)
    const scale = config.loraAlpha / config.loraRank
    const btz = new Float32Array(config.loraRank)
    const zOffset = agent * config.dV
    const aOffset = ((agent * config.numDirections + slot) * config.dE) * config.loraRank
    const bOffset = ((agent * config.numDirections + slot) * config.dV) * config.loraRank

    for (let e = 0; e < config.dE; e++) {
        let value = 0
        for (let d = 0; d < config.dV; d++) {
            value += params.restrictionShared.values[e * config.dV + d] * z[zOffset + d]
        }
        out[e] += sign * value
    }

    for (let r = 0; r < config.loraRank; r++) {
        let value = 0
        for (let d = 0; d < config.dV; d++) {
            value += encoded.loraB[bOffset + d * config.loraRank + r] * z[zOffset + d]
        }
        btz[r] = value
    }
    for (let e = 0; e < config.dE; e++) {
        let value = 0
        for (let r = 0; r < config.loraRank; r++) {
            value += encoded.loraA[aOffset + e * config.loraRank + r] * btz[r]
        }
        out[e] += sign * scale * value
    }
}

function adjointEndpoint(
    model: TrainedSheafAdmmMnistModel,
    encoded: EncodedAgents,
    agent: number,
    slot: number,
    residual: Float32Array,
    out: Float32Array
) {
    const { config, params } = model
    out.fill(0)
    const scale = config.loraAlpha / config.loraRank
    const atr = new Float32Array(config.loraRank)
    const aOffset = ((agent * config.numDirections + slot) * config.dE) * config.loraRank
    const bOffset = ((agent * config.numDirections + slot) * config.dV) * config.loraRank

    for (let d = 0; d < config.dV; d++) {
        let value = 0
        for (let e = 0; e < config.dE; e++) {
            value += params.restrictionShared.values[e * config.dV + d] * residual[e]
        }
        out[d] = value
    }
    for (let r = 0; r < config.loraRank; r++) {
        let value = 0
        for (let e = 0; e < config.dE; e++) {
            value += encoded.loraA[aOffset + e * config.loraRank + r] * residual[e]
        }
        atr[r] = value
    }
    for (let d = 0; d < config.dV; d++) {
        let value = 0
        for (let r = 0; r < config.loraRank; r++) {
            value += encoded.loraB[bOffset + d * config.loraRank + r] * atr[r]
        }
        out[d] += scale * value
    }
}

function makeSnapshot(
    model: TrainedSheafAdmmMnistModel,
    agents: PatchAgent[],
    encoded: EncodedAgents,
    state: TrainedState,
    iteration: number
): SheafAdmmSnapshot {
    const xLogits = decodeAgents(model, state.x)
    const zLogits = decodeAgents(model, state.z)
    const probabilities = meanSoftmax(xLogits, model.config.numAgents, model.config.numClasses)
    const edgeStats = edgeResiduals(model, encoded, state.z)
    const primal = rmsDifference(state.x, state.z)
    const dual = model.config.rho * rmsDifference(state.z, state.zPrev)

    return {
        iteration,
        x: logitsToRows(xLogits, model.config.numAgents, model.config.numClasses),
        z: logitsToRows(zLogits, model.config.numAgents, model.config.numClasses),
        u: latentPrefixRows(state.y, model.config.numAgents, model.config.dV, model.config.numClasses),
        prediction: argmax(probabilities),
        localPrediction: argmax(meanSoftmax(xLogits, model.config.numAgents, model.config.numClasses)),
        probabilities,
        primalResidual: primal,
        consensusResidual: edgeStats.rms,
        dualEnergy: dual,
        edges: model.edges.map((edge, index) => ({
            from: edge.from,
            to: edge.to,
            weight: 1,
            mismatch: edgeStats.mismatches[index] ?? 0,
        })),
    }
}

function makePatchAgents(
    model: TrainedSheafAdmmMnistModel,
    pixels: Float32Array,
    encoded: EncodedAgents
): PatchAgent[] {
    const initialLogits = decodeAgents(model, encoded.h)
    return model.centers.map((center, id) => ({
        id,
        row: center.row,
        col: center.col,
        ink: patchInk(pixels, center.y, center.x, model.config.patchSize),
        target: Array.from(initialLogits.subarray(id * DIGIT_COUNT, (id + 1) * DIGIT_COUNT)),
        reliability: Array.from({ length: DIGIT_COUNT }, () => 1),
    }))
}

function decodeAgents(model: TrainedSheafAdmmMnistModel, states: Float32Array): Float32Array {
    const { config, params } = model
    const out = new Float32Array(config.numAgents * config.numClasses)
    for (let agent = 0; agent < config.numAgents; agent++) {
        for (let c = 0; c < config.numClasses; c++) {
            let value = params.decoderBias.values[c]
            for (let d = 0; d < config.dV; d++) {
                value += states[agent * config.dV + d] * params.decoderKernel.values[d * config.numClasses + c]
            }
            out[agent * config.numClasses + c] = value
        }
    }
    return out
}

function denseVec(input: Float32Array, kernel: Tensor, bias: Tensor): Float32Array {
    const inDim = kernel.shape[0]
    const outDim = kernel.shape[1]
    const out = new Float32Array(outDim)
    for (let j = 0; j < outDim; j++) {
        let value = bias.values[j] ?? 0
        for (let i = 0; i < inDim; i++) value += input[i] * kernel.values[i * outDim + j]
        out[j] = value
    }
    return out
}

function rmsNorm(input: Float32Array, scale: Tensor): Float32Array {
    let meanSquare = 0
    for (const value of input) meanSquare += value * value
    const inv = 1 / Math.sqrt(meanSquare / input.length + 1e-6)
    const out = new Float32Array(input.length)
    for (let i = 0; i < input.length; i++) out[i] = input[i] * inv * scale.values[i]
    return out
}

function layerNorm(input: Float32Array, scale: Tensor, bias: Tensor): Float32Array {
    let mean = 0
    for (const value of input) mean += value
    mean /= input.length
    let variance = 0
    for (const value of input) {
        const diff = value - mean
        variance += diff * diff
    }
    const inv = 1 / Math.sqrt(variance / input.length + 1e-6)
    const out = new Float32Array(input.length)
    for (let i = 0; i < input.length; i++) out[i] = (input[i] - mean) * inv * scale.values[i] + bias.values[i]
    return out
}

function geluVec(input: Float32Array): Float32Array {
    const out = new Float32Array(input.length)
    for (let i = 0; i < input.length; i++) {
        const x = input[i]
        out[i] = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))
    }
    return out
}

function addVec(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(a.length)
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]
    return out
}

function softplus(value: number): number {
    return Math.log1p(Math.exp(-Math.abs(value))) + Math.max(value, 0)
}

function softThreshold(value: number, threshold: number): number {
    if (value > threshold) return value - threshold
    if (value < -threshold) return value + threshold
    return 0
}

function dot(a: Float32Array, b: Float32Array): number {
    let out = 0
    for (let i = 0; i < a.length; i++) out += a[i] * b[i]
    return out
}

function rmsDifference(a: Float32Array, b: Float32Array): number {
    let sum = 0
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i]
        sum += diff * diff
    }
    return Math.sqrt(sum / Math.max(1, a.length))
}

function patchInk(pixels: Float32Array, centerY: number, centerX: number, patchSize: number): number {
    const pad = Math.floor(patchSize / 2)
    let sum = 0
    let count = 0
    for (let py = 0; py < patchSize; py++) {
        for (let px = 0; px < patchSize; px++) {
            const y = centerY + py - pad
            const x = centerX + px - pad
            if (y >= 0 && y < MNIST_SIZE && x >= 0 && x < MNIST_SIZE) {
                sum += pixels[y * MNIST_SIZE + x]
                count += 1
            }
        }
    }
    return sum / Math.max(1, count)
}

function meanSoftmax(logits: Float32Array, rows: number, cols: number): number[] {
    const out = Array.from({ length: cols }, () => 0)
    for (let row = 0; row < rows; row++) {
        let max = Number.NEGATIVE_INFINITY
        for (let c = 0; c < cols; c++) max = Math.max(max, logits[row * cols + c])
        let sum = 0
        const exp = new Float32Array(cols)
        for (let c = 0; c < cols; c++) {
            exp[c] = Math.exp(logits[row * cols + c] - max)
            sum += exp[c]
        }
        for (let c = 0; c < cols; c++) out[c] += exp[c] / Math.max(sum, 1e-9)
    }
    for (let c = 0; c < cols; c++) out[c] /= rows
    return out
}

function logitsToRows(logits: Float32Array, rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, (_, row) => Array.from(logits.subarray(row * cols, (row + 1) * cols)))
}

function latentPrefixRows(latent: Float32Array, rows: number, dV: number, cols: number): number[][] {
    return Array.from({ length: rows }, (_, row) => Array.from(latent.subarray(row * dV, row * dV + cols)))
}

function argmax(values: readonly number[]): number {
    let best = 0
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i
    }
    return best
}
