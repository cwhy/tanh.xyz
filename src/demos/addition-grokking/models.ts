import { nn, numpy as np, tree, valueAndGrad } from '@jax-js/jax'
import {
    applyOptimizerUpdates,
    createOptimizerState,
    initJax,
    trainStepWithSolver,
    type OptState,
    type OptimizerOptions,
    type OptimizerSolver,
    type OptimizerType,
    type Schedule,
} from '../../lib/deeplearning'
import {
    ADDITION_LM_INPUT_LEN,
    ADDITION_PROMPT_LEN,
    ADDITION_SUM_DIGITS,
    ADDITION_TARGET_LEN,
    ADDITION_VOCAB_SIZE,
} from '../../lib/datasets/jax/ten-digit-addition'

export { initJax, type OptState, type OptimizerSolver, type OptimizerType }

export type ModelType = 'mini-transformer' | 'mlp'

export interface AdditionMlpParams {
    W1: np.Array
    b1: np.Array
    W2: np.Array
    b2: np.Array
    [key: string]: np.Array
}

export interface AdditionMiniTransformerParams {
    tokenEmbedding: np.Array // [14, 7]
    posA: np.Array // [33, 3]
    posB: np.Array // [3, 7]
    ln1Weight: np.Array // [7]
    qkvA: np.Array // [7, 3]
    qkvBq: np.Array // [3, 7]
    qkvBkv: np.Array // [3, 7]
    attnOutA: np.Array // [7, 2]
    attnOutB: np.Array // [2, 7]
    ln2Weight: np.Array // [7]
    ffnUpA: np.Array // [7, 3]
    ffnUpB: np.Array // [3, 14]
    ffnDownA: np.Array // [14, 3]
    ffnDownB: np.Array // [3, 7]
    lnFWeight: np.Array // [7]
    [key: string]: np.Array
}

interface SeededRng {
    next: () => number
    nextNormal: () => number
}

const D_MODEL = 7
const FFN_DIM = 14
const POS_RANK = 3
const QKV_RANK = 3
const ATTN_OUT_RANK = 2
const FFN_RANK = 3

export function miniTransformerParamCount(): number {
    return 456
}

export function createOptimizer(
    params: Record<string, np.Array>,
    learningRate: number | Schedule,
    optimizerType: OptimizerType,
    options: OptimizerOptions = {}
): { solver: OptimizerSolver; optState: OptState } {
    return createOptimizerState(params, learningRate, optimizerType, options)
}

export function initMlpParams(
    seed: number,
    inputSize: number,
    hiddenSize: number,
    outputSize: number
): AdditionMlpParams {
    const rng = createSeededRng(seed)
    const w1 = new Float32Array(inputSize * hiddenSize)
    const w2 = new Float32Array(hiddenSize * outputSize)
    const scale1 = Math.sqrt(2 / inputSize)
    const scale2 = Math.sqrt(2 / hiddenSize)

    for (let i = 0; i < w1.length; i++) w1[i] = rng.nextNormal() * scale1
    for (let i = 0; i < w2.length; i++) w2[i] = rng.nextNormal() * scale2

    return {
        W1: np.array(w1).reshape([inputSize, hiddenSize]),
        b1: np.zeros([hiddenSize]),
        W2: np.array(w2).reshape([hiddenSize, outputSize]),
        b2: np.zeros([outputSize]),
    }
}

export function initMiniTransformerParams(seed: number): AdditionMiniTransformerParams {
    const rng = createSeededRng(seed)
    const initNormal = (size: number, std: number) => {
        const out = new Float32Array(size)
        for (let i = 0; i < size; i++) out[i] = rng.nextNormal() * std
        return out
    }
    const initLowRank = (inDim: number, outDim: number, rank: number): { A: Float32Array; B: Float32Array } => ({
        A: initNormal(inDim * rank, Math.sqrt(2 / (inDim + rank))),
        B: initNormal(rank * outDim, Math.sqrt(2 / (rank + outDim))),
    })

    const attnOut = initLowRank(D_MODEL, D_MODEL, ATTN_OUT_RANK)
    const ffnUp = initLowRank(D_MODEL, FFN_DIM, FFN_RANK)
    const ffnDown = initLowRank(FFN_DIM, D_MODEL, FFN_RANK)

    return {
        tokenEmbedding: np.array(initNormal(ADDITION_VOCAB_SIZE * D_MODEL, 0.02)).reshape([ADDITION_VOCAB_SIZE, D_MODEL]),
        posA: np.array(initNormal(ADDITION_LM_INPUT_LEN * POS_RANK, 0.02)).reshape([ADDITION_LM_INPUT_LEN, POS_RANK]),
        posB: np.array(initNormal(POS_RANK * D_MODEL, 0.02)).reshape([POS_RANK, D_MODEL]),
        ln1Weight: np.ones([D_MODEL]),
        qkvA: np.array(initLowRank(D_MODEL, D_MODEL, QKV_RANK).A).reshape([D_MODEL, QKV_RANK]),
        qkvBq: np.array(initLowRank(D_MODEL, D_MODEL, QKV_RANK).B).reshape([QKV_RANK, D_MODEL]),
        qkvBkv: np.array(initLowRank(D_MODEL, D_MODEL, QKV_RANK).B).reshape([QKV_RANK, D_MODEL]),
        attnOutA: np.array(attnOut.A).reshape([D_MODEL, ATTN_OUT_RANK]),
        attnOutB: np.array(attnOut.B).reshape([ATTN_OUT_RANK, D_MODEL]),
        ln2Weight: np.ones([D_MODEL]),
        ffnUpA: np.array(ffnUp.A).reshape([D_MODEL, FFN_RANK]),
        ffnUpB: np.array(ffnUp.B).reshape([FFN_RANK, FFN_DIM]),
        ffnDownA: np.array(ffnDown.A).reshape([FFN_DIM, FFN_RANK]),
        ffnDownB: np.array(ffnDown.B).reshape([FFN_RANK, D_MODEL]),
        lnFWeight: np.ones([D_MODEL]),
    }
}

export async function trainMlpBatchStep(
    params: AdditionMlpParams,
    optState: OptState,
    solver: OptimizerSolver,
    X: np.Array,
    y: np.Array,
    weightDecay: number
): Promise<{ params: AdditionMlpParams; optState: OptState; loss: number }> {
    return trainStepWithSolver(params, optState, solver, X, y, (p, x, target) => mseLoss(tree.ref(p), x, target, weightDecay))
}

export async function trainMiniTransformerBatchStep(
    params: AdditionMiniTransformerParams,
    optState: OptState,
    solver: OptimizerSolver,
    inputTokens: np.Array,
    labelTokens: np.Array,
    labelMask: np.Array
): Promise<{ params: AdditionMiniTransformerParams; optState: OptState; loss: number }> {
    const [lossVal, lossGrad] = valueAndGrad(transformerCrossEntropyLoss)(
        tree.ref(params),
        inputTokens,
        labelTokens,
        labelMask
    ) as [np.Array, AdditionMiniTransformerParams]

    const loss = await lossVal.jsAsync() as number
    const [updates, nextOptState] = solver.update(lossGrad, optState, tree.ref(params))
    const nextParams = applyOptimizerUpdates(params, updates as AdditionMiniTransformerParams)
    return { params: nextParams, optState: nextOptState, loss }
}

export async function evaluateMlpLoss(
    params: AdditionMlpParams,
    X: np.Array,
    y: np.Array
): Promise<number> {
    const value = mseLoss(tree.ref(params), X.ref, y.ref)
    return await value.jsAsync() as number
}

export async function evaluateMiniTransformerLoss(
    params: AdditionMiniTransformerParams,
    inputTokens: np.Array,
    labelTokens: np.Array,
    labelMask: np.Array
): Promise<number> {
    const value = transformerCrossEntropyLoss(tree.ref(params), inputTokens.ref, labelTokens.ref, labelMask.ref)
    return await value.jsAsync() as number
}

export async function evaluateMlpExactMatch(
    params: AdditionMlpParams,
    X: np.Array,
    targetDigits: Uint8Array
): Promise<number> {
    const logits = predictMlpLogits(tree.ref(params), X.ref)
    const values = await logits.jsAsync() as number[][]
    return exactMatchFromDigitLogits(values, targetDigits, ADDITION_SUM_DIGITS)
}

export async function evaluateMiniTransformerExactMatch(
    params: AdditionMiniTransformerParams,
    promptTokens: np.Array,
    targetDigits: Uint8Array
): Promise<number> {
    const batchSize = promptTokens.shape[0]
    let generated = promptTokens.ref

    for (let i = 0; i < ADDITION_TARGET_LEN; i++) {
        const logits = predictMiniTransformerLogits(tree.ref(params), generated.ref)
        const lastIndex = generated.shape[1] - 1
        const lastTokenLogits = np.take(logits, np.array([lastIndex], { dtype: np.int32 }), 1).reshape([batchSize, ADDITION_VOCAB_SIZE])
        const nextToken = np.argmax(lastTokenLogits, -1).reshape([batchSize, 1])
        generated = np.concatenate([generated, nextToken], 1)
    }

    const tokens = await generated.jsAsync() as number[][]
    let correct = 0
    for (let i = 0; i < batchSize; i++) {
        let allCorrect = true
        for (let d = 0; d < ADDITION_SUM_DIGITS; d++) {
            const predicted = tokens[i]?.[ADDITION_PROMPT_LEN + d] ?? -1
            const expected = targetDigits[i * ADDITION_SUM_DIGITS + d]
            if (predicted !== expected) {
                allCorrect = false
                break
            }
        }
        if (allCorrect) correct++
    }
    return batchSize > 0 ? correct / batchSize : 0
}

function predictMlpLogits(params: AdditionMlpParams, X: np.Array): np.Array {
    const hidden = np.tanh(np.matmul(X, params.W1).add(params.b1))
    return np.matmul(hidden, params.W2).add(params.b2)
}

function mseLoss(
    params: AdditionMlpParams,
    X: np.Array,
    y: np.Array,
    weightDecay: number = 0
): np.Array {
    const logits = predictMlpLogits(tree.ref(params), X)
    const dataLoss = np.square(logits.sub(y)).mean()
    if (weightDecay <= 0) return dataLoss

    const l2 = np.square(params.W1).mean()
        .add(np.square(params.b1).mean())
        .add(np.square(params.W2).mean())
        .add(np.square(params.b2).mean())
    return dataLoss.add(l2.mul(weightDecay))
}

function transformerCrossEntropyLoss(
    params: AdditionMiniTransformerParams,
    inputTokens: np.Array,
    labelTokens: np.Array,
    labelMask: np.Array
): np.Array {
    const logits = predictMiniTransformerLogits(tree.ref(params), inputTokens)
    const safeLabels = np.clip(labelTokens, 0, ADDITION_VOCAB_SIZE - 1).astype(np.int32)
    const oneHot = nn.oneHot(safeLabels, ADDITION_VOCAB_SIZE)
    const logProbs = nn.logSoftmax(logits, -1)
    const tokenNll = oneHot.mul(logProbs).sum(-1).neg()
    const masked = tokenNll.mul(labelMask.ref)
    return masked.sum().div(labelMask.sum().add(1e-6))
}

function predictMiniTransformerLogits(
    params: AdditionMiniTransformerParams,
    tokenIds: np.Array
): np.Array {
    // Workaround: avoid `np.take` gather in the training graph, because jax-js
    // does not yet implement the gather transpose rule needed by backprop.
    const batchSize = tokenIds.shape[0]
    const seqLen = tokenIds.shape[1]

    const tokenOneHot = nn.oneHot(tokenIds.astype(np.int32), ADDITION_VOCAB_SIZE)
    const tokenEmb = np.matmul(tokenOneHot, params.tokenEmbedding.ref)

    const posEmb = np.matmul(params.posA, params.posB)
    const posIndices = np.arange(0, seqLen, 1, { dtype: np.int32 })
    const posOneHot = nn.oneHot(posIndices, ADDITION_LM_INPUT_LEN).reshape([1, seqLen, ADDITION_LM_INPUT_LEN])
    const pos = np.matmul(posOneHot, posEmb)

    let x = tokenEmb.add(pos)

    const norm1 = rmsNorm(x.ref, params.ln1Weight.ref)
    const attnOut = selfAttention(norm1, params)
    x = x.add(attnOut)

    const norm2 = rmsNorm(x.ref, params.ln2Weight.ref)
    const ffHidden = nn.gelu(lowRankLinear(norm2.ref, params.ffnUpA.ref, params.ffnUpB.ref))
    const ffOut = lowRankLinear(ffHidden, params.ffnDownA.ref, params.ffnDownB.ref)
    x = x.add(ffOut)

    const finalNorm = rmsNorm(x, params.lnFWeight)
    const tokenEmbT = params.tokenEmbedding.transpose([1, 0])
    return np.matmul(finalNorm, tokenEmbT).reshape([batchSize, seqLen, ADDITION_VOCAB_SIZE])
}

function selfAttention(
    x: np.Array,
    params: AdditionMiniTransformerParams
): np.Array {
    const batchSize = x.shape[0]
    const seqLen = x.shape[1]

    const bottleneck = np.matmul(x, params.qkvA)
    const q = np.matmul(bottleneck.ref, params.qkvBq)
    const kv = np.matmul(bottleneck, params.qkvBkv)
    const q4 = q.reshape([batchSize, seqLen, 1, D_MODEL])
    const kv4 = kv.reshape([batchSize, seqLen, 1, D_MODEL])

    const attended = nn.dotProductAttention(q4, kv4.ref, kv4, { isCausal: true })
    const attn3 = attended.reshape([batchSize, seqLen, D_MODEL])
    return lowRankLinear(attn3, params.attnOutA, params.attnOutB)
}

function lowRankLinear(x: np.Array, A: np.Array, B: np.Array): np.Array {
    return np.matmul(np.matmul(x, A), B)
}

function rmsNorm(x: np.Array, weight: np.Array, eps: number = 1e-8): np.Array {
    const rms = np.sqrt(np.mean(np.square(x.ref), -1, { keepdims: true }).add(eps))
    return x.div(rms).mul(weight)
}

function exactMatchFromDigitLogits(values: number[][], targetDigits: Uint8Array, digitsPerSample: number): number {
    const sampleCount = Math.min(values.length, Math.floor(targetDigits.length / digitsPerSample))
    if (sampleCount <= 0) return 0

    let exact = 0
    for (let i = 0; i < sampleCount; i++) {
        const row = values[i] ?? []
        let allCorrect = true
        for (let pos = 0; pos < digitsPerSample; pos++) {
            const predicted = argmaxSlice(row, pos * 10, pos * 10 + 10)
            const actual = targetDigits[i * digitsPerSample + pos]
            if (predicted !== actual) {
                allCorrect = false
                break
            }
        }
        if (allCorrect) exact++
    }

    return exact / sampleCount
}

function argmaxSlice(values: number[], start: number, end: number): number {
    let bestOffset = 0
    let bestValue = Number.NEGATIVE_INFINITY
    for (let i = start; i < end; i++) {
        const value = values[i] ?? Number.NEGATIVE_INFINITY
        if (value > bestValue) {
            bestValue = value
            bestOffset = i - start
        }
    }
    return bestOffset
}

function createSeededRng(seed: number): SeededRng {
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
