import { numpy as np, tree, valueAndGrad } from '@jax-js/jax'
import { adam, sgd, applyUpdates, type OptState } from '@jax-js/optax'
import { initJax } from '../../lib/deeplearning/runtime'

export { initJax, type OptState }

export interface MnistMlpParams {
    W1: np.Array
    b1: np.Array
    W2: np.Array
    b2: np.Array
    [key: string]: np.Array
}

export interface SeededRng {
    next: () => number
    nextNormal: () => number
}

export type OptimizerType = 'adam' | 'sgd'
export type OptimizerSolver = ReturnType<typeof adam> | ReturnType<typeof sgd>

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

export function initParams(seed: number, hiddenSize: number = 64): MnistMlpParams {
    const rng = createSeededRng(seed)
    const inputSize = 28 * 28
    const outputSize = 10

    const w1 = new Float32Array(inputSize * hiddenSize)
    const w2 = new Float32Array(hiddenSize * outputSize)
    const scale1 = Math.sqrt(2 / inputSize)
    const scale2 = Math.sqrt(2 / hiddenSize)

    for (let i = 0; i < w1.length; i++) {
        w1[i] = rng.nextNormal() * scale1
    }
    for (let i = 0; i < w2.length; i++) {
        w2[i] = rng.nextNormal() * scale2
    }

    return {
        W1: np.array(w1).reshape([inputSize, hiddenSize]),
        b1: np.zeros([hiddenSize]),
        W2: np.array(w2).reshape([hiddenSize, outputSize]),
        b2: np.zeros([outputSize]),
    }
}

export function createOptimizer(
    params: MnistMlpParams,
    learningRate: number,
    optimizerType: OptimizerType = 'adam'
): {
    solver: OptimizerSolver
    optState: OptState
} {
    const solver = optimizerType === 'adam' ? adam(learningRate) : sgd(learningRate)
    const optState = solver.init(tree.ref(params))
    return { solver, optState }
}

export function predictLogits(params: MnistMlpParams, X: np.Array): np.Array {
    const hidden = np.tanh(np.matmul(X, params.W1).add(params.b1))
    return np.matmul(hidden, params.W2).add(params.b2)
}

export function mseLoss(params: MnistMlpParams, X: np.Array, y: np.Array): np.Array {
    const logits = predictLogits(params, X)
    return np.square(logits.sub(y)).mean()
}

async function gradientNormL2(grads: MnistMlpParams): Promise<number> {
    const normSq = np
        .square(grads.W1.ref)
        .sum()
        .add(np.square(grads.b1.ref).sum())
        .add(np.square(grads.W2.ref).sum())
        .add(np.square(grads.b2.ref).sum())
    const value = await normSq.jsAsync() as number
    return Math.sqrt(Math.max(value, 0))
}

export async function trainOnlineStep(
    params: MnistMlpParams,
    optState: OptState,
    solver: OptimizerSolver,
    X: np.Array,
    y: np.Array
): Promise<{ params: MnistMlpParams; optState: OptState; loss: number; gradNorm: number }> {
    const [lossVal, lossGrad] = valueAndGrad(mseLoss)(tree.ref(params), X, y) as [np.Array, MnistMlpParams]
    const gradNorm = await gradientNormL2(tree.ref(lossGrad))
    const [updates, newOptState] = solver.update(lossGrad, optState)
    const newParams = applyUpdates(params, updates) as MnistMlpParams
    const loss = await lossVal.jsAsync() as number
    return { params: newParams, optState: newOptState, loss, gradNorm }
}

export async function evaluateLoss(
    params: MnistMlpParams,
    X: np.Array,
    y: np.Array
): Promise<number> {
    const value = mseLoss(tree.ref(params), X.ref, y.ref)
    return await value.jsAsync() as number
}

export async function computeSampleGradNorm(
    params: MnistMlpParams,
    X: np.Array,
    y: np.Array
): Promise<number> {
    const [, lossGrad] = valueAndGrad(mseLoss)(tree.ref(params), X, y) as [np.Array, MnistMlpParams]
    return await gradientNormL2(lossGrad)
}

function argmax(values: number[]): number {
    let bestIndex = 0
    let bestValue = Number.NEGATIVE_INFINITY
    for (let i = 0; i < values.length; i++) {
        if (values[i] > bestValue) {
            bestValue = values[i]
            bestIndex = i
        }
    }
    return bestIndex
}

export async function predictSampleLabel(
    params: MnistMlpParams,
    X: np.Array
): Promise<number> {
    const logits = predictLogits(tree.ref(params), X)
    const values = await logits.jsAsync() as number[][]
    const row = values[0] ?? []
    return argmax(row)
}

export async function evaluateAccuracy(
    params: MnistMlpParams,
    X: np.Array,
    labels: Int32Array
): Promise<number> {
    const metrics = await evaluateClassificationMetrics(params, X, labels)
    return metrics.accuracy
}

export async function evaluateClassificationMetrics(
    params: MnistMlpParams,
    X: np.Array,
    labels: Int32Array
): Promise<{ accuracy: number; recallByDigit: number[] }> {
    const logits = predictLogits(tree.ref(params), X.ref)
    const values = await logits.jsAsync() as number[][]
    const n = Math.min(values.length, labels.length)
    if (n === 0) return { accuracy: 0, recallByDigit: Array.from({ length: 10 }, () => 0) }

    let correct = 0
    const totalByDigit = Array.from({ length: 10 }, () => 0)
    const correctByDigit = Array.from({ length: 10 }, () => 0)

    for (let i = 0; i < n; i++) {
        const predicted = argmax(values[i] ?? [])
        const actual = labels[i]
        if (actual >= 0 && actual < 10) {
            totalByDigit[actual] += 1
        }
        if (predicted === actual) {
            correct++
            if (actual >= 0 && actual < 10) {
                correctByDigit[actual] += 1
            }
        }
    }

    const recallByDigit = totalByDigit.map((total, digit) => (
        total > 0 ? correctByDigit[digit] / total : 0
    ))

    return {
        accuracy: correct / n,
        recallByDigit,
    }
}
