/**
 * Simple neural network for binary classification using jax-js
 * Implements a 2-layer MLP with ReLU activation
 */

import { numpy as np, init, defaultDevice, valueAndGrad, tree } from '@jax-js/jax'
import { adam, applyUpdates, type OptState } from '@jax-js/optax'

export interface NetworkParams {
    W1: np.Array  // [2, hiddenSize]
    b1: np.Array  // [hiddenSize]
    W2: np.Array  // [hiddenSize, 1]
    b2: np.Array  // [1]
    [key: string]: np.Array  // Index signature for JsTree compatibility
}

export interface TrainingState {
    params: NetworkParams
    optState: OptState
    epoch: number
    loss: number
}

let initialized = false

/**
 * Initialize jax-js with WebGPU or WASM fallback
 */
export async function initJax(): Promise<string> {
    if (initialized) return defaultDevice() as string

    const devices = await init()
    if (devices.includes('webgpu')) {
        defaultDevice('webgpu')
        console.log('Using WebGPU backend')
    } else {
        console.log('WebGPU not available, using WASM backend')
    }
    initialized = true
    return defaultDevice() as string
}

/**
 * Initialize network parameters with Xavier initialization
 */
export function initParams(hiddenSize: number = 16): NetworkParams {
    // Increase scale for first layer to ensure we hit non-linear region of tanh
    // sqrt(2/input_dim) is He init, which is better. input_dim=2.
    // We multiply by 2.0 to be safe and ensure non-linearity.
    const scale1 = Math.sqrt(2 / 2) * 2.0
    const scale2 = Math.sqrt(2 / (hiddenSize + 1))

    return {
        W1: np.array(
            Array.from({ length: 2 }, () =>
                Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * 2 * scale1)
            )
        ),
        b1: np.zeros([hiddenSize]),
        W2: np.array(
            Array.from({ length: hiddenSize }, () =>
                [(Math.random() - 0.5) * 2 * scale2]
            )
        ),
        b2: np.zeros([1]),
    }
}

/**
 * Initialize optimizer state
 */
export function initOptimizer(params: NetworkParams, learningRate: number): OptState {
    const solver = adam(learningRate)
    return solver.init(tree.ref(params))
}

/**
 * Forward pass through the network
 */
export function predict(
    params: NetworkParams,
    X: np.Array
): np.Array {
    // Hidden layer: tanh(X @ W1 + b1)
    const h1 = np.tanh(
        np.matmul(X, params.W1).add(params.b1)
    )

    // Output layer: sigmoid(h1 @ W2 + b2)
    const logits = np.matmul(h1, params.W2).add(params.b2)

    // Sigmoid: 1 / (1 + exp(-x))
    return np.reciprocal(np.exp(logits.neg()).add(1))
}

/**
 * Binary cross-entropy loss
 * Structured to only use y once to avoid reference counting issues
 */
export function loss(params: NetworkParams, X: np.Array, y: np.Array): np.Array {
    const preds = predict(params, X)

    // Binary cross-entropy using a formulation that only uses y once
    // BCE = -mean(y * log(p) + (1-y) * log(1-p))
    // We can rewrite as: -mean(y * (log(p) - log(1-p)) + log(1-p))
    // Note: preds is used twice, so we need preds.ref for the first usage
    const logP = np.log(preds.ref.add(1e-7))
    const log1mP = np.log(np.array(1).sub(preds).add(1e-7))

    // This uses y only once, but log1mP is used twice!
    // log1mP is used in sub() and add()
    return y.mul(logP.sub(log1mP.ref)).add(log1mP).mean().neg()
}

/**
 * Single training step with Adam optimizer
 */
export function trainStep(
    params: NetworkParams,
    optState: OptState,
    X: np.Array,
    y: np.Array,
    learningRate: number
): { params: NetworkParams; optState: OptState; loss: number } {
    // Compute loss and gradients using valueAndGrad
    // Note: valueAndGrad will consume X and y, but not params (we use tree.ref)
    const [lossVal, lossGrad] = valueAndGrad(loss)(
        tree.ref(params),
        X,
        y
    ) as [np.Array, NetworkParams]

    const lossValue = lossVal.js() as number
    // lossVal.dispose() // Error: Referenced tracer Array:float32[] freed

    // Apply optimizer updates
    const solver = adam(learningRate)
    const [updates, newOptState] = solver.update(lossGrad, optState)
    const newParams = applyUpdates(params, updates) as NetworkParams

    return {
        params: newParams,
        optState: newOptState,
        loss: lossValue,
    }
}

/**
 * Predict class for a batch of points
 */
export function predictBatch(params: NetworkParams, X: np.Array): number[] {
    const preds = predict(tree.ref(params), X.ref)
    const result = preds.js() as number[][]
    // preds.dispose() // Error: Referenced tracer Array:float32[1600,1] freed
    return result.flat()
}

/**
 * Generate decision boundary predictions for visualization grid
 */
export function predictGrid(
    params: NetworkParams,
    resolution: number = 50
): number[][] {
    const grid: number[][] = []

    // Create grid of points [0, 1] x [0, 1]
    const points: number[][] = []
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            // Invert y so i=0 (top row) corresponds to y=1 (max value)
            // This matches the canvas coordinate system where y=0 is top
            // but our data plot uses y=1 as top (1 - point.y)
            points.push([j / (resolution - 1), 1 - i / (resolution - 1)])
        }
    }

    const X = np.array(points)
    const preds = predictBatch(params, X)
    X.dispose()

    // Reshape to grid
    for (let i = 0; i < resolution; i++) {
        grid.push(preds.slice(i * resolution, (i + 1) * resolution))
    }

    return grid
}
