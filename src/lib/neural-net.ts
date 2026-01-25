/**
 * Simple neural network for binary classification using jax-js
 * Implements a 2-layer MLP with sigmoid activation
 */

import { numpy as np, grad, jit, init, defaultDevice } from '@jax-js/jax'

export interface NetworkParams {
    W1: np.Array  // [2, hiddenSize]
    b1: np.Array  // [hiddenSize]
    W2: np.Array  // [hiddenSize, 1]
    b2: np.Array  // [1]
}

export interface TrainingState {
    params: NetworkParams
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
    const scale1 = Math.sqrt(2 / (2 + hiddenSize))
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
 * Sigmoid activation function
 */
function sigmoid(x: np.Array): np.Array {
    return np.exp(x.ref.neg()).add(1).reciprocal()
}

/**
 * Forward pass through the network
 */
export const forward = jit(function forward(
    params: NetworkParams,
    X: np.Array
): np.Array {
    // Hidden layer: relu(X @ W1 + b1)
    const h1 = np.maximum(
        np.matmul(X.ref, params.W1.ref).add(params.b1.ref),
        np.array(0)
    )

    // Output layer: sigmoid(h1 @ W2 + b2)
    const logits = np.matmul(h1, params.W2.ref).add(params.b2.ref)
    return sigmoid(logits)
})

/**
 * Binary cross-entropy loss
 */
export function loss(params: NetworkParams, X: np.Array, y: np.Array): np.Array {
    const preds = forward({ ...params, W1: params.W1.ref, b1: params.b1.ref, W2: params.W2.ref, b2: params.b2.ref }, X.ref)
    const eps = 1e-7

    // -mean(y * log(p) + (1-y) * log(1-p))
    const logP = np.log(preds.ref.add(eps))
    const log1mP = np.log(np.array(1).sub(preds).add(eps))

    const bce = y.ref.mul(logP).add(
        np.array(1).sub(y.ref).mul(log1mP)
    ).neg().mean()

    return bce
}

/**
 * Compute gradients using automatic differentiation
 */
export const computeGrads = grad(loss)

/**
 * Single training step with SGD
 */
export function trainStep(
    params: NetworkParams,
    X: np.Array,
    y: np.Array,
    learningRate: number
): { params: NetworkParams; loss: number } {
    // Compute loss for logging
    const currentLoss = loss(
        { W1: params.W1.ref, b1: params.b1.ref, W2: params.W2.ref, b2: params.b2.ref },
        X.ref,
        y.ref
    )
    const lossValue = currentLoss.js() as number
    currentLoss.dispose()

    // Compute gradients
    const grads = computeGrads(
        { W1: params.W1.ref, b1: params.b1.ref, W2: params.W2.ref, b2: params.b2.ref },
        X.ref,
        y.ref
    ) as NetworkParams

    // SGD update: param = param - lr * grad
    const lr = np.array(learningRate)

    const newW1 = params.W1.sub(grads.W1.mul(lr.ref))
    const newB1 = params.b1.sub(grads.b1.mul(lr.ref))
    const newW2 = params.W2.sub(grads.W2.mul(lr.ref))
    const newB2 = params.b2.sub(grads.b2.mul(lr.ref))

    lr.dispose()

    return {
        params: { W1: newW1, b1: newB1, W2: newW2, b2: newB2 },
        loss: lossValue,
    }
}

/**
 * Predict class for a batch of points
 */
export function predict(params: NetworkParams, X: np.Array): number[] {
    const preds = forward(
        { W1: params.W1.ref, b1: params.b1.ref, W2: params.W2.ref, b2: params.b2.ref },
        X.ref
    )
    const result = preds.js() as number[][]
    preds.dispose()
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
            points.push([j / (resolution - 1), i / (resolution - 1)])
        }
    }

    const X = np.array(points)
    const preds = predict(params, X)
    X.dispose()

    // Reshape to grid
    for (let i = 0; i < resolution; i++) {
        grid.push(preds.slice(i * resolution, (i + 1) * resolution))
    }

    return grid
}
