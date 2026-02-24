/**
 * Neural network model for two-moons binary classification
 * 2-layer MLP with tanh hidden layer and sigmoid output
 */

import { numpy as np, tree } from '@jax-js/jax'
import {
    initJax,
    initOptimizer,
    trainStep as genericTrainStep,
    type OptState,
} from '../../lib/deeplearning'

// Re-export for convenience
export { initJax, initOptimizer, type OptState }

export interface NetworkParams {
    W1: np.Array  // [2, hiddenSize]
    b1: np.Array  // [hiddenSize]
    W2: np.Array  // [hiddenSize, 1]
    b2: np.Array  // [1]
    [key: string]: np.Array  // Index signature for JsTree compatibility
}

/**
 * Initialize network parameters with He initialization
 */
export function initParams(hiddenSize: number = 16): NetworkParams {
    // He init: sqrt(2/input_dim), multiplied by 2.0 to ensure non-linearity
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
 * Forward pass through the network
 */
export function predict(params: NetworkParams, X: np.Array): np.Array {
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
 */
export function loss(params: NetworkParams, X: np.Array, y: np.Array): np.Array {
    const preds = predict(params, X)

    // BCE = -mean(y * (log(p) - log(1-p)) + log(1-p))
    const logP = np.log(preds.ref.add(1e-7))
    const log1mP = np.log(np.array(1).sub(preds).add(1e-7))

    return y.mul(logP.sub(log1mP.ref)).add(log1mP).mean().neg()
}

/**
 * Single training step using the model's loss function
 */
export async function trainStep(
    params: NetworkParams,
    optState: OptState,
    X: np.Array,
    y: np.Array,
    learningRate: number,
    optimizerType: 'adam' | 'sgd' = 'adam'
): Promise<{ params: NetworkParams; optState: OptState; loss: number }> {
    return genericTrainStep(params, optState, X, y, loss, learningRate, optimizerType)
}

/**
 * Predict class for a batch of points
 */
export async function predictBatch(params: NetworkParams, X: np.Array): Promise<number[]> {
    const preds = predict(tree.ref(params), X.ref)
    const result = await preds.jsAsync() as number[][]
    return result.flat()
}

/**
 * Generate decision boundary predictions for visualization grid
 */
export async function predictGrid(params: NetworkParams, resolution: number = 50): Promise<number[][]> {
    const grid: number[][] = []

    // Create grid of points [0, 1] x [0, 1]
    const points: number[][] = []
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            // Invert y so i=0 (top row) corresponds to y=1 (max value)
            points.push([j / (resolution - 1), 1 - i / (resolution - 1)])
        }
    }

    const X = np.array(points)
    const preds = await predictBatch(params, X)
    X.dispose()

    // Reshape to grid
    for (let i = 0; i < resolution; i++) {
        grid.push(preds.slice(i * resolution, (i + 1) * resolution))
    }

    return grid
}
