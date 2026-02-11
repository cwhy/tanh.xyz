/**
 * Reusable deep learning utilities for jax-js
 */

import { numpy as np, init, defaultDevice, valueAndGrad, tree } from '@jax-js/jax'
import { adam, sgd, applyUpdates, type OptState } from '@jax-js/optax'

export type { OptState }

export interface TrainingState<P> {
    params: P
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
 * Initialize optimizer state for any parameter structure
 */
export function initOptimizer<P extends Record<string, np.Array>>(
    params: P,
    learningRate: number,
    optimizerType: 'adam' | 'sgd' = 'adam'
): OptState {
    const solver = optimizerType === 'adam' ? adam(learningRate) : sgd(learningRate)
    return solver.init(tree.ref(params))
}

/**
 * Generic training step that works with any loss function
 */
export async function trainStep<P extends Record<string, np.Array>>(
    params: P,
    optState: OptState,
    X: np.Array,
    y: np.Array,
    lossFn: (params: P, X: np.Array, y: np.Array) => np.Array,
    learningRate: number,
    optimizerType: 'adam' | 'sgd' = 'adam'
): Promise<{ params: P; optState: OptState; loss: number }> {
    // Compute loss and gradients using valueAndGrad
    // @ts-expect-error — P satisfies MappedJsTree<P, Array, ArrayLike> but TS can't verify conditional types on generics
    const [lossVal, lossGrad] = valueAndGrad(lossFn)(tree.ref(params), X, y) as [np.Array, P]

    const lossValue = await lossVal.jsAsync() as number

    // Apply optimizer updates
    const solver = optimizerType === 'adam' ? adam(learningRate) : sgd(learningRate)
    const [updates, newOptState] = solver.update(lossGrad, optState)
    const newParams = applyUpdates(params, updates) as P

    return {
        params: newParams,
        optState: newOptState,
        loss: lossValue,
    }
}
