/**
 * Reusable deep learning utilities for jax-js
 */

import { tree, type numpy as np } from '@jax-js/jax'
import { type OptState } from '@jax-js/optax'
import {
    createOptimizer,
    trainStep as trainStepGeneric,
    type OptimizerType,
} from '../deeplearning'

export type { OptState }

export interface TrainingState<P> {
    params: P
    optState: OptState
    epoch: number
    loss: number
}

/**
 * Initialize optimizer state for any parameter structure
 */
export function initOptimizer<P extends Record<string, np.Array>>(
    params: P,
    learningRate: number,
    optimizerType: OptimizerType = 'adam'
): OptState {
    const solver = createOptimizer(learningRate, optimizerType)
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
    optimizerType: OptimizerType = 'adam'
): Promise<{ params: P; optState: OptState; loss: number }> {
    return trainStepGeneric(params, optState, X, y, lossFn, learningRate, optimizerType)
}
export * from '../deeplearning'
