import { valueAndGrad, tree, type numpy as np } from '@jax-js/jax'
import {
    createOptimizer,
    applyOptimizerUpdates,
    type OptimizerSolver,
    type OptimizerType,
    type OptState,
} from './optimization'

export async function trainStepWithSolver<P extends Record<string, np.Array>>(
    params: P,
    optState: OptState,
    solver: OptimizerSolver,
    X: np.Array,
    y: np.Array,
    lossFn: (params: P, X: np.Array, y: np.Array) => np.Array
): Promise<{ params: P; optState: OptState; loss: number }> {
    // @ts-expect-error — P satisfies MappedJsTree<P, Array, ArrayLike> but TS cannot verify conditional types on generics
    const [lossVal, lossGrad] = valueAndGrad(lossFn)(tree.ref(params), X, y) as [np.Array, P]
    const loss = await lossVal.jsAsync() as number
    const [updates, nextOptState] = solver.update(lossGrad, optState, tree.ref(params))
    const nextParams = applyOptimizerUpdates(params, updates) as P
    return {
        params: nextParams,
        optState: nextOptState,
        loss,
    }
}

/**
 * Stateless helper for one-off optimization steps.
 * This re-creates the solver each call and is less efficient than `trainStepWithSolver`.
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
    const solver = createOptimizer(learningRate, optimizerType)
    return trainStepWithSolver(params, optState, solver, X, y, lossFn)
}
