import { numpy as np, tree, type JsTree } from '@jax-js/jax'
import {
    adam,
    adamw,
    applyUpdates,
    sgd,
    treeZerosLike,
    type GradientTransformation,
    type OptState,
} from '@jax-js/optax'

export type { OptState }

export type Schedule = (count: number) => number
export type LearningRate = number | Schedule
export type OptimizerType = 'adam' | 'adamw' | 'sgd' | 'rmsprop'
export type OptimizerSolver = GradientTransformation

export interface RmspropOptions {
    decay?: number
    eps?: number
    momentum?: number
}

export interface OptimizerOptions {
    weightDecay?: number
    rmsprop?: RmspropOptions
}

interface RmspropState {
    count: np.Array
    avgSq: JsTree<np.Array>
    momentum: JsTree<np.Array>
}

/**
 * RMSProp optimizer implemented as an Optax-compatible gradient transformation.
 */
export function rmsprop(
    learningRate: LearningRate,
    options: RmspropOptions = {}
): GradientTransformation {
    const decay = options.decay ?? 0.9
    const eps = options.eps ?? 1e-8
    const momentum = options.momentum ?? 0

    return {
        init<Params extends JsTree<np.Array>>(params: Params): OptState {
            return {
                count: np.array(0, { dtype: np.uint32 }),
                avgSq: treeZerosLike(tree.ref(params)),
                momentum: treeZerosLike(params),
            } as unknown as OptState
        },
        update<Params extends JsTree<np.Array>>(
            updates: Params,
            state: OptState,
            params?: Params
        ): [Params, OptState] {
            tree.dispose(params)
            const localState = state as unknown as RmspropState
            const step = localState.count.item()
            const lr = typeof learningRate === 'function' ? learningRate(step) : learningRate

            const avgSq = tree.map(
                (avg, grad) => (
                    (avg as np.Array).mul(decay).add(np.square(grad as np.Array).mul(1 - decay))
                ),
                localState.avgSq as Params,
                tree.ref(updates)
            ) as Params

            const scaled = tree.map(
                (grad, avg) => (
                    (grad as np.Array).div(np.sqrt((avg as np.Array).add(eps))).mul(-lr)
                ),
                updates,
                tree.ref(avgSq)
            ) as Params

            if (momentum > 0) {
                const momentumState = tree.map(
                    (mom, delta) => (mom as np.Array).mul(momentum).add(delta as np.Array),
                    localState.momentum as Params,
                    tree.ref(scaled)
                ) as Params

                return [
                    tree.ref(momentumState) as Params,
                    {
                        count: localState.count.add(1),
                        avgSq,
                        momentum: momentumState,
                    } as unknown as OptState,
                ]
            }

            return [
                scaled,
                {
                    count: localState.count.add(1),
                    avgSq,
                    momentum: localState.momentum,
                } as unknown as OptState,
            ]
        },
    }
}

export function createOptimizer(
    learningRate: LearningRate,
    optimizerType: OptimizerType = 'adam',
    options: OptimizerOptions = {}
): OptimizerSolver {
    if (optimizerType === 'sgd') return sgd(learningRate)
    if (optimizerType === 'rmsprop') return rmsprop(learningRate, options.rmsprop)
    if (optimizerType === 'adamw') return adamw(learningRate, { weightDecay: options.weightDecay ?? 0.01 })
    return adam(learningRate)
}

export function createOptimizerState<P extends JsTree<np.Array>>(
    params: P,
    learningRate: LearningRate,
    optimizerType: OptimizerType = 'adam',
    options: OptimizerOptions = {}
): { solver: OptimizerSolver; optState: OptState } {
    const solver = createOptimizer(learningRate, optimizerType, options)
    const optState = solver.init(tree.ref(params))
    return { solver, optState }
}

export function initOptimizer<P extends JsTree<np.Array>>(
    params: P,
    learningRate: LearningRate,
    optimizerType: OptimizerType = 'adam',
    options: OptimizerOptions = {}
): OptState {
    return createOptimizer(learningRate, optimizerType, options).init(tree.ref(params))
}

export function applyOptimizerUpdates<P extends JsTree<np.Array>>(params: P, updates: P): P {
    return applyUpdates(params, updates) as P
}
