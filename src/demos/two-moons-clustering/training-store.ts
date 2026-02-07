/**
 * Training store for two-moons classification demo
 * Separates training logic from UI components
 */

import { createSignal } from 'solid-js'
import { createStore } from 'solid-js/store'
import { generateTwoMoons, normalizeForVisualization, type DataPoint } from '../../lib/datasets/jax/two-moons'
import {
    initJax,
    initParams,
    initOptimizer,
    trainStep,
    predictGrid,
    type NetworkParams,
    type OptState,
} from './models'
import { numpy as np, tree } from '@jax-js/jax'

export interface TrainingConfig {
    hiddenSize?: number
    gridResolution?: number
}

export interface TrainingState {
    isLoading: boolean
    device: string
    points: DataPoint[]
    params: NetworkParams | null
    decisionGrid: number[][] | null
    epoch: number
    loss: number | null
    isTraining: boolean
}

export interface HyperParams {
    learningRate: number
    optimizerType: 'adam' | 'sgd'
    epochsPerRound: number
    nSamples: number
    noise: number
    batchSize: number
}

export function createTrainingStore(config: TrainingConfig = {}) {
    const { hiddenSize = 16, gridResolution = 40 } = config

    // Reactive state
    const [state, setState] = createStore<TrainingState>({
        isLoading: true,
        device: '',
        points: [],
        params: null,
        decisionGrid: null,
        epoch: 0,
        loss: null,
        isTraining: false,
    })

    // Hyperparameters as signals for UI binding
    const [learningRate, setLearningRate] = createSignal(0.01)
    const [optimizerType, setOptimizerType] = createSignal<'adam' | 'sgd'>('adam')
    const [epochsPerRound, setEpochsPerRound] = createSignal(500)
    const [nSamples, setNSamples] = createSignal(100)
    const [noise, setNoise] = createSignal(0.15)
    const [batchSize, setBatchSize] = createSignal(200)

    // Non-reactive training data (internal)
    let X: np.Array | null = null
    let y: np.Array | null = null
    let trainingX: number[][] = []
    let trainingY: number[] = []
    let optState: OptState | null = null

    /**
     * Initialize JAX runtime
     */
    async function initialize(): Promise<string> {
        const dev = await initJax()
        setState({ isLoading: false, device: dev })
        generateData()
        return dev
    }

    /**
     * Generate new dataset and reinitialize network
     */
    function generateData() {
        // Cleanup old data
        X?.dispose()
        y?.dispose()
        tree.dispose(state.params)

        const dataset = normalizeForVisualization(
            generateTwoMoons(nSamples(), noise())
        )

        // Store raw normalized data for training
        trainingX = dataset.X
        trainingY = dataset.y

        // Create jax arrays for training
        X = np.array(dataset.X)
        y = np.array(dataset.y.map(v => [v]))

        // Initialize new network
        const newParams = initParams(hiddenSize)
        optState = initOptimizer(newParams, learningRate(), optimizerType())

        setState({
            points: dataset.points,
            params: newParams,
            epoch: 0,
            loss: null,
            decisionGrid: null,
        })
    }

    /**
     * Reinitialize optimizer (e.g., when switching optimizer type)
     */
    function reinitOptimizer() {
        if (state.params) {
            optState = initOptimizer(state.params, learningRate(), optimizerType())
        }
    }

    /**
     * Run training for configured number of epochs
     */
    async function runTraining(): Promise<void> {
        if (!state.params || !optState || state.isTraining) return

        setState({ isTraining: true })

        // Clamp batch size if needed
        const maxBatch = nSamples() * 2
        if (batchSize() > maxBatch) {
            setBatchSize(maxBatch)
        }

        const totalEpochs = epochsPerRound()
        const bs = batchSize()
        const n = trainingX.length

        for (let i = 0; i < totalEpochs; i++) {
            // Shuffle indices
            const indices = Array.from({ length: n }, (_, k) => k)
            for (let k = n - 1; k > 0; k--) {
                const j = Math.floor(Math.random() * (k + 1))
                const temp = indices[k]
                indices[k] = indices[j]
                indices[j] = temp
            }

            let epochTotalLoss = 0
            let epochSamples = 0

            // Mini-batch training
            for (let j = 0; j < n; j += bs) {
                const currentBatchSize = Math.min(bs, n - j)

                let X_batch: np.Array
                let y_batch: np.Array

                // Optimization: Use pre-loaded full dataset if doing full batch
                if (bs >= n && X && y) {
                    X_batch = X.ref
                    y_batch = y.ref
                } else {
                    const batchIdx = indices.slice(j, j + bs)
                    X_batch = np.array(batchIdx.map(idx => trainingX[idx]))
                    y_batch = np.array(batchIdx.map(idx => [trainingY[idx]]))
                }

                const result = trainStep(
                    state.params!,
                    optState!,
                    X_batch,
                    y_batch,
                    learningRate(),
                    optimizerType()
                )

                optState = result.optState
                setState({ params: result.params })

                epochTotalLoss += result.loss * currentBatchSize
                epochSamples += currentBatchSize
            }

            setState(s => ({
                loss: epochTotalLoss / epochSamples,
                epoch: s.epoch + 1,
            }))

            // Update decision boundary periodically
            if ((i + 1) % 10 === 0 || i === totalEpochs - 1) {
                const grid = predictGrid(state.params!, gridResolution)
                setState({ decisionGrid: grid })
            }

            // Allow UI to update
            await new Promise(r => setTimeout(r, 0))
        }

        setState({ isTraining: false })
    }

    return {
        // State
        state,

        // Hyperparameter signals
        learningRate,
        setLearningRate,
        optimizerType,
        setOptimizerType,
        epochsPerRound,
        setEpochsPerRound,
        nSamples,
        setNSamples,
        noise,
        setNoise,
        batchSize,
        setBatchSize,

        // Actions
        initialize,
        generateData,
        reinitOptimizer,
        runTraining,
    }
}

export type TrainingStore = ReturnType<typeof createTrainingStore>
