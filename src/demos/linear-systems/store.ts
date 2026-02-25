import { batch } from 'solid-js'
import { createStore, reconcile } from 'solid-js/store'
import { generateMatrix, computeSparsity, stepSimulationJit, type MatrixType } from './math-utils'
import { init, defaultDevice, numpy as np } from '@jax-js/jax'

export type StateNormType = 'none' | 'clip' | 'step-minmax' | 'ema-layernorm' | 'softmax' | 'tanh'

export interface AppState {
    N: number
    globalMin: number
    globalMax: number
    W: number[][]
    x0: number[]
    history: number[][]
    sparsity: number
    isPlaying: boolean
    isLoading: boolean
    device: string
    matrixType: MatrixType
    stateNorm: StateNormType
    emaMean: number
    emaVar: number
}

export function createSimulationStore() {
    const defaultN = 6
    const defaultMin = -2
    const defaultMax = 2

    const [state, setState] = createStore<AppState>({
        N: defaultN,
        globalMin: defaultMin,
        globalMax: defaultMax,
        W: [],
        x0: [],
        history: [],
        sparsity: 0,
        isPlaying: false,
        isLoading: true,
        device: 'Loading...',
        matrixType: 'lattice',
        stateNorm: 'none',
        emaMean: 0,
        emaVar: 1
    })

    // Previously we recomputed the entire history. Now normalization affects the *actual* state evolution,
    // so we can't retroactively apply it to history. History is what actually happened.
    const resetEmats = () => {
        setState('emaMean', 0)
        setState('emaVar', 1)
    }

    const initialize = async () => {
        const devices = await init()
        const selectedDevice = devices.includes('webgpu') ? 'webgpu' : devices[0]
        if (devices.includes('webgpu')) defaultDevice('webgpu')

        // Generate initial vectors
        const W = generateMatrix('lattice', defaultN, defaultMin, defaultMax)
        const x0 = Array(defaultN).fill(1).map(() => Math.random() * (defaultMax - defaultMin) + defaultMin)
        const sparsity = computeSparsity(W)

        setState({
            isLoading: false,
            device: selectedDevice,
            W: W,
            x0: [...x0],
            history: [[...x0]],
            sparsity: sparsity,
        })
    }

    const setN = (n: number) => {
        if (n === state.N) return
        batch(() => {
            setState('N', n)
            // Resize W and x0
            const W = generateMatrix(state.matrixType, n, state.globalMin, state.globalMax)
            const x0 = Array(n).fill(1).map(() => Math.random() * (state.globalMax - state.globalMin) + state.globalMin)
            const sparsity = computeSparsity(W)

            setState({ W, x0, history: [[...x0]], sparsity, isPlaying: false })
        })
    }

    const setMatrixType = (type: MatrixType) => {
        batch(() => {
            setState('matrixType', type)
            const W = generateMatrix(type, state.N, state.globalMin, state.globalMax)
            const sparsity = computeSparsity(W)
            setState({ W, sparsity, isPlaying: false, history: [[...state.x0]] })
        })
    }

    const setMatrixValue = (r: number, c: number, value: number) => {
        setState('W', r, c, Math.min(Math.max(value, state.globalMin), state.globalMax))
        setState('sparsity', computeSparsity(state.W))
    }

    const setInitialStateValue = (idx: number, value: number) => {
        setState('x0', idx, Math.min(Math.max(value, state.globalMin), state.globalMax))
    }

    const randomizeInitialState = () => {
        const newX0 = Array(state.N).fill(1).map(() => Math.random() * (state.globalMax - state.globalMin) + state.globalMin)
        batch(() => {
            setState('x0', newX0)
            if (!state.isPlaying) {
                resetEmats()
                setState('history', reconcile([newX0]))
            }
        })
    }

    const resetSimulation = () => {
        resetEmats()
        setState('history', reconcile([[...state.x0]]))
        setState('isPlaying', false)
    }

    let rafId: number | null = null

    const simulationLoop = async () => {
        if (!state.isPlaying || state.isLoading) return

        const W_arr = np.array(state.W)
        const x_current = np.array(state.history[state.history.length - 1])

        try {
            const next_x = stepSimulationJit(x_current, W_arr)
            // get js value async
            const result = await next_x.jsAsync() as number[]

            // Note: We are NO LONGER limiting bounds to globalMax/globalMin for the state evolution,
            // as requested by the user. Let it blow up or go to zero as linear systems normally do!
            // Just bound it to Javascript Infinity threshold to avoid NaN crashes
            const boundedResult = result.map(v => Math.max(Number.MIN_SAFE_INTEGER, Math.min(Number.MAX_SAFE_INTEGER, v)))

            const norm = state.stateNorm
            const maxAbs = Math.max(Math.abs(state.globalMin), Math.abs(state.globalMax), 0.1)
            let normStep: number[] = []

            if (norm === 'none') {
                normStep = boundedResult
            } else if (norm === 'clip') {
                normStep = boundedResult.map(v => Math.max(state.globalMin, Math.min(state.globalMax, v)))
            } else if (norm === 'step-minmax') {
                let min = Infinity, max = -Infinity
                for (const v of boundedResult) {
                    if (v < min) min = v
                    if (v > max) max = v
                }
                if (max === min) normStep = boundedResult.map(() => 0)
                else normStep = boundedResult.map(v => (((v - min) / (max - min)) * (state.globalMax - state.globalMin)) + state.globalMin)
            } else if (norm === 'ema-layernorm') {
                let mean = 0
                for (const v of boundedResult) mean += v
                mean /= boundedResult.length

                let variance = 0
                for (const v of boundedResult) variance += (v - mean) ** 2
                variance /= boundedResult.length

                setState('emaMean', m => 0.9 * m + 0.1 * mean)
                setState('emaVar', v => 0.9 * v + 0.1 * variance)

                const std = Math.sqrt(state.emaVar) || 1
                // Scale normalized value by maxAbs/3 so it fits roughly within global bounds
                const scale = maxAbs / 3
                normStep = boundedResult.map(v => ((v - state.emaMean) / std) * scale)
            } else if (norm === 'softmax') {
                const maxVal = Math.max(...boundedResult)
                const exps = boundedResult.map(v => Math.exp(v - maxVal))
                const sumExps = exps.reduce((acc, val) => acc + val, 0)
                normStep = exps.map(v => v / sumExps)
            } else if (norm === 'tanh') {
                normStep = boundedResult.map(v => Math.tanh(v))
            }

            setState('history', h => [...h, normStep])
        } catch (e) {
            console.error('Simulation error', e)
            setState('isPlaying', false)
            return
        }

        // Throttle simulation to roughly 10 steps per second
        setTimeout(() => {
            if (state.isPlaying) {
                rafId = requestAnimationFrame(simulationLoop)
            }
        }, 100)
    }

    const togglePlaying = () => {
        const nextState = !state.isPlaying
        setState('isPlaying', nextState)
        if (nextState) {
            rafId = requestAnimationFrame(simulationLoop)
        } else if (rafId) {
            cancelAnimationFrame(rafId)
            rafId = null
        }
    }

    const setGlobalBounds = (newMin: number, newMax: number) => {
        batch(() => {
            setState({ globalMin: newMin, globalMax: newMax })
        })
    }

    const setStateNorm = (norm: StateNormType) => {
        batch(() => {
            setState('stateNorm', norm)
            if (!state.isPlaying) resetSimulation()
        })
    }

    return {
        state,
        initialize,
        setN,
        setMatrixType,
        setMatrixValue,
        setInitialStateValue,
        randomizeInitialState,
        resetSimulation,
        togglePlaying,
        setGlobalBounds,
        setStateNorm
    }
}

export const simulationStore = createSimulationStore()
