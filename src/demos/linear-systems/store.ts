import { createContext, createStore, useContext } from 'solid-js'
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

    let disposed = false

    const initialize = async () => {
        const devices = await init()
        if (disposed) return
        const selectedDevice = devices.includes('webgpu') ? 'webgpu' : devices[0]
        if (devices.includes('webgpu')) defaultDevice('webgpu')

        // Generate initial vectors
        const W = generateMatrix('lattice', defaultN, defaultMin, defaultMax)
        const x0 = Array(defaultN).fill(1).map(() => Math.random() * (defaultMax - defaultMin) + defaultMin)
        const sparsity = computeSparsity(W)

        setState(s => {
            s.isLoading = false
            s.device = selectedDevice
            s.W = W
            s.x0 = [...x0]
            s.history = [[...x0]]
            s.sparsity = sparsity
        })
    }

    const setN = (n: number) => {
        setState(s => {
            if (n === s.N) return
            s.N = n
            s.W = generateMatrix(s.matrixType, n, s.globalMin, s.globalMax)
            s.x0 = Array.from({ length: n }, () => Math.random() * (s.globalMax - s.globalMin) + s.globalMin)
            s.history = [[...s.x0]]
            s.sparsity = computeSparsity(s.W)
            s.isPlaying = false
        })
    }

    const setMatrixType = (type: MatrixType) => {
        setState(s => {
            s.matrixType = type
            s.W = generateMatrix(type, s.N, s.globalMin, s.globalMax)
            s.sparsity = computeSparsity(s.W)
            s.isPlaying = false
            s.history = [[...s.x0]]
        })
    }

    const setMatrixValue = (r: number, c: number, value: number) => {
        setState(s => {
            s.W[r][c] = Math.min(Math.max(value, s.globalMin), s.globalMax)
            s.sparsity = computeSparsity(s.W)
        })
    }

    const setInitialStateValue = (idx: number, value: number) => {
        setState(s => { s.x0[idx] = Math.min(Math.max(value, s.globalMin), s.globalMax) })
    }

    const randomizeInitialState = () => {
        setState(s => {
            s.x0 = Array.from({ length: s.N }, () => Math.random() * (s.globalMax - s.globalMin) + s.globalMin)
            if (!s.isPlaying) {
                s.emaMean = 0
                s.emaVar = 1
                s.history = [[...s.x0]]
            }
        })
    }

    const resetSimulation = () => {
        setState(s => {
            s.emaMean = 0
            s.emaVar = 1
            s.history = [[...s.x0]]
            s.isPlaying = false
        })
    }

    let rafId: number | null = null
    let timer: ReturnType<typeof setTimeout> | null = null

    const dispose = () => {
        disposed = true
        if (rafId !== null) cancelAnimationFrame(rafId)
        if (timer !== null) clearTimeout(timer)
        rafId = null
        timer = null
    }

    const simulationLoop = async () => {
        if (disposed || !state.isPlaying || state.isLoading) return

        const W_arr = np.array(state.W)
        const x_current = np.array(state.history[state.history.length - 1])

        try {
            const next_x = stepSimulationJit(x_current, W_arr)
            // get js value async
            const result = await next_x.jsAsync() as number[]
            if (disposed || !state.isPlaying) return

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

                const emaMean = 0.9 * state.emaMean + 0.1 * mean
                const emaVar = 0.9 * state.emaVar + 0.1 * variance
                setState(s => { s.emaMean = emaMean; s.emaVar = emaVar })

                const std = Math.sqrt(emaVar) || 1
                // Scale normalized value by maxAbs/3 so it fits roughly within global bounds
                const scale = maxAbs / 3
                normStep = boundedResult.map(v => ((v - emaMean) / std) * scale)
            } else if (norm === 'softmax') {
                const maxVal = Math.max(...boundedResult)
                const exps = boundedResult.map(v => Math.exp(v - maxVal))
                const sumExps = exps.reduce((acc, val) => acc + val, 0)
                normStep = exps.map(v => v / sumExps)
            } else if (norm === 'tanh') {
                normStep = boundedResult.map(v => Math.tanh(v))
            }

            setState(s => { s.history.push(normStep) })
        } catch (e) {
            console.error('Simulation error', e)
            setState(s => { s.isPlaying = false })
            return
        }

        // Throttle simulation to roughly 10 steps per second
        timer = setTimeout(() => {
            timer = null
            if (!disposed && state.isPlaying) {
                rafId = requestAnimationFrame(simulationLoop)
            }
        }, 100)
    }

    const togglePlaying = () => {
        const nextState = !state.isPlaying
        setState(s => { s.isPlaying = nextState })
        if (nextState) {
            rafId = requestAnimationFrame(simulationLoop)
        } else if (rafId) {
            cancelAnimationFrame(rafId)
            rafId = null
        }
    }

    const setGlobalBounds = (newMin: number, newMax: number) => {
        setState(s => {
            s.globalMin = newMin
            s.globalMax = newMax
        })
    }

    const setStateNorm = (norm: StateNormType) => {
        setState(s => { s.stateNorm = norm })
        if (!state.isPlaying) resetSimulation()
    }

    return {
        state,
        initialize,
        dispose,
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

export const SimulationContext = createContext<ReturnType<typeof createSimulationStore>>()
export const useSimulationStore = () => useContext(SimulationContext)
