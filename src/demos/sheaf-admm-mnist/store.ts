import { createSignal } from 'solid-js'
import { createStore } from 'solid-js/store'
import { fetchMnistRaw, type MnistFetchProgress } from '../../lib/datasources/mnist'
import { formatMnistForMlp, getImagePixels, type MnistMlpDataset } from '../../lib/datasets/jax/mnist'
import {
    buildMnistPrototypes,
    createSheafAdmmSession,
    runSheafAdmm,
    stepSheafAdmm,
    type MnistClassPrototypes,
    type PatchAgent,
    type SheafAdmmSession,
    type SheafAdmmSnapshot,
} from './algorithm'

export interface ResidualPoint {
    iteration: number
    primal: number
    consensus: number
}

export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface SheafAdmmStoreState {
    loadStatus: LoadStatus
    loadProgress: MnistFetchProgress | null
    error: string | null
    currentLabel: number | null
    currentPixels: Float32Array
    history: SheafAdmmSnapshot[]
    isRunning: boolean
}

export interface SheafAdmmStore {
    state: SheafAdmmStoreState
    sampleIndex: () => number
    setSampleIndex: (index: number) => void
    maxIterations: () => number
    setMaxIterations: (iterations: number) => void
    diffusionSteps: () => number
    setDiffusionSteps: (steps: number) => void
    rho: () => number
    setRho: (rho: number) => void
    gamma: () => number
    setGamma: (gamma: number) => void
    runDelayMs: () => number
    setRunDelayMs: (delay: number) => void
    datasetCount: () => number
    agents: () => PatchAgent[]
    latest: () => SheafAdmmSnapshot | null
    residualHistory: () => ResidualPoint[]
    initialize: () => Promise<void>
    stepOnce: () => void
    runToLimit: () => void
    stop: () => void
    resetCurrent: () => void
    pickNextMistake: () => void
}

const EMPTY_PIXELS = new Float32Array(28 * 28)
const PROTOTYPE_TRAIN_LIMIT = 12_000

export function createSheafAdmmMnistStore(): SheafAdmmStore {
    const [sampleIndex, setSampleIndexSignal] = createSignal(0)
    const [maxIterations, setMaxIterations] = createSignal(16)
    const [diffusionSteps, setDiffusionSteps] = createSignal(5)
    const [rho, setRho] = createSignal(1.2)
    const [gamma, setGamma] = createSignal(0.07)
    const [runDelayMs, setRunDelayMs] = createSignal(240)
    const [datasetCount, setDatasetCount] = createSignal(0)

    const [state, setState] = createStore<SheafAdmmStoreState>({
        loadStatus: 'idle',
        loadProgress: null,
        error: null,
        currentLabel: null,
        currentPixels: EMPTY_PIXELS,
        history: [],
        isRunning: false,
    })

    let dataset: MnistMlpDataset | null = null
    let prototypes: MnistClassPrototypes | null = null
    let session: SheafAdmmSession | null = null
    let timer: ReturnType<typeof setTimeout> | null = null

    function latest(): SheafAdmmSnapshot | null {
        return state.history[state.history.length - 1] ?? null
    }

    function agents(): PatchAgent[] {
        return session?.agents ?? []
    }

    function params() {
        return {
            rho: rho(),
            gamma: gamma(),
            diffusionSteps: diffusionSteps(),
        }
    }

    function stop() {
        if (timer) {
            clearTimeout(timer)
            timer = null
        }
        setState({ isRunning: false })
    }

    function loadSample(index: number) {
        if (!dataset || !prototypes) return
        stop()
        const boundedIndex = Math.max(0, Math.min(dataset.test.count - 1, Math.round(index)))
        setSampleIndexSignal(boundedIndex)
        const pixels = getImagePixels(dataset.test, boundedIndex)
        session = createSheafAdmmSession(pixels, prototypes)
        setState({
            currentLabel: dataset.test.labels[boundedIndex],
            currentPixels: pixels,
            history: [session.initial],
            error: null,
        })
    }

    async function initialize() {
        if (state.loadStatus === 'loading' || state.loadStatus === 'ready') return
        setState({ loadStatus: 'loading', loadProgress: null, error: null })

        try {
            const raw = await fetchMnistRaw({
                onProgress: progress => setState({ loadProgress: progress }),
            })
            dataset = formatMnistForMlp(raw)
            prototypes = buildMnistPrototypes(dataset.train.images, dataset.train.labels, PROTOTYPE_TRAIN_LIMIT)
            setDatasetCount(dataset.test.count)
            setState({ loadStatus: 'ready', loadProgress: null })
            loadSample(sampleIndex())
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to load MNIST.'
            setState({ loadStatus: 'error', loadProgress: null, error: message })
        }
    }

    function resetCurrent() {
        if (!dataset || !prototypes) return
        loadSample(sampleIndex())
    }

    function stepOnce() {
        if (!session) return
        const current = latest()
        if (!current || current.iteration >= maxIterations()) return
        const next = stepSheafAdmm(session.agents, current, params())
        setState('history', history => [...history, next])
    }

    function scheduleRun() {
        timer = setTimeout(() => {
            timer = null
            const current = latest()
            if (!session || !current || current.iteration >= maxIterations()) {
                setState({ isRunning: false })
                return
            }
            stepOnce()
            scheduleRun()
        }, runDelayMs())
    }

    function runToLimit() {
        if (!session) return
        stop()
        const current = latest()
        if (!current) return
        if (runDelayMs() <= 20) {
            const remaining = Math.max(0, maxIterations() - current.iteration)
            const appended = []
            let cursor = current
            for (let i = 0; i < remaining; i++) {
                cursor = stepSheafAdmm(session.agents, cursor, params())
                appended.push(cursor)
            }
            setState('history', history => [...history, ...appended])
            return
        }
        setState({ isRunning: true })
        scheduleRun()
    }

    function pickNextMistake() {
        if (!dataset) return
        stop()
        const count = dataset.test.count
        let nextIndex = sampleIndex()
        for (let offset = 1; offset < Math.min(count, 600); offset++) {
            const candidate = (sampleIndex() + offset) % count
            const pixels = getImagePixels(dataset.test, candidate)
            const candidateSession = createSheafAdmmSession(pixels, prototypes)
            const candidateHistory = runSheafAdmm(candidateSession, params(), Math.min(6, maxIterations()))
            const prediction = candidateHistory[candidateHistory.length - 1].prediction
            if (prediction !== dataset.test.labels[candidate]) {
                nextIndex = candidate
                break
            }
        }
        loadSample(nextIndex)
        if (session) {
            const preview = runSheafAdmm(session, params(), Math.min(6, maxIterations())).slice(1)
            setState('history', history => [...history, ...preview])
        }
    }

    function setSampleIndex(index: number) {
        loadSample(index)
    }

    function residualHistory(): ResidualPoint[] {
        return state.history.map(snapshot => ({
            iteration: snapshot.iteration,
            primal: snapshot.primalResidual,
            consensus: snapshot.consensusResidual,
        }))
    }

    return {
        state,
        sampleIndex,
        setSampleIndex,
        maxIterations,
        setMaxIterations: value => setMaxIterations(Math.round(value)),
        diffusionSteps,
        setDiffusionSteps: value => setDiffusionSteps(Math.round(value)),
        rho,
        setRho,
        gamma,
        setGamma,
        runDelayMs,
        setRunDelayMs: value => setRunDelayMs(Math.round(value)),
        datasetCount,
        agents,
        latest,
        residualHistory,
        initialize,
        stepOnce,
        runToLimit,
        stop,
        resetCurrent,
        pickNextMistake,
    }
}
