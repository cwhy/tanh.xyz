import { createSignal } from 'solid-js'
import { createStore, produce } from 'solid-js/store'
import { numpy as np, tree } from '@jax-js/jax'
import { fetchMnistRaw, type MnistFetchProgress } from '../../lib/datasources/mnist'
import { formatMnistForMlp, type MnistMlpDataset } from '../../lib/datasets/jax/mnist'
import {
    initJax,
    createSeededRng,
    initDamParams,
    createOptimizer,
    trainStep,
    recallStep,
    extractBasicMemory,
    addNoise,
    mse,
    type DamParams,
    type OptState,
} from './models'

export interface RecallFrame {
    visible: Float32Array
    hBinary: Float32Array
    hActivated: Float32Array
    h: Float32Array
    mseToTarget: number
    step: number
}

export interface DamDemoState {
    // Loading
    isLoading: boolean
    loadingMessage: string
    loadingProgress: number
    device: string
    error: string | null

    // Training
    isTraining: boolean
    trainLosses: number[]
    trainEpochsDone: number
    learnedTheta: number

    // Stored memories metadata
    numMemoriesStored: number
    memoryLabels: number[]

    // Basic memories (columns of ξ)
    basicMemoryImages: Float32Array[] // Nh arrays of length Nv

    // Recall
    isRecalling: boolean
    recallFrames: RecallFrame[]
    currentRecallStep: number
    selectedCueIndex: number
    cueImage: Float32Array | null
    noisyCueImage: Float32Array | null
    originalImage: Float32Array | null
}

export function createDamStore() {
    const [state, setState] = createStore<DamDemoState>({
        isLoading: true,
        loadingMessage: 'Initializing jax-js...',
        loadingProgress: 0,
        device: '',
        error: null,
        isTraining: false,
        trainLosses: [],
        trainEpochsDone: 0,
        learnedTheta: 0.5,
        numMemoriesStored: 0,
        memoryLabels: [],
        basicMemoryImages: [],
        isRecalling: false,
        recallFrames: [],
        currentRecallStep: 0,
        selectedCueIndex: 0,
        cueImage: null,
        noisyCueImage: null,
        originalImage: null,
    })

    // Paper defaults: Nh=50, M=60000 for MNIST, θ learned ≈ 0.21
    const [numHidden, setNumHidden] = createSignal(50)
    const [numMemories, setNumMemories] = createSignal(60000)
    const [trainEpochs, setTrainEpochs] = createSignal(400)
    const [learningRate, setLearningRate] = createSignal(0.005)
    const [noiseLevel, setNoiseLevel] = createSignal(0.5)
    const [tauRatio, setTauRatio] = createSignal(10)
    const [recallSteps, setRecallSteps] = createSignal(30)
    const [seed, setSeed] = createSignal(42)
    // Recall threshold — null = use learned θ (what the paper does)
    const [recallTheta, setRecallTheta] = createSignal<number | null>(null)
    // Recall activation steepness — 0=hard Θ, 20=sharp sigmoid matching training
    const [recallBeta, setRecallBeta] = createSignal(20)

    const Nv = 784

    let dataset: MnistMlpDataset | null = null
    let trainParams: DamParams | null = null
    let xiData: Float32Array | null = null
    let memoryImages: Float32Array | null = null
    let stopRequested = false

    async function initialize(): Promise<void> {
        try {
            setState({
                isLoading: true,
                loadingMessage: 'Initializing jax-js...',
                loadingProgress: 0,
                error: null,
            })
            const dev = await initJax()
            setState({ device: dev, loadingMessage: 'Loading MNIST...' })

            const raw = await fetchMnistRaw({
                onProgress(p: MnistFetchProgress) {
                    setState({
                        loadingMessage: `${p.stage} ${p.file} (${p.completedFiles}/${p.totalFiles})`,
                        loadingProgress: p.totalFiles > 0 ? p.completedFiles / p.totalFiles : 0,
                    })
                },
            })
            dataset = formatMnistForMlp(raw)
            setState({ isLoading: false, loadingMessage: 'Ready', loadingProgress: 1 })
        } catch (err) {
            setState({
                isLoading: false,
                error: err instanceof Error ? err.message : 'Failed to initialize',
            })
        }
    }

    async function startTraining(): Promise<void> {
        if (state.isTraining || state.isLoading || !dataset) return
        stopRequested = false

        const Nh = numHidden()
        const M = Math.min(numMemories(), dataset.train.count)
        const epochs = trainEpochs()
        const lr = learningRate()
        const BATCH_SIZE = Math.min(512, M)

        setState({
            isTraining: true,
            trainLosses: [],
            trainEpochsDone: 0,
            error: null,
            basicMemoryImages: [],
            recallFrames: [],
            currentRecallStep: 0,
        })

        // Build memory array (CPU-side, no GPU tensor for the full set)
        memoryImages = new Float32Array(M * Nv)
        const labels = new Int32Array(M)
        for (let i = 0; i < M; i++) {
            const srcStart = i * Nv
            memoryImages.set(dataset.train.images.subarray(srcStart, srcStart + Nv), i * Nv)
            labels[i] = dataset.train.labels[i]
        }
        setState({ numMemoriesStored: M, memoryLabels: Array.from(labels) })

        // Init params & optimizer
        if (trainParams) tree.dispose(trainParams)
        trainParams = initDamParams(seed(), Nv, Nh)
        const { solver, optState: initOpt } = createOptimizer(trainParams, lr)
        let currentOptState: OptState = initOpt

        const rng = createSeededRng(seed() + 999)

        try {
            for (let epoch = 0; epoch < epochs; epoch++) {
                if (stopRequested) break

                // Sample a random mini-batch
                const batchData = new Float32Array(BATCH_SIZE * Nv)
                for (let b = 0; b < BATCH_SIZE; b++) {
                    const idx = Math.floor(rng.next() * M)
                    batchData.set(
                        memoryImages.subarray(idx * Nv, (idx + 1) * Nv),
                        b * Nv
                    )
                }
                const batchTensor = np.array(batchData).reshape([BATCH_SIZE, Nv])

                const result = await trainStep(trainParams, currentOptState, solver, batchTensor)
                trainParams = result.params
                currentOptState = result.optState
                
                let basics: Float32Array[] | undefined
                let partialTheta: number | undefined
                
                // Periodically extract basic memories to visualize training progress
                if (epoch === 0 || epoch % 5 === 0) {
                    const tempXi = new Float32Array(await trainParams.xi.ref.data() as ArrayLike<number>)
                    partialTheta = (await trainParams.theta.ref.jsAsync() as number[])[0]
                    basics = []
                    for (let mu = 0; mu < Nh; mu++) {
                        basics.push(extractBasicMemory(tempXi, Nv, Nh, mu))
                    }
                    xiData = tempXi 
                }

                setState(produce(s => {
                    s.trainLosses.push(result.loss)
                    s.trainEpochsDone = epoch + 1
                    if (basics) s.basicMemoryImages = basics
                    if (partialTheta !== undefined) s.learnedTheta = partialTheta
                }))

                // Yield to UI thread periodically
                if (epoch % 5 === 0) {
                    await new Promise(r => setTimeout(r, 0))
                }
            }

            // Extract results
            xiData = new Float32Array(await trainParams.xi.ref.data() as ArrayLike<number>)
            const theta = (await trainParams.theta.ref.jsAsync() as number[])[0]
            setState({ learnedTheta: theta })

            // Extract basic memories
            const basics: Float32Array[] = []
            for (let mu = 0; mu < Nh; mu++) {
                basics.push(extractBasicMemory(xiData, Nv, Nh, mu))
            }
            setState({ basicMemoryImages: basics })
        } catch (err) {
            setState({ error: err instanceof Error ? err.message : 'Training failed' })
        } finally {
            setState({ isTraining: false })
        }
    }

    function stopTraining() {
        stopRequested = true
    }

    function runRecall(cueIndex?: number): void {
        if (!xiData || !memoryImages || !dataset) return

        const idx = cueIndex ?? state.selectedCueIndex
        const Nh = numHidden()
        const ratio = tauRatio()
        const steps = recallSteps()
        const noise = noiseLevel()

        const originalImg = new Float32Array(
            dataset.train.images.subarray(idx * Nv, (idx + 1) * Nv)
        )
        const rng = createSeededRng(idx + 1000)
        const noisyCue = addNoise(originalImg, noise, rng)

        setState({
            selectedCueIndex: idx,
            originalImage: originalImg,
            noisyCueImage: noisyCue,
            cueImage: new Float32Array(noisyCue),
            isRecalling: true,
            recallFrames: [],
            currentRecallStep: 0,
        })

        // Use configurable recall threshold, default to learned theta
        const theta = recallTheta() ?? state.learnedTheta
        let v = new Float32Array(noisyCue)

        // Store frame 0 (the noisy cue)
        const frame0: RecallFrame = {
            visible: new Float32Array(v),
            hBinary: new Float32Array(Nh),
            hActivated: new Float32Array(Nh),
            h: new Float32Array(Nh),
            mseToTarget: mse(v, originalImg),
            step: 0,
        }

        const frames: RecallFrame[] = [frame0]

        for (let step = 1; step <= steps; step++) {
            const result = recallStep(xiData, theta, v, Nv, Nh, ratio, recallBeta())
            v = result.v as any
            frames.push({
                visible: new Float32Array(result.v),
                hBinary: new Float32Array(result.hBinary),
                hActivated: new Float32Array(result.hActivated),
                h: new Float32Array(result.h),
                mseToTarget: mse(result.v, originalImg),
                step,
            })
        }

        setState({
            recallFrames: frames,
            currentRecallStep: frames.length - 1,
            isRecalling: false,
        })
    }

    function setCurrentStep(step: number) {
        const maxStep = state.recallFrames.length - 1
        setState({ currentRecallStep: Math.max(0, Math.min(step, maxStep)) })
    }

    function getMemoryPixels(idx: number): Float32Array {
        if (!memoryImages) return new Float32Array(Nv)
        return new Float32Array(memoryImages.subarray(idx * Nv, (idx + 1) * Nv))
    }

    return {
        state,
        numHidden, setNumHidden,
        numMemories, setNumMemories,
        trainEpochs, setTrainEpochs,
        learningRate, setLearningRate,
        noiseLevel, setNoiseLevel,
        tauRatio, setTauRatio,
        recallSteps, setRecallSteps,
        seed, setSeed,
        recallTheta, setRecallTheta,
        recallBeta, setRecallBeta,
        initialize,
        startTraining,
        stopTraining,
        runRecall,
        setCurrentStep,
        getMemoryPixels,
    }
}

export type DamStore = ReturnType<typeof createDamStore>
