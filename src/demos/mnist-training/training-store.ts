import { createSignal } from 'solid-js'
import { createStore } from 'solid-js/store'
import { numpy as np, tree } from '@jax-js/jax'
import { fetchMnistRaw, type MnistFetchProgress } from '../../lib/datasources/mnist'
import { formatMnistForMlp, type MnistMlpDataset } from '../../lib/datasets/jax/mnist'
import {
    initJax,
    initParams,
    createOptimizer,
    trainOnlineStep,
    evaluateLoss,
    evaluateAccuracy,
    evaluateClassificationMetrics,
    computeSampleGradNorm,
    predictSampleLabel,
    createSeededRng,
    type OptimizerType,
    type MnistMlpParams,
    type OptState,
    type OptimizerSolver,
} from './models'

export type RankingMode = 'all' | 'window'

export interface LossPoint {
    samples: number
    trainLoss: number
    testLoss: number
}

export interface GradientRankSample {
    trainPosition: number
    sourceIndex: number
    label: number
    predictedLabel: number
    gradNorm: number
    trainedAt: number
}

export interface MnistTrainingState {
    isLoading: boolean
    loadingMessage: string
    loadingProgress: number
    device: string
    error: string | null
    isTraining: boolean
    isStopping: boolean
    sampleBudget: number
    totalSamplesTrained: number
    latestSampleLoss: number | null
    trainLoss: number | null
    testLoss: number | null
    trainAccuracy: number | null
    testAccuracy: number | null
    testRecallByDigit: number[]
    lossCurve: LossPoint[]
    topSamplesAll: GradientRankSample[]
    topSamplesWindow: GradientRankSample[]
}

const MAX_TRAIN_SAMPLES = 10000
const PIXELS_PER_IMAGE = 28 * 28
const TOP_K = 10

function setStageText(progress: MnistFetchProgress): string {
    const stageMap: Record<MnistFetchProgress['stage'], string> = {
        'check-cache': 'Checking cache',
        downloading: 'Downloading',
        decompressing: 'Decompressing',
        parsing: 'Parsing IDX',
        ready: 'Ready',
    }
    return `${stageMap[progress.stage]} ${progress.file} (${progress.completedFiles}/${progress.totalFiles})`
}

function topKFromWindow(items: GradientRankSample[]): GradientRankSample[] {
    return [...items].sort((a, b) => b.gradNorm - a.gradNorm).slice(0, TOP_K)
}

export function createMnistTrainingStore() {
    const [state, setState] = createStore<MnistTrainingState>({
        isLoading: true,
        loadingMessage: 'Initializing jax-js...',
        loadingProgress: 0,
        device: '',
        error: null,
        isTraining: false,
        isStopping: false,
        sampleBudget: 0,
        totalSamplesTrained: 0,
        latestSampleLoss: null,
        trainLoss: null,
        testLoss: null,
        trainAccuracy: null,
        testAccuracy: null,
        testRecallByDigit: Array.from({ length: 10 }, () => 0),
        lossCurve: [],
        topSamplesAll: [],
        topSamplesWindow: [],
    })

    const [seed, setSeed] = createSignal(42)
    const [maxSamples, setMaxSamples] = createSignal(5000)
    const [hiddenSize, setHiddenSize] = createSignal(64)
    const [learningRate, setLearningRate] = createSignal(0.001)
    const [optimizerType, setOptimizerType] = createSignal<OptimizerType>('adam')
    const [updateIntervalSec, setUpdateIntervalSec] = createSignal(1)
    const [rankingMode, setRankingMode] = createSignal<RankingMode>('all')

    let dataset: MnistMlpDataset | null = null
    let params: MnistMlpParams | null = null
    let solver: OptimizerSolver | null = null
    let optState: OptState | null = null

    let trainEvalX: np.Array | null = null
    let trainEvalY: np.Array | null = null
    let testEvalX: np.Array | null = null
    let testEvalY: np.Array | null = null

    let activeTrainImages: Float32Array | null = null
    let activeTrainOneHot: Float32Array | null = null
    let activeTrainLabels: Int32Array | null = null
    let activeSourceIndices: Int32Array | null = null

    let stopRequested = false

    function disposePreparedState() {
        trainEvalX?.dispose()
        trainEvalY?.dispose()
        testEvalX?.dispose()
        testEvalY?.dispose()
        trainEvalX = null
        trainEvalY = null
        testEvalX = null
        testEvalY = null

        tree.dispose(params)
        params = null

        tree.dispose(optState)
        optState = null
        solver = null

        activeTrainImages = null
        activeTrainOneHot = null
        activeTrainLabels = null
        activeSourceIndices = null
    }

    async function initialize(): Promise<void> {
        try {
            setState({
                isLoading: true,
                loadingMessage: 'Initializing jax-js...',
                loadingProgress: 0,
                error: null,
            })

            const dev = await initJax()
            setState({
                device: dev,
                loadingMessage: 'Loading MNIST...',
            })

            const raw = await fetchMnistRaw({
                onProgress(progress) {
                    setState({
                        loadingMessage: setStageText(progress),
                        loadingProgress: progress.totalFiles > 0
                            ? progress.completedFiles / progress.totalFiles
                            : 0,
                    })
                },
            })
            dataset = formatMnistForMlp(raw)

            setState({
                isLoading: false,
                loadingMessage: 'MNIST loaded',
                loadingProgress: 1,
            })
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to initialize MNIST demo'
            setState({
                isLoading: false,
                error: message,
            })
        }
    }

    function buildTrainingSubset() {
        if (!dataset) throw new Error('MNIST dataset has not been loaded')

        const totalTrainCount = dataset.train.count
        const budget = Math.min(Math.max(1, maxSamples()), MAX_TRAIN_SAMPLES, totalTrainCount)
        const rng = createSeededRng(seed())

        const shuffled = Array.from({ length: totalTrainCount }, (_, i) => i)
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(rng.next() * (i + 1))
            const temp = shuffled[i]
            shuffled[i] = shuffled[j]
            shuffled[j] = temp
        }

        activeSourceIndices = Int32Array.from(shuffled.slice(0, budget))
        activeTrainImages = new Float32Array(budget * PIXELS_PER_IMAGE)
        activeTrainOneHot = new Float32Array(budget * 10)
        activeTrainLabels = new Int32Array(budget)

        for (let i = 0; i < budget; i++) {
            const sourceIndex = activeSourceIndices[i]
            const srcImageStart = sourceIndex * PIXELS_PER_IMAGE
            const srcImageEnd = srcImageStart + PIXELS_PER_IMAGE
            const dstImageStart = i * PIXELS_PER_IMAGE
            activeTrainImages.set(dataset.train.images.subarray(srcImageStart, srcImageEnd), dstImageStart)

            const srcOneHotStart = sourceIndex * 10
            const srcOneHotEnd = srcOneHotStart + 10
            const dstOneHotStart = i * 10
            activeTrainOneHot.set(dataset.train.oneHot.subarray(srcOneHotStart, srcOneHotEnd), dstOneHotStart)

            activeTrainLabels[i] = dataset.train.labels[sourceIndex]
        }

        trainEvalX = np.array(activeTrainImages).reshape([budget, PIXELS_PER_IMAGE])
        trainEvalY = np.array(activeTrainOneHot).reshape([budget, 10])
        testEvalX = np.array(dataset.test.images).reshape([dataset.test.count, PIXELS_PER_IMAGE])
        testEvalY = np.array(dataset.test.oneHot).reshape([dataset.test.count, 10])

        params = initParams(seed(), hiddenSize())
        const optimizer = createOptimizer(params, learningRate(), optimizerType())
        solver = optimizer.solver
        optState = optimizer.optState

        setState({
            sampleBudget: budget,
            totalSamplesTrained: 0,
            latestSampleLoss: null,
            trainLoss: null,
            testLoss: null,
            trainAccuracy: null,
            testAccuracy: null,
            testRecallByDigit: Array.from({ length: 10 }, () => 0),
            lossCurve: [],
            topSamplesAll: [],
            topSamplesWindow: [],
            error: null,
        })
    }

    function getTrainImagePixels(trainPosition: number): Float32Array {
        if (!activeTrainImages) {
            return new Float32Array(PIXELS_PER_IMAGE)
        }
        const start = trainPosition * PIXELS_PER_IMAGE
        const end = start + PIXELS_PER_IMAGE
        return activeTrainImages.subarray(start, end)
    }

    function createSampleArrays(trainPosition: number): { X: np.Array; y: np.Array } {
        if (!activeTrainImages || !activeTrainOneHot) {
            throw new Error('Training sample buffers are not initialized')
        }
        const xStart = trainPosition * PIXELS_PER_IMAGE
        const yStart = trainPosition * 10
        const X = np.array(activeTrainImages.subarray(xStart, xStart + PIXELS_PER_IMAGE)).reshape([1, PIXELS_PER_IMAGE])
        const y = np.array(activeTrainOneHot.subarray(yStart, yStart + 10)).reshape([1, 10])
        return { X, y }
    }

    function createSampleInput(trainPosition: number): np.Array {
        if (!activeTrainImages) {
            throw new Error('Training image buffer is not initialized')
        }
        const xStart = trainPosition * PIXELS_PER_IMAGE
        return np.array(activeTrainImages.subarray(xStart, xStart + PIXELS_PER_IMAGE)).reshape([1, PIXELS_PER_IMAGE])
    }

    async function attachPredictions(
        currentParams: MnistMlpParams,
        samples: GradientRankSample[]
    ): Promise<GradientRankSample[]> {
        const enriched: GradientRankSample[] = []
        for (const sample of samples) {
            const X = createSampleInput(sample.trainPosition)
            const predictedLabel = await predictSampleLabel(currentParams, X)
            enriched.push({ ...sample, predictedLabel })
        }
        return enriched
    }

    async function recalculateTopAllSamples(
        currentParams: MnistMlpParams,
        existingTop: GradientRankSample[],
        newSample: GradientRankSample
    ): Promise<GradientRankSample[]> {
        const rescored: GradientRankSample[] = []
        for (const sample of existingTop) {
            const { X, y } = createSampleArrays(sample.trainPosition)
            const gradNorm = await computeSampleGradNorm(currentParams, X.ref, y)
            const predictedLabel = await predictSampleLabel(currentParams, X)
            rescored.push({ ...sample, gradNorm, predictedLabel })
        }

        const { X: newX, y: newY } = createSampleArrays(newSample.trainPosition)
        const newGradNorm = await computeSampleGradNorm(currentParams, newX.ref, newY)
        const newPredictedLabel = await predictSampleLabel(currentParams, newX)

        return [...rescored, { ...newSample, gradNorm: newGradNorm, predictedLabel: newPredictedLabel }]
            .sort((a, b) => b.gradNorm - a.gradNorm)
            .slice(0, TOP_K)
    }

    async function evaluateSeenTrainMetrics(
        currentParams: MnistMlpParams,
        seenCount: number
    ): Promise<{ trainLoss: number; trainAccuracy: number }> {
        if (!activeTrainImages || !activeTrainOneHot || !activeTrainLabels) {
            throw new Error('Training buffers are missing')
        }

        const seen = Math.max(1, seenCount)
        const X = np.array(activeTrainImages.subarray(0, seen * PIXELS_PER_IMAGE)).reshape([seen, PIXELS_PER_IMAGE])
        const y = np.array(activeTrainOneHot.subarray(0, seen * 10)).reshape([seen, 10])
        const labels = activeTrainLabels.subarray(0, seen)

        const trainLoss = await evaluateLoss(currentParams, X, y)
        const trainAccuracy = await evaluateAccuracy(currentParams, X, labels)
        X.dispose()
        y.dispose()
        return { trainLoss, trainAccuracy }
    }

    async function startTraining(): Promise<void> {
        if (state.isTraining || state.isLoading) return
        if (!dataset) {
            setState({ error: 'Dataset is not ready yet' })
            return
        }

        disposePreparedState()
        try {
            buildTrainingSubset()
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to prepare training run'
            setState({ error: message })
            return
        }

        if (!params || !solver || !optState || !trainEvalX || !trainEvalY || !testEvalX || !testEvalY) {
            setState({ error: 'Training was not properly initialized' })
            return
        }
        if (!activeTrainImages || !activeTrainOneHot || !activeTrainLabels || !activeSourceIndices) {
            setState({ error: 'Training data is missing' })
            return
        }

        setState({
            isTraining: true,
            isStopping: false,
            error: null,
        })
        stopRequested = false

        let localParams = params
        let localOptState = optState
        let localTopAll: GradientRankSample[] = []
        let localWindowSamples: GradientRankSample[] = []
        let localTopWindowLatest: GradientRankSample[] = []
        let localCurve: LossPoint[] = []
        let localLatestLoss: number
        let lastVizAt = performance.now()

        const budget = state.sampleBudget
        for (let i = 0; i < budget; i++) {
            if (stopRequested) break

            const { X: xSample, y: ySample } = createSampleArrays(i)

            const result = await trainOnlineStep(localParams, localOptState, solver, xSample, ySample)
            localParams = result.params
            localOptState = result.optState
            localLatestLoss = result.loss

            const sampleMeta: GradientRankSample = {
                trainPosition: i,
                sourceIndex: activeSourceIndices[i],
                label: activeTrainLabels[i],
                predictedLabel: -1,
                gradNorm: result.gradNorm,
                trainedAt: i + 1,
            }

            localTopAll = await recalculateTopAllSamples(localParams, localTopAll, sampleMeta)
            localWindowSamples.push(sampleMeta)

            const now = performance.now()
            const currentIntervalMs = Math.min(10, Math.max(0.5, updateIntervalSec())) * 1000
            const isLastSample = i === budget - 1
            const shouldUpdateViz = isLastSample || stopRequested || now - lastVizAt >= currentIntervalMs

            if (shouldUpdateViz) {
                const { trainLoss, trainAccuracy } = await evaluateSeenTrainMetrics(localParams, i + 1)
                const testLoss = await evaluateLoss(localParams, testEvalX, testEvalY)
                const testMetrics = await evaluateClassificationMetrics(localParams, testEvalX, dataset.test.labels)
                const testAccuracy = testMetrics.accuracy

                const point: LossPoint = {
                    samples: i + 1,
                    trainLoss,
                    testLoss,
                }
                localCurve = [...localCurve, point]

                const localTopWindow = topKFromWindow(localWindowSamples)
                localWindowSamples = []
                if (localTopWindow.length > 0) {
                    localTopWindowLatest = await attachPredictions(localParams, localTopWindow)
                }

                setState({
                    totalSamplesTrained: i + 1,
                    latestSampleLoss: localLatestLoss,
                    trainLoss,
                    testLoss,
                    trainAccuracy,
                    testAccuracy,
                    testRecallByDigit: testMetrics.recallByDigit,
                    lossCurve: localCurve,
                    topSamplesAll: localTopAll,
                    topSamplesWindow: localTopWindowLatest,
                })

                lastVizAt = now
                await new Promise(resolve => setTimeout(resolve, 0))
            }
        }

        params = localParams
        optState = localOptState
        setState({
            isTraining: false,
            isStopping: false,
        })
        stopRequested = false
    }

    function stopTraining() {
        if (!state.isTraining) return
        stopRequested = true
        setState({ isStopping: true })
    }

    function getDisplayedTopSamples(): GradientRankSample[] {
        return rankingMode() === 'all' ? state.topSamplesAll : state.topSamplesWindow
    }

    return {
        state,
        seed,
        setSeed,
        maxSamples,
        setMaxSamples,
        hiddenSize,
        setHiddenSize,
        learningRate,
        setLearningRate,
        optimizerType,
        setOptimizerType,
        updateIntervalSec,
        setUpdateIntervalSec,
        rankingMode,
        setRankingMode,
        initialize,
        startTraining,
        stopTraining,
        getDisplayedTopSamples,
        getTrainImagePixels,
    }
}

export type MnistTrainingStore = ReturnType<typeof createMnistTrainingStore>
