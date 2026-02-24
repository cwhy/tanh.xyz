import { createSignal } from 'solid-js'
import { createStore } from 'solid-js/store'
import { numpy as np, tree } from '@jax-js/jax'
import {
    ADDITION_INPUT_SIZE,
    ADDITION_LM_INPUT_LEN,
    ADDITION_PROMPT_LEN,
    ADDITION_TARGET_SIZE,
    DEFAULT_ADDITION_CURRICULUM,
    createSeededRng,
    encodeAdditionLmPairs,
    encodeAdditionPairs,
    getCurriculumPhase,
    pairKey,
    sampleAdditionPairs,
    sampleCurriculumBatch,
    sampleCurriculumLmBatch,
    type CurriculumPhase,
    type EncodedAdditionBatch,
    type EncodedAdditionLmBatch,
} from '../../lib/datasets/jax/ten-digit-addition'
import {
    createOptimizer,
    evaluateMiniTransformerExactMatch,
    evaluateMiniTransformerLoss,
    evaluateMlpExactMatch,
    evaluateMlpLoss,
    initJax,
    initMiniTransformerParams,
    initMlpParams,
    miniTransformerParamCount,
    trainMiniTransformerBatchStep,
    trainMlpBatchStep,
    type AdditionMiniTransformerParams,
    type AdditionMlpParams,
    type ModelType,
    type OptState,
    type OptimizerSolver,
    type OptimizerType,
} from './models'

export interface GrokkingLossPoint {
    step: number
    trainLoss: number
    testLoss: number
}

export interface AdditionGrokkingState {
    isLoading: boolean
    device: string
    error: string | null
    isTraining: boolean
    isStopping: boolean
    currentStep: number
    configuredSteps: number
    phaseLabel: string
    trainLoss: number | null
    testLoss: number | null
    trainExact: number | null
    testExact: number | null
    lossCurve: GrokkingLossPoint[]
}

const MLP_HIDDEN_SIZE = 96
const REPO_WARMUP_STEPS = 1350
const REPO_MIN_LR_RATIO = 0.1

export function createAdditionGrokkingStore() {
    const [state, setState] = createStore<AdditionGrokkingState>({
        isLoading: true,
        device: '',
        error: null,
        isTraining: false,
        isStopping: false,
        currentStep: 0,
        configuredSteps: 0,
        phaseLabel: '1-3 digits',
        trainLoss: null,
        testLoss: null,
        trainExact: null,
        testExact: null,
        lossCurve: [],
    })

    const [modelType, setModelType] = createSignal<ModelType>('mini-transformer')
    const [seed, setSeed] = createSignal(43)
    const [learningRate, setLearningRate] = createSignal(0.02)
    const [optimizerType, setOptimizerType] = createSignal<OptimizerType>('adamw')
    const [weightDecay, setWeightDecay] = createSignal(0.01)
    const [batchSize, setBatchSize] = createSignal(512)
    const [totalSteps, setTotalSteps] = createSignal(54000)
    const [evalInterval, setEvalInterval] = createSignal(1000)
    const [testSetSize, setTestSetSize] = createSignal(5000)

    let mlpParams: AdditionMlpParams | null = null
    let miniTransformerParams: AdditionMiniTransformerParams | null = null
    let solver: OptimizerSolver | null = null
    let optState: OptState | null = null

    let testMlpBatch: EncodedAdditionBatch | null = null
    let testMlpX: np.Array | null = null
    let testMlpY: np.Array | null = null

    let testLmBatch: EncodedAdditionLmBatch | null = null
    let testLmPromptTokens: np.Array | null = null
    let testLmInputTokens: np.Array | null = null
    let testLmLabelTokens: np.Array | null = null
    let testLmLabelMask: np.Array | null = null

    let testReserved = new Set<string>()
    let stopRequested = false

    function disposeTreeSafe(value: Record<string, np.Array> | OptState | null) {
        if (!value) return
        try {
            tree.dispose(value)
        } catch {
            // Already moved/freed in jax-js ownership model.
        }
    }

    function disposeModelState() {
        const oldMlp = mlpParams
        const oldTransformer = miniTransformerParams
        const oldOptState = optState

        mlpParams = null
        miniTransformerParams = null
        optState = null
        solver = null

        disposeTreeSafe(oldMlp)
        disposeTreeSafe(oldTransformer)
        disposeTreeSafe(oldOptState)
    }

    function disposeEvalState() {
        testMlpX?.dispose()
        testMlpY?.dispose()
        testLmPromptTokens?.dispose()
        testLmInputTokens?.dispose()
        testLmLabelTokens?.dispose()
        testLmLabelMask?.dispose()

        testMlpX = null
        testMlpY = null
        testLmPromptTokens = null
        testLmInputTokens = null
        testLmLabelTokens = null
        testLmLabelMask = null
    }

    function phaseLabel(phase: CurriculumPhase): string {
        return `${phase.minDigits}-${phase.maxDigits} digits`
    }

    function cosineLr(step: number, maxSteps: number, baseLr: number, warmupSteps: number, minLrRatio: number): number {
        if (step < warmupSteps) {
            return baseLr * (step + 1) / Math.max(1, warmupSteps)
        }
        if (step >= maxSteps) {
            return baseLr * minLrRatio
        }
        const progress = (step - warmupSteps) / Math.max(1, maxSteps - warmupSteps)
        const cosine = 0.5 * (1 + Math.cos(Math.PI * progress))
        const minLr = baseLr * minLrRatio
        return minLr + (baseLr - minLr) * cosine
    }

    function learningRateSchedule(step: number): number {
        return cosineLr(step, totalSteps(), learningRate(), REPO_WARMUP_STEPS, REPO_MIN_LR_RATIO)
    }

    function prepareEvalSet() {
        const evalRng = createSeededRng(seed() + 1_000)
        const pairs = sampleAdditionPairs({
            count: testSetSize(),
            minDigits: 1,
            maxDigits: 10,
            rng: evalRng,
        })
        testReserved = new Set(pairs.map(pair => pairKey(pair.a, pair.b)))

        testMlpBatch = encodeAdditionPairs(pairs)
        testLmBatch = encodeAdditionLmPairs(pairs)

        disposeEvalState()

        testMlpX = np.array(testMlpBatch.inputs).reshape([testMlpBatch.batchSize, ADDITION_INPUT_SIZE])
        testMlpY = np.array(testMlpBatch.targets).reshape([testMlpBatch.batchSize, ADDITION_TARGET_SIZE])

        testLmPromptTokens = np.array(testLmBatch.promptTokens, { dtype: np.int32 }).reshape([testLmBatch.batchSize, ADDITION_PROMPT_LEN])
        testLmInputTokens = np.array(testLmBatch.inputTokens, { dtype: np.int32 }).reshape([testLmBatch.batchSize, ADDITION_LM_INPUT_LEN])
        testLmLabelTokens = np.array(testLmBatch.labelTokens, { dtype: np.int32 }).reshape([testLmBatch.batchSize, ADDITION_LM_INPUT_LEN])
        testLmLabelMask = np.array(testLmBatch.labelMask).reshape([testLmBatch.batchSize, ADDITION_LM_INPUT_LEN])
    }

    async function initialize() {
        try {
            setState({ isLoading: true, error: null })
            const device = await initJax()
            prepareEvalSet()
            setState({
                isLoading: false,
                device,
                configuredSteps: totalSteps(),
            })
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to initialize addition demo'
            setState({
                isLoading: false,
                error: message,
            })
        }
    }

    async function runMlpTraining() {
        if (!testMlpBatch || !testMlpX || !testMlpY) throw new Error('MLP eval dataset not initialized')

        mlpParams = initMlpParams(seed(), ADDITION_INPUT_SIZE, MLP_HIDDEN_SIZE, ADDITION_TARGET_SIZE)
        const optimizer = createOptimizer(mlpParams, learningRateSchedule, optimizerType(), { weightDecay: weightDecay() })
        solver = optimizer.solver
        optState = optimizer.optState

        let localParams = mlpParams
        let localOptState = optState
        let localCurve: GrokkingLossPoint[] = []
        const mlpLossWeightDecay = optimizerType() === 'adamw' ? 0 : weightDecay()
        const curriculum = DEFAULT_ADDITION_CURRICULUM
        const trainRng = createSeededRng(seed())

        for (let step = 0; step < totalSteps(); step++) {
            if (stopRequested) break

            const shouldEval = step % Math.max(1, evalInterval()) === 0 || step === totalSteps() - 1 || stopRequested
            const batch = sampleCurriculumBatch({
                batchSize: batchSize(),
                step,
                rng: trainRng,
                phases: curriculum,
                reserved: testReserved,
            })
            const X = np.array(batch.inputs).reshape([batch.batchSize, ADDITION_INPUT_SIZE])
            const y = np.array(batch.targets).reshape([batch.batchSize, ADDITION_TARGET_SIZE])
            const evalX = shouldEval ? X.ref : null

            const result = await trainMlpBatchStep(localParams, localOptState, solver!, X, y, mlpLossWeightDecay)
            localParams = result.params
            localOptState = result.optState

            if (shouldEval) {
                const phase = getCurriculumPhase(step, curriculum)
                const testLoss = await evaluateMlpLoss(localParams, testMlpX, testMlpY)
                const trainExact = await evaluateMlpExactMatch(localParams, evalX!, batch.targetDigits)
                const testExact = await evaluateMlpExactMatch(localParams, testMlpX, testMlpBatch.targetDigits)

                localCurve = [...localCurve, {
                    step: step + 1,
                    trainLoss: result.loss,
                    testLoss,
                }]

                setState({
                    currentStep: step + 1,
                    phaseLabel: phaseLabel(phase),
                    trainLoss: result.loss,
                    testLoss,
                    trainExact,
                    testExact,
                    lossCurve: localCurve,
                })

                await new Promise(resolve => setTimeout(resolve, 0))
            }

        }

        mlpParams = tree.ref(localParams)
        optState = tree.ref(localOptState)
    }

    async function runMiniTransformerTraining() {
        if (!testLmBatch || !testLmPromptTokens || !testLmInputTokens || !testLmLabelTokens || !testLmLabelMask) {
            throw new Error('Transformer eval dataset not initialized')
        }

        miniTransformerParams = initMiniTransformerParams(seed())
        const optimizer = createOptimizer(miniTransformerParams, learningRateSchedule, optimizerType(), { weightDecay: weightDecay() })
        solver = optimizer.solver
        optState = optimizer.optState

        let localParams = miniTransformerParams
        let localOptState = optState
        let localCurve: GrokkingLossPoint[] = []
        const curriculum = DEFAULT_ADDITION_CURRICULUM
        const trainRng = createSeededRng(seed())

        for (let step = 0; step < totalSteps(); step++) {
            if (stopRequested) break

            const shouldEval = step % Math.max(1, evalInterval()) === 0 || step === totalSteps() - 1 || stopRequested
            const batch = sampleCurriculumLmBatch({
                batchSize: batchSize(),
                step,
                rng: trainRng,
                phases: curriculum,
                reserved: testReserved,
            })

            const inputTokens = np.array(batch.inputTokens, { dtype: np.int32 }).reshape([batch.batchSize, ADDITION_LM_INPUT_LEN])
            const labelTokens = np.array(batch.labelTokens, { dtype: np.int32 }).reshape([batch.batchSize, ADDITION_LM_INPUT_LEN])
            const labelMask = np.array(batch.labelMask).reshape([batch.batchSize, ADDITION_LM_INPUT_LEN])
            const promptTokens = shouldEval
                ? np.array(batch.promptTokens, { dtype: np.int32 }).reshape([batch.batchSize, ADDITION_PROMPT_LEN])
                : null

            const result = await trainMiniTransformerBatchStep(localParams, localOptState, solver!, inputTokens, labelTokens, labelMask)
            localParams = result.params
            localOptState = result.optState

            if (shouldEval) {
                const phase = getCurriculumPhase(step, curriculum)
                const testLoss = await evaluateMiniTransformerLoss(localParams, testLmInputTokens, testLmLabelTokens, testLmLabelMask)
                const trainExact = await evaluateMiniTransformerExactMatch(localParams, promptTokens!, batch.targetDigits)
                const testExact = await evaluateMiniTransformerExactMatch(localParams, testLmPromptTokens, testLmBatch.targetDigits)

                localCurve = [...localCurve, {
                    step: step + 1,
                    trainLoss: result.loss,
                    testLoss,
                }]

                setState({
                    currentStep: step + 1,
                    phaseLabel: phaseLabel(phase),
                    trainLoss: result.loss,
                    testLoss,
                    trainExact,
                    testExact,
                    lossCurve: localCurve,
                })

                await new Promise(resolve => setTimeout(resolve, 0))
            }

        }

        miniTransformerParams = tree.ref(localParams)
        optState = tree.ref(localOptState)
    }

    async function startTraining() {
        if (state.isLoading || state.isTraining) return
        if (!testMlpBatch || !testLmBatch) {
            setState({ error: 'Evaluation dataset is not ready. Reinitialize and try again.' })
            return
        }

        console.log('[addition-grokking] start run', {
            modelType: modelType(),
            optimizer: optimizerType(),
            batchSize: batchSize(),
            totalSteps: totalSteps(),
        })

        disposeModelState()
        prepareEvalSet()

        stopRequested = false
        setState({
            isTraining: true,
            isStopping: false,
            error: null,
            currentStep: 0,
            configuredSteps: totalSteps(),
            trainLoss: null,
            testLoss: null,
            trainExact: null,
            testExact: null,
            lossCurve: [],
        })

        try {
            if (modelType() === 'mini-transformer') {
                await runMiniTransformerTraining()
            } else {
                await runMlpTraining()
            }
            setState({
                isTraining: false,
                isStopping: false,
            })
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Training failed'
            setState({
                isTraining: false,
                isStopping: false,
                error: message,
            })
        } finally {
            stopRequested = false
        }
    }

    function stopTraining() {
        if (!state.isTraining) return
        stopRequested = true
        setState({ isStopping: true })
    }

    function resetRepoDefaults() {
        setLearningRate(0.02)
        setWeightDecay(0.01)
        setBatchSize(512)
        setTotalSteps(54000)
        setEvalInterval(1000)
        setTestSetSize(5000)
        setOptimizerType('adamw')
    }

    function resetMlpDefaults() {
        setLearningRate(0.005)
        setWeightDecay(0.00002)
        setBatchSize(128)
        setTotalSteps(6000)
        setEvalInterval(200)
        setTestSetSize(1024)
        setOptimizerType('adam')
    }

    return {
        state,
        modelType,
        setModelType,
        seed,
        setSeed,
        learningRate,
        setLearningRate,
        optimizerType,
        setOptimizerType,
        weightDecay,
        setWeightDecay,
        batchSize,
        setBatchSize,
        totalSteps,
        setTotalSteps,
        evalInterval,
        setEvalInterval,
        testSetSize,
        setTestSetSize,
        initialize,
        startTraining,
        stopTraining,
        miniTransformerParamCount,
        resetRepoDefaults,
        resetMlpDefaults,
    }
}

export type AdditionGrokkingStore = ReturnType<typeof createAdditionGrokkingStore>
