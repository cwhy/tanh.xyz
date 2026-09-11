import { For, Show, createMemo, onMount } from 'solid-js'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { SingleStageLayout } from '../../components/demo-layouts/SingleStage'
import { LossPlot, type LossPlotPoint } from '../../components/plots/loss'
import { DEFAULT_ADDITION_CURRICULUM } from '../../lib/datasets/jax/ten-digit-addition'
import { createAdditionGrokkingStore } from './training-store'

const modelOptions: Array<{ label: string; value: 'mini-transformer' | 'mlp' }> = [
    { label: 'Mini Transformer (456p)', value: 'mini-transformer' },
    { label: 'MLP Baseline', value: 'mlp' },
]

const optimizerOptions: Array<{ label: string; value: 'adamw' | 'adam' | 'sgd' | 'rmsprop' }> = [
    { label: 'AdamW', value: 'adamw' },
    { label: 'RMSProp', value: 'rmsprop' },
    { label: 'Adam', value: 'adam' },
    { label: 'SGD', value: 'sgd' },
]

export function AdditionGrokkingPage() {
    return <AdditionGrokkingDemo />
}

export function AdditionGrokkingDemo() {
    const store = createAdditionGrokkingStore()
    const {
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
    } = store

    onMount(() => {
        void initialize()
    })

    const plotPoints = createMemo<LossPlotPoint[]>(() => state.lossCurve.map(point => ({
        x: point.step,
        train: point.trainLoss,
        test: point.testLoss,
    })))
    const phasePlots = createMemo(() => {
        let cumulativeEnd = 0
        return DEFAULT_ADDITION_CURRICULUM.map((phase, index) => {
            const phaseStart = cumulativeEnd + 1
            cumulativeEnd += phase.steps
            const phaseEnd = cumulativeEnd
            const points = plotPoints()
                .filter(point => point.x >= phaseStart && point.x <= phaseEnd)
                .map(point => ({
                    x: point.x - phaseStart + 1,
                    train: point.train,
                    test: point.test,
                }))
            return {
                key: `phase-${index + 1}`,
                title: `Phase ${index + 1} (${phase.minDigits}-${phase.maxDigits} digits)`,
                xMax: phase.steps,
                points,
            }
        })
    })

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-5">
            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Step</div>
                    <div class="stat-value text-2xl text-primary">{state.currentStep}</div>
                </div>
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Model</div>
                    <div class="stat-value text-sm text-secondary leading-tight">
                        {modelType() === 'mini-transformer' ? `Mini Transformer (${miniTransformerParamCount()}p)` : 'MLP Baseline'}
                    </div>
                </div>
            </div>

            <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                <div class="stat-title text-xs opacity-60">Curriculum Phase</div>
                <div class="stat-value text-base text-secondary">{state.phaseLabel}</div>
            </div>

            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/40 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Train Exact</div>
                    <div class="stat-value text-lg text-blue-700">
                        {state.trainExact !== null ? `${(state.trainExact * 100).toFixed(2)}%` : '—'}
                    </div>
                </div>
                <div class="stat bg-base-300/40 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Test Exact</div>
                    <div class="stat-value text-lg text-fuchsia-600">
                        {state.testExact !== null ? `${(state.testExact * 100).toFixed(2)}%` : '—'}
                    </div>
                </div>
            </div>

            <ConfigSlider
                label="Seed"
                value={seed()}
                min={1}
                max={100}
                step={1}
                onChange={(value) => setSeed(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="1"
                maxLabel="100"
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigRadioGroup
                label="Model"
                name="addition-model"
                value={modelType()}
                options={modelOptions}
                onChange={(next) => {
                    setModelType(next)
                    if (next === 'mini-transformer') {
                        resetRepoDefaults()
                    } else {
                        resetMlpDefaults()
                    }
                }}
                controlsInline
                disabled={state.isTraining || state.isLoading}
            />

            <ConfigRadioGroup
                label="Optimizer"
                name="addition-optimizer"
                value={optimizerType()}
                options={optimizerOptions}
                onChange={setOptimizerType}
                controlsInline
                disabled={state.isTraining || state.isLoading}
            />

            <ConfigSlider
                label="Learning Rate"
                value={learningRate()}
                min={0.0002}
                max={0.02}
                step={0.0002}
                onChange={setLearningRate}
                disabled={state.isTraining || state.isLoading}
                minLabel="2e-4"
                maxLabel="2e-2"
                valueFormatter={(value) => value.toFixed(4)}
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="Weight Decay"
                value={weightDecay()}
                min={0}
                max={0.05}
                step={0.001}
                onChange={setWeightDecay}
                disabled={state.isTraining || state.isLoading}
                minLabel="0"
                maxLabel="0.05"
                valueFormatter={(value) => value.toFixed(3)}
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
            />

            <ConfigSlider
                label="Batch Size"
                value={batchSize()}
                min={64}
                max={1024}
                step={32}
                onChange={(value) => setBatchSize(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="64"
                maxLabel="1024"
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="Total Steps"
                value={totalSteps()}
                min={1000}
                max={54000}
                step={500}
                onChange={(value) => setTotalSteps(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="1000"
                maxLabel="54000"
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
            />

            <ConfigSlider
                label="Eval Interval"
                value={evalInterval()}
                min={100}
                max={5000}
                step={100}
                onChange={(value) => setEvalInterval(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="100"
                maxLabel="5000"
                valueClass="label-text-alt text-accent font-mono"
                rangeClass="range range-accent range-sm"
            />

            <ConfigSlider
                label="Validation Size"
                value={testSetSize()}
                min={512}
                max={10000}
                step={256}
                onChange={(value) => setTestSetSize(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="512"
                maxLabel="10000"
                valueClass="label-text-alt text-info font-mono"
                rangeClass="range range-info range-sm"
            />

            <div class="flex gap-3 pt-1">
                <button
                    class="btn btn-primary flex-1 shadow-lg shadow-primary/20"
                    onClick={() => void startTraining()}
                    disabled={state.isTraining || state.isLoading}
                >
                    <Show when={state.isTraining} fallback="Train Run">
                        <span class="loading loading-spinner loading-sm" />
                        Training...
                    </Show>
                </button>

                <button
                    class="btn btn-outline flex-1"
                    onClick={stopTraining}
                    disabled={!state.isTraining || state.isLoading}
                >
                    Stop
                </button>
            </div>

            <button
                class="btn btn-sm btn-ghost w-full"
                onClick={resetRepoDefaults}
                disabled={state.isTraining || state.isLoading}
            >
                Reset to Repo Defaults
            </button>

            <Show when={state.error}>
                <div class="alert alert-error text-sm">{state.error}</div>
            </Show>
        </div>
    )

    const StagePanel = () => (
        <div class="card h-full bg-base-200/60 backdrop-blur-xl shadow-2xl border border-base-content/5">
            <div class="card-body h-full min-h-0 p-4 md:p-6 space-y-4">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold text-base-content">Grokking Loss Curves</h2>
                    <Show when={!state.isLoading}>
                        <div class="badge badge-lg gap-2 bg-primary/20 border-primary/30 text-primary">
                            <div class="w-2 h-2 rounded-full bg-primary animate-pulse" />
                            {state.device}
                        </div>
                    </Show>
                </div>

                <Show
                    when={!state.isLoading}
                    fallback={
                        <div class="flex flex-col items-center justify-center h-full gap-4">
                            <span class="loading loading-spinner loading-lg text-primary" />
                            <span class="text-base-content/50">Initializing jax-js...</span>
                        </div>
                    }
                >
                    <div class="w-full flex-1 min-h-0 flex flex-col gap-3 overflow-y-auto pr-1">
                        <For each={phasePlots()}>
                            {(phasePlot) => (
                                <div class="w-full space-y-2 shrink-0">
                                    <div class="text-sm font-semibold text-base-content/75 px-1">{phasePlot.title}</div>
                                    <div class="w-full rounded-2xl border border-base-content/10 bg-base-100/90 p-3 h-[340px] shadow-xl shadow-base-content/5">
                                        <LossPlot
                                            points={phasePlot.points}
                                            xMax={phasePlot.xMax}
                                            xAxisLabel="Phase step"
                                            trainLabel="Train loss"
                                            testLabel="Test loss"
                                            emptyLabel="Start training to render this phase curve."
                                        />
                                    </div>
                                </div>
                            )}
                        </For>
                    </div>
                </Show>
            </div>
        </div>
    )

    return (
        <SingleStageLayout
            title="10-Digit Addition Grokking"
            subtitle={() => (
                <>
                    Replica-style setup with{' '}
                    <a
                        href="https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition"
                        target="_blank"
                        rel="noreferrer"
                        class="link link-hover"
                    >
                        a 456-parameter mini-transformer
                    </a>
                    {' '}default (plus MLP baseline) on 10-digit addition.
                </>
            )}
            backHref="/demo"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
