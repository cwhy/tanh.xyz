import { For, Index, Show, createEffect, createMemo, createSignal, onMount, type JSX } from 'solid-js'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { SingleStageFullLayout } from '../../components/demo-layouts/SingleStageFull'
import { LossPlot, type LossPlotPoint } from '../../components/plots/loss'
import { createMnistTrainingStore, type GradientRankSample, type RankingMode } from './training-store'

const optimizerOptions: Array<{ label: string; value: 'adam' | 'sgd' }> = [
    { label: 'Adam', value: 'adam' },
    { label: 'SGD', value: 'sgd' },
]

const rankingOptions: Array<{ label: string; value: RankingMode }> = [
    { label: 'All Seen', value: 'all' },
    { label: 'Latest Window', value: 'window' },
]

function SamplePreview(props: { pixels: Float32Array }): JSX.Element {
    let canvas!: HTMLCanvasElement

    createEffect(() => {
        const pixels = props.pixels
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const imageData = ctx.createImageData(28, 28)
        for (let i = 0; i < 28 * 28; i++) {
            const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)))
            imageData.data[i * 4 + 0] = v
            imageData.data[i * 4 + 1] = v
            imageData.data[i * 4 + 2] = v
            imageData.data[i * 4 + 3] = 255
        }
        ctx.putImageData(imageData, 0, 0)
    })

    return (
        <canvas
            ref={canvas}
            width={28}
            height={28}
            class="w-48 h-48 rounded-lg bg-base-100 [image-rendering:pixelated] shadow-md shadow-base-content/10"
        />
    )
}

function SampleCard(props: {
    sample: GradientRankSample
    getPixels: (trainPosition: number) => Float32Array
    showLabel: boolean
}): JSX.Element {
    const pixels = createMemo(() => props.getPixels(props.sample.trainPosition))

    return (
        <div class="flex justify-center">
            <div class="relative">
                <SamplePreview pixels={pixels()} />
                <div class="absolute left-1 top-1 rounded bg-black/55 px-1.5 py-0.5 text-[10px] text-white font-mono">
                    g {props.sample.gradNorm.toFixed(3)}
                </div>
                <Show when={props.showLabel}>
                    <div class="absolute right-1 top-1 rounded bg-primary/85 px-1.5 py-0.5 text-[10px] text-primary-content font-semibold">
                        {props.sample.label}
                    </div>
                </Show>
                <div class="absolute right-1 bottom-1 rounded bg-black/55 px-1.5 py-0.5 text-[10px] text-white font-mono">
                    p {props.sample.predictedLabel}
                </div>
            </div>
        </div>
    )
}

function DigitRecallBars(props: { recall: number[] }): JSX.Element {
    return (
        <div class="rounded-2xl border border-base-content/10 bg-base-100/90 p-4 shadow-xl shadow-base-content/5">
            <h4 class="text-sm font-semibold mb-2">Digit Recall (Test Set)</h4>
            <div class="space-y-2">
                <For each={props.recall}>
                    {(value, index) => (
                        <div class="flex items-center gap-2">
                            <span class="w-4 text-xs font-mono text-base-content/70">{index()}</span>
                            <div class="h-4 flex-1 rounded bg-base-300/60 overflow-hidden">
                                <div
                                    class="h-full bg-emerald-500"
                                    style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
                                />
                            </div>
                            <span class="w-12 text-right text-xs font-mono text-base-content/80">
                                {(value * 100).toFixed(1)}%
                            </span>
                        </div>
                    )}
                </For>
            </div>
        </div>
    )
}

export function MnistTrainingPage() {
    return <MnistTrainingDemo />
}

export function MnistTrainingDemo() {
    const store = createMnistTrainingStore()
    const {
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
    } = store

    onMount(() => {
        void initialize()
    })
    const [showDigitLabels, setShowDigitLabels] = createSignal(true)
    const displayedSamples = createMemo(() => getDisplayedTopSamples())
    const lossPlotPoints = createMemo<LossPlotPoint[]>(() => state.lossCurve.map(point => ({
        x: point.samples,
        train: point.trainLoss,
        test: point.testLoss,
    })))

    const learningRateConfig = () => {
        if (optimizerType() === 'adam') {
            return {
                min: 0.0001,
                max: 0.01,
                step: 0.0001,
                minLabel: '1e-4',
                maxLabel: '1e-2',
                fallback: 0.001,
            }
        }
        return {
            min: 0.001,
            max: 0.2,
            step: 0.001,
            minLabel: '1e-3',
            maxLabel: '2e-1',
            fallback: 0.05,
        }
    }

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-5">
            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Trained</div>
                    <div class="stat-value text-2xl text-primary">{state.totalSamplesTrained}</div>
                </div>
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Budget</div>
                    <div class="stat-value text-2xl text-secondary">{state.sampleBudget || '—'}</div>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/40 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Train Accuracy</div>
                    <div class="stat-value text-lg text-blue-700">
                        {state.trainAccuracy !== null ? `${(state.trainAccuracy * 100).toFixed(2)}%` : '—'}
                    </div>
                </div>
                <div class="stat bg-base-300/40 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Test Accuracy</div>
                    <div class="stat-value text-lg text-fuchsia-600">
                        {state.testAccuracy !== null ? `${(state.testAccuracy * 100).toFixed(2)}%` : '—'}
                    </div>
                </div>
            </div>

            <label class="form-control">
                <div class="label pb-2 flex justify-between items-center">
                    <span class="label-text font-medium">Seed</span>
                    <span class="label-text-alt font-mono text-primary">{seed()}</span>
                </div>
                <input
                    type="number"
                    class="input input-sm input-bordered"
                    value={seed()}
                    min={0}
                    max={2147483647}
                    step={1}
                    disabled={state.isTraining || state.isLoading}
                    onInput={(event) => {
                        const value = Number.parseInt(event.currentTarget.value, 10)
                        if (Number.isFinite(value)) {
                            setSeed(Math.max(0, Math.min(2147483647, value)))
                        }
                    }}
                />
            </label>

            <ConfigSlider
                label="Max Samples (<1 epoch)"
                value={maxSamples()}
                min={100}
                max={10000}
                step={100}
                onChange={(value) => setMaxSamples(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="100"
                maxLabel="10000"
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="Hidden Size"
                value={hiddenSize()}
                min={16}
                max={256}
                step={16}
                onChange={(value) => setHiddenSize(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="16"
                maxLabel="256"
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
            />

            <ConfigSlider
                label="Learning Rate"
                value={learningRate()}
                min={learningRateConfig().min}
                max={learningRateConfig().max}
                step={learningRateConfig().step}
                onChange={setLearningRate}
                disabled={state.isTraining || state.isLoading}
                minLabel={learningRateConfig().minLabel}
                maxLabel={learningRateConfig().maxLabel}
                valueFormatter={(value) => value.toFixed(4)}
                valueClass="label-text-alt text-accent font-mono"
                rangeClass="range range-accent range-sm"
            />

            <ConfigRadioGroup
                label="Optimizer"
                name="optimizer"
                value={optimizerType()}
                options={optimizerOptions}
                onChange={(next) => {
                    setOptimizerType(next)
                    const config = next === 'adam'
                        ? { min: 0.0001, max: 0.01, fallback: 0.001 }
                        : { min: 0.001, max: 0.2, fallback: 0.05 }
                    if (learningRate() < config.min || learningRate() > config.max) {
                        setLearningRate(config.fallback)
                    }
                }}
                controlsInline
                disabled={state.isTraining || state.isLoading}
            />

            <ConfigSlider
                label="Viz Update (sec)"
                value={updateIntervalSec()}
                min={0.5}
                max={10}
                step={0.1}
                onChange={(value) => setUpdateIntervalSec(Math.round(value * 10) / 10)}
                disabled={state.isLoading}
                minLabel="0.5"
                maxLabel="10.0"
                valueFormatter={(value) => value.toFixed(1)}
                helperText="Refresh interval for both loss curve and gradient-top samples."
                valueClass="label-text-alt text-info font-mono"
                rangeClass="range range-info range-sm"
            />

            <ConfigRadioGroup
                label="Top-10 Ranking"
                name="ranking-mode"
                value={rankingMode()}
                options={rankingOptions}
                onChange={setRankingMode}
                controlsInline
                disabled={state.isLoading}
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

            <Show when={state.error}>
                <div class="alert alert-error text-sm">{state.error}</div>
            </Show>
        </div>
    )

    const StagePanel = () => (
        <div class="h-full min-h-0 flex flex-col gap-4 pr-4 md:pr-6">
            <Show
                when={!state.isLoading}
                fallback={
                    <div class="flex flex-col items-center justify-center h-full gap-4">
                        <span class="loading loading-spinner loading-lg text-primary" />
                        <span class="text-base-content/60 text-center">
                            {state.loadingMessage}
                            <br />
                            <span class="text-sm">Progress: {(state.loadingProgress * 100).toFixed(0)}%</span>
                        </span>
                    </div>
                }
            >
                <div class="min-h-0 grow flex flex-col gap-4">
                    <div class="rounded-2xl border border-base-content/10 bg-base-100/90 p-3 min-h-[280px] shadow-xl shadow-base-content/5">
                        <LossPlot
                            points={lossPlotPoints()}
                            xMax={state.sampleBudget}
                            xAxisLabel="Total samples trained"
                            trainLabel="Train Curve"
                            testLabel="Test Curve"
                            emptyLabel="Start training to render the curve."
                        />
                    </div>

                    <div class="min-h-0 grow grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-4">
                        <div class="rounded-2xl border border-base-content/10 bg-base-100/90 p-4 min-h-0 flex flex-col shadow-xl shadow-base-content/5">
                            <div class="flex items-center justify-between mb-3">
                                <h3 class="text-base font-semibold">Top 10 Gradient Norm Samples</h3>
                                <div class="flex items-center gap-2">
                                    <span class="badge badge-outline text-xs">
                                        {rankingMode() === 'all' ? 'All Seen' : 'Latest Window'}
                                    </span>
                                    <button
                                        type="button"
                                        class="btn btn-xs btn-ghost btn-square"
                                        aria-label={showDigitLabels() ? 'Hide digit labels' : 'Show digit labels'}
                                        title={showDigitLabels() ? 'Hide digit labels' : 'Show digit labels'}
                                        onClick={() => setShowDigitLabels(v => !v)}
                                    >
                                        <Show
                                            when={showDigitLabels()}
                                            fallback={
                                                <svg viewBox="0 0 24 24" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2">
                                                    <path d="M3 3l18 18" />
                                                    <path d="M10.58 10.59a2 2 0 0 0 2.83 2.82" />
                                                    <path d="M9.88 5.09A10.94 10.94 0 0 1 12 5c5 0 9.27 3.11 11 7-1.05 2.36-2.96 4.29-5.33 5.37" />
                                                    <path d="M6.61 6.61C4.62 7.85 3.08 9.73 2 12c1.73 3.89 6 7 10 7a10.94 10.94 0 0 0 2.73-.34" />
                                                </svg>
                                            }
                                        >
                                            <svg viewBox="0 0 24 24" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2">
                                                <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" />
                                                <circle cx="12" cy="12" r="3" />
                                            </svg>
                                        </Show>
                                    </button>
                                </div>
                            </div>

                            <div class="text-xs text-base-content/60 mb-3">
                                {rankingMode() === 'all'
                                    ? 'Ranks samples by largest ||∇θ loss_i||2 over all processed samples in this run.'
                                    : 'Ranks samples by largest ||∇θ loss_i||2 within the latest visualization window.'}
                            </div>

                            <div class="min-h-0 grow overflow-y-auto pr-1">
                                <Show
                                    when={displayedSamples().length > 0}
                                    fallback={<div class="text-sm text-base-content/50 pt-3">No ranked samples yet.</div>}
                                >
                                    <div class="flex flex-wrap gap-3">
                                        <Index each={displayedSamples()}>
                                            {(sample) => (
                                                <SampleCard
                                                    sample={sample()}
                                                    getPixels={getTrainImagePixels}
                                                    showLabel={showDigitLabels()}
                                                />
                                            )}
                                        </Index>
                                    </div>
                                </Show>
                            </div>
                        </div>

                        <div class="min-h-0 overflow-y-auto pr-1">
                            <DigitRecallBars recall={state.testRecallByDigit} />
                        </div>
                    </div>
                </div>
            </Show>
        </div>
    )

    return (
        <SingleStageFullLayout
            title="MNIST with Gradient Norm"
            subtitle="Online training on a shuffled MNIST subset with seeded reproducibility and gradient-norm sample tracking."
            backHref="/demo"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
