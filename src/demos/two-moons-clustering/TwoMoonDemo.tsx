import { For, Show, onMount } from 'solid-js'
import { createTrainingStore } from './training-store'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { SingleStageLayout } from '../../components/demo-layouts/single-stage'

const optimizerOptions: Array<{ label: string; value: 'adam' | 'sgd' }> = [
    { label: 'Adam', value: 'adam' },
    { label: 'SGD', value: 'sgd' },
]

export function TwoMoonPage() {
    return <TwoMoonDemo />
}

export function TwoMoonDemo() {
    const store = createTrainingStore({ hiddenSize: 16, gridResolution: 40 })
    const {
        state,
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
        generateData,
        reinitOptimizer,
        runTraining,
    } = store

    onMount(() => store.initialize())

    const canvasSize = 760
    const gridRes = 40

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-5">
            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Epoch</div>
                    <div class="stat-value text-2xl text-primary">{state.epoch}</div>
                </div>
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Loss</div>
                    <div class="stat-value text-2xl text-secondary">{state.loss !== null ? state.loss.toFixed(4) : '—'}</div>
                </div>
            </div>

            <Show when={!state.isLoading}>
                <div class="badge badge-lg gap-2 bg-primary/20 border-primary/30 text-primary">
                    <div class="w-2 h-2 rounded-full bg-primary animate-pulse" />
                    {state.device}
                </div>
            </Show>

            <ConfigRadioGroup
                label="Optimizer"
                name="optimizer"
                value={optimizerType()}
                options={optimizerOptions}
                onChange={(value) => {
                    setOptimizerType(value)
                    reinitOptimizer()
                }}
                controlsInline
                disabled={state.isTraining || state.isLoading}
            />

            <ConfigSlider
                label="Learning Rate"
                value={learningRate()}
                min={0.001}
                max={1}
                step={0.001}
                onChange={setLearningRate}
                disabled={state.isTraining || state.isLoading}
                minLabel="0.001"
                maxLabel="1.000"
                valueFormatter={(value) => value.toFixed(3)}
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="Epochs per Train"
                value={epochsPerRound()}
                min={10}
                max={1000}
                step={10}
                onChange={(value) => setEpochsPerRound(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="10"
                maxLabel="1000"
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="Batch Size"
                value={batchSize()}
                min={1}
                max={nSamples() * 2}
                step={1}
                onChange={(value) => setBatchSize(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="1"
                maxLabel={String(nSamples() * 2)}
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
            />

            <ConfigSlider
                label="Samples per Moon"
                value={nSamples()}
                min={20}
                max={200}
                step={10}
                onChange={(value) => setNSamples(Math.round(value))}
                disabled={state.isTraining || state.isLoading}
                minLabel="20"
                maxLabel="200"
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
            />

            <ConfigSlider
                label="Noise Level"
                value={noise()}
                min={0}
                max={0.5}
                step={0.01}
                onChange={setNoise}
                disabled={state.isTraining || state.isLoading}
                minLabel="0.00"
                maxLabel="0.50"
                valueFormatter={(value) => value.toFixed(2)}
                valueClass="label-text-alt text-accent font-mono"
                rangeClass="range range-accent range-sm"
            />

            <div class="flex gap-3 pt-1">
                <button
                    class="btn btn-primary flex-1 shadow-lg shadow-primary/20"
                    onClick={() => runTraining()}
                    disabled={state.isTraining || state.isLoading}
                >
                    <Show when={state.isTraining} fallback={
                        <>
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            Train
                        </>
                    }>
                        <span class="loading loading-spinner loading-sm" />
                        Training...
                    </Show>
                </button>

                <button
                    class="btn btn-outline flex-1"
                    onClick={generateData}
                    disabled={state.isTraining || state.isLoading}
                >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    New Data
                </button>
            </div>

            <div class="alert bg-base-300/30 border-base-content/10">
                <svg class="w-5 h-5 text-info" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span class="text-xs text-base-content/60">
                    A 2-layer neural network with automatic differentiation, running entirely in your browser.
                </span>
            </div>
        </div>
    )

    const StagePanel = () => (
        <div class="card h-full bg-base-200/60 backdrop-blur-xl shadow-2xl border border-base-content/5">
            <div class="card-body h-full min-h-0 p-4 md:p-6 space-y-4">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold text-base-content">
                        Two-Moon Classification
                    </h2>
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
                    <>
                        <div class="relative min-h-0 grow rounded-2xl overflow-hidden border border-base-content/10 bg-base-300/50">
                            <svg
                                width={canvasSize}
                                height={canvasSize}
                                viewBox={`0 0 ${canvasSize} ${canvasSize}`}
                                class="w-full h-full"
                            >
                                <Show when={state.decisionGrid}>
                                    <For each={state.decisionGrid!}>
                                        {(row, i) => (
                                            <For each={row}>
                                                {(prob, j) => {
                                                    const hue = prob < 0.5 ? 200 : 340
                                                    const lightness = 50 + (0.5 - Math.abs(prob - 0.5)) * 40
                                                    return (
                                                        <rect
                                                            x={(j() / gridRes) * canvasSize}
                                                            y={(i() / gridRes) * canvasSize}
                                                            width={canvasSize / gridRes + 1}
                                                            height={canvasSize / gridRes + 1}
                                                            fill={`hsl(${hue}, 70%, ${lightness}%)`}
                                                            opacity={0.5}
                                                        />
                                                    )
                                                }}
                                            </For>
                                        )}
                                    </For>
                                </Show>

                                <For each={state.points}>
                                    {(point) => (
                                        <circle
                                            cx={point.x * canvasSize}
                                            cy={(1 - point.y) * canvasSize}
                                            r={6}
                                            fill={point.label === 0 ? '#22d3ee' : '#f472b6'}
                                            stroke="rgba(0,0,0,0.3)"
                                            stroke-width={2}
                                            class="drop-shadow-lg"
                                        />
                                    )}
                                </For>
                            </svg>
                        </div>

                        <div class="flex gap-6 justify-center text-sm">
                            <span class="flex items-center gap-2">
                                <span class="w-4 h-4 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50" />
                                <span class="text-base-content/70">Class 0</span>
                            </span>
                            <span class="flex items-center gap-2">
                                <span class="w-4 h-4 rounded-full bg-pink-400 shadow-lg shadow-pink-400/50" />
                                <span class="text-base-content/70">Class 1</span>
                            </span>
                        </div>
                    </>
                </Show>
            </div>
        </div>
    )

    return (
        <SingleStageLayout
            title="Two-Moon Classification"
            subtitle="Train a neural network to classify two interleaving half-moon shapes."
            backHref="/"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
