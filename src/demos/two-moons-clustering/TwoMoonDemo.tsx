import { onMount, For, Show } from 'solid-js'
import { createTrainingStore } from './training-store'

export function TwoMoonPage() {
    return (
        <div class="min-h-screen" style={{
            'background-color': '#fdfbf7',
            'background-image': 'radial-gradient(#e5e0d8 1px, transparent 1px)',
            'background-size': '24px 24px'
        }}>
            {/* Content */}
            <div class="relative">
                {/* Header */}
                <header class="pt-8 pb-4 px-6">
                    <div class="max-w-6xl mx-auto">
                        <a href="/" class="inline-flex items-center gap-3 mb-4 hover:opacity-70 transition-opacity group" style={{"color":"#2d2d2d","font-family":"'Patrick Hand', cursive"}}>
                            <svg class="w-5 h-5 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                            </svg>
                            Back to Demos
                        </a>
                        <h1 class="text-4xl md:text-5xl font-bold mb-2" style={{"font-family":"'Kalam', cursive","color":"#2d2d2d"}}>
                            Two-Moon Classification
                        </h1>
                        <p class="text-lg" style={{"font-family":"'Patrick Hand', cursive","color":"#2d2d2d","opacity":"0.7"}}>
                            Train a neural network to classify two interleaving half-moon shapes
                        </p>
                    </div>
                </header>

                {/* Demo Section */}
                <main class="container mx-auto px-4 pb-20 max-w-6xl">
                    <TwoMoonDemo />
                </main>
            </div>
        </div>
    )
}

export function TwoMoonDemo() {
    const store = createTrainingStore({ hiddenSize: 16, gridResolution: 40 })
    const {
        state,
        learningRate, setLearningRate,
        optimizerType, setOptimizerType,
        epochsPerRound, setEpochsPerRound,
        nSamples, setNSamples,
        noise, setNoise,
        batchSize, setBatchSize,
        generateData, reinitOptimizer, runTraining,
    } = store

    onMount(() => store.initialize())

    const canvasSize = 500
    const gridRes = 40

    return (
        <div class="card bg-base-200/50 backdrop-blur-xl shadow-2xl border border-base-content/5">
            <div class="card-body p-8">
                {/* Header */}
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-3xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                        Two-Moon Classification
                    </h2>
                    <Show when={!state.isLoading}>
                        <div class="badge badge-lg gap-2 bg-primary/20 border-primary/30 text-primary">
                            <div class="w-2 h-2 rounded-full bg-primary animate-pulse" />
                            {state.device}
                        </div>
                    </Show>
                </div>

                <Show when={!state.isLoading} fallback={
                    <div class="flex flex-col items-center justify-center h-96 gap-4">
                        <span class="loading loading-spinner loading-lg text-primary" />
                        <span class="text-base-content/50">Initializing jax-js...</span>
                    </div>
                }>
                    <div class="grid grid-cols-1 xl:grid-cols-[1.2fr,1fr] gap-8">
                        {/* Visualization */}
                        <div class="space-y-4">
                            <div class="relative rounded-2xl overflow-hidden border border-base-content/10 bg-base-300/50">
                                <svg
                                    width={canvasSize}
                                    height={canvasSize}
                                    viewBox={`0 0 ${canvasSize} ${canvasSize}`}
                                    class="w-full h-auto"
                                >
                                    {/* Decision boundary heatmap */}
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

                                    {/* Data points */}
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

                            {/* Legend */}
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
                        </div>

                        {/* Controls */}
                        <div class="space-y-6">
                            {/* Stats */}
                            <div class="grid grid-cols-2 gap-4">
                                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-4">
                                    <div class="stat-title text-xs opacity-60">Epoch</div>
                                    <div class="stat-value text-3xl text-primary">{state.epoch}</div>
                                </div>
                                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-4">
                                    <div class="stat-title text-xs opacity-60">Loss</div>
                                    <div class="stat-value text-3xl text-secondary">
                                        {state.loss !== null ? state.loss.toFixed(4) : '—'}
                                    </div>
                                </div>
                            </div>

                            {/* Sliders */}
                            <div class="space-y-5">
                                <div class="form-control">
                                    <label class="label pb-2">
                                        <span class="label-text font-medium">Optimizer</span>
                                    </label>
                                    <div class="join">
                                        <input
                                            class="join-item btn btn-sm btn-outline px-6"
                                            type="radio"
                                            name="optimizer"
                                            aria-label="Adam"
                                            checked={optimizerType() === 'adam'}
                                            onClick={() => {
                                                setOptimizerType('adam')
                                                reinitOptimizer()
                                            }}
                                            disabled={state.isTraining}
                                        />
                                        <input
                                            class="join-item btn btn-sm btn-outline px-6"
                                            type="radio"
                                            name="optimizer"
                                            aria-label="SGD"
                                            checked={optimizerType() === 'sgd'}
                                            onClick={() => {
                                                setOptimizerType('sgd')
                                                reinitOptimizer()
                                            }}
                                            disabled={state.isTraining}
                                        />
                                    </div>
                                </div>

                                <div class="form-control">
                                    <label class="label pb-2">
                                        <span class="label-text font-medium">Learning Rate</span>
                                        <span class="label-text-alt text-primary font-mono">{learningRate().toFixed(3)}</span>
                                    </label>
                                    <input
                                        type="range"
                                        min="0.001"
                                        max="1"
                                        step="0.001"
                                        value={learningRate()}
                                        onInput={(e) => setLearningRate(parseFloat(e.currentTarget.value))}
                                        class="range range-primary range-sm"
                                        disabled={state.isTraining}
                                    />
                                    <div class="flex justify-between text-xs text-base-content/40 mt-1">
                                        <span>0.001</span>
                                        <span>1.000</span>
                                    </div>
                                </div>

                                <div class="form-control">
                                    <label class="label pb-2">
                                        <span class="label-text font-medium">Batch Size</span>
                                        <span class="label-text-alt text-secondary font-mono">{batchSize()}</span>
                                    </label>
                                    <input
                                        type="range"
                                        min="1"
                                        max={nSamples() * 2}
                                        step="1"
                                        value={batchSize()}
                                        onInput={(e) => setBatchSize(parseInt(e.currentTarget.value))}
                                        class="range range-secondary range-sm"
                                        disabled={state.isTraining}
                                    />
                                    <div class="flex justify-between text-xs text-base-content/40 mt-1">
                                        <span>1</span>
                                        <span>{nSamples() * 2}</span>
                                    </div>
                                </div>

                                <div class="form-control">
                                    <label class="label pb-2">
                                        <span class="label-text font-medium">Samples per Moon</span>
                                        <span class="label-text-alt text-secondary font-mono">{nSamples()}</span>
                                    </label>
                                    <input
                                        type="range"
                                        min="20"
                                        max="200"
                                        step="10"
                                        value={nSamples()}
                                        onInput={(e) => setNSamples(parseInt(e.currentTarget.value))}
                                        class="range range-secondary range-sm"
                                        disabled={state.isTraining}
                                    />
                                    <div class="flex justify-between text-xs text-base-content/40 mt-1">
                                        <span>20</span>
                                        <span>200</span>
                                    </div>
                                </div>

                                <div class="form-control">
                                    <label class="label pb-2">
                                        <span class="label-text font-medium">Noise Level</span>
                                        <span class="label-text-alt text-accent font-mono">{noise().toFixed(2)}</span>
                                    </label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="0.5"
                                        step="0.01"
                                        value={noise()}
                                        onInput={(e) => setNoise(parseFloat(e.currentTarget.value))}
                                        class="range range-accent range-sm"
                                        disabled={state.isTraining}
                                    />
                                    <div class="flex justify-between text-xs text-base-content/40 mt-1">
                                        <span>0.00</span>
                                        <span>0.50</span>
                                    </div>
                                </div>
                            </div>

                            {/* Buttons */}
                            <div class="flex gap-3 pt-2">
                                <div
                                    class={`btn btn-primary flex-1 shadow-lg shadow-primary/20 flex flex-row items-center gap-2 ${state.isTraining ? 'cursor-not-allowed opacity-50' : ''}`}
                                    onClick={(e) => {
                                        const target = e.target as HTMLElement;
                                        if (target.tagName !== 'INPUT' && !state.isTraining) {
                                            runTraining();
                                        }
                                    }}
                                    role="button"
                                    tabindex="0"
                                >
                                    <Show when={state.isTraining} fallback={
                                        <>
                                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                                            </svg>
                                            Train for
                                            <input
                                                type="number"
                                                min="10"
                                                max="1000"
                                                step="10"
                                                value={epochsPerRound()}
                                                onInput={(e) => setEpochsPerRound(e.currentTarget.valueAsNumber)}
                                                onClick={(e) => e.stopPropagation()}
                                                class="input input-sm input-ghost w-20 text-center bg-white/20 text-primary-content font-bold focus:bg-white/30 focus:outline-none p-1 rounded h-8"
                                            />
                                            epochs
                                        </>
                                    }>
                                        <span class="loading loading-spinner loading-sm" />
                                        Training...
                                    </Show>
                                </div>
                                <button
                                    class="btn btn-outline flex-1"
                                    onClick={generateData}
                                    disabled={state.isTraining}
                                >
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                    New Data
                                </button>
                            </div>

                            {/* Info */}
                            <div class="alert bg-base-300/30 border-base-content/10">
                                <svg class="w-5 h-5 text-info" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span class="text-xs text-base-content/60">
                                    A 2-layer neural network with automatic differentiation, running entirely in your browser
                                </span>
                            </div>
                        </div>
                    </div>
                </Show>
            </div>
        </div>
    )
}
