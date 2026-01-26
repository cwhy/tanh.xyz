import { createSignal, onMount, For, Show } from 'solid-js'
import { generateTwoMoons, normalizeForVisualization, type DataPoint } from '../lib/two-moons'
import {
    initJax,
    initParams,
    initOptimizer,
    trainStep,
    predictGrid,
    type NetworkParams,
} from '../lib/neural-net'
import { numpy as np, tree } from '@jax-js/jax'
import type { OptState } from '@jax-js/optax'

export function TwoMoonDemo() {
    const [isLoading, setIsLoading] = createSignal(true)
    const [device, setDevice] = createSignal<string>('')
    const [points, setPoints] = createSignal<DataPoint[]>([])
    const [params, setParams] = createSignal<NetworkParams | null>(null)
    const [decisionGrid, setDecisionGrid] = createSignal<number[][] | null>(null)
    const [epoch, setEpoch] = createSignal(0)
    const [loss, setLoss] = createSignal<number | null>(null)
    const [isTraining, setIsTraining] = createSignal(false)
    const [learningRate, setLearningRate] = createSignal(0.01)
    const [optimizerType, setOptimizerType] = createSignal<'adam' | 'sgd'>('adam')
    const [epochsPerRound, setEpochsPerRound] = createSignal(500)
    const [nSamples, setNSamples] = createSignal(100)
    const [noise, setNoise] = createSignal(0.15)

    const [batchSize, setBatchSize] = createSignal(200)

    // Training data
    let X: np.Array | null = null
    let y: np.Array | null = null
    let trainingX: number[][] = []
    let trainingY: number[] = []

    // Training state
    let optState: OptState | null = null

    onMount(async () => {
        const dev = await initJax()
        setDevice(dev)
        setIsLoading(false)
        generateData()
    })

    const generateData = () => {
        // Cleanup old data
        X?.dispose()
        y?.dispose()

        // Cleanup old params using tree.dispose
        tree.dispose(params())

        const dataset = normalizeForVisualization(
            generateTwoMoons(nSamples(), noise())
        )
        setPoints(dataset.points)

        // Store raw normalized data for training (unpadded 0-1 range)
        trainingX = dataset.X
        trainingY = dataset.y

        // Create jax arrays for training (optimization for full batch)
        X = np.array(dataset.X)
        y = np.array(dataset.y.map(v => [v]))

        // Initialize new network
        const newParams = initParams(16)
        optState = initOptimizer(newParams, learningRate(), optimizerType())
        setParams(newParams)
        setEpoch(0)
        setLoss(null)
        setDecisionGrid(null)
    }

    const runTraining = async () => {
        if (!params() || !optState || isTraining()) return

        setIsTraining(true)
        // Update batch size max when samples change
        if (batchSize() > nSamples() * 2) {
            setBatchSize(nSamples() * 2)
        }

        const totalEpochs = epochsPerRound()
        const bs = batchSize()
        const n = trainingX.length

        for (let i = 0; i < totalEpochs; i++) {
            // Shuffle indices
            const indices = Array.from({ length: n }, (_, k) => k)
            for (let k = n - 1; k > 0; k--) {
                const j = Math.floor(Math.random() * (k + 1))
                const temp = indices[k]
                indices[k] = indices[j]
                indices[j] = temp
            }

            let epochTotalLoss = 0
            let epochSamples = 0

            // Mini-batch training
            for (let j = 0; j < n; j += bs) {
                const currentBatchSize = Math.min(bs, n - j)

                let X_batch: np.Array
                let y_batch: np.Array

                // Optimization: Use pre-loaded full dataset if doing full batch
                if (bs >= n && X && y) {
                    X_batch = X.ref
                    y_batch = y.ref
                } else {
                    const batchIdx = indices.slice(j, j + bs)
                    // Create tensors for this batch using trainingX/Y (0-1 range)
                    X_batch = np.array(batchIdx.map(idx => trainingX[idx]))
                    y_batch = np.array(batchIdx.map(idx => [trainingY[idx]]))
                }

                const currentParams = params()!
                const result = trainStep(currentParams, optState!, X_batch, y_batch, learningRate(), optimizerType())

                // Note: trainStep consumes X_batch and y_batch (via valueAndGrad),
                // so we don't need to dispose them manually.

                // Update state
                optState = result.optState
                setParams(result.params)

                // Accumulate loss
                epochTotalLoss += result.loss * currentBatchSize
                epochSamples += currentBatchSize
            }

            setLoss(epochTotalLoss / epochSamples)

            setEpoch(e => e + 1)

            // Update decision boundary every 10 epochs
            if ((i + 1) % 10 === 0 || i === totalEpochs - 1) {
                // Ensure we call predictGrid with current params
                // predictGrid creates its own temporary X, so it's safe
                const grid = predictGrid(params()!, 40)
                setDecisionGrid(grid)
            }

            // Allow UI to update
            await new Promise(r => setTimeout(r, 0))
        }

        setIsTraining(false)
    }

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
                    <Show when={!isLoading()}>
                        <div class="badge badge-lg gap-2 bg-primary/20 border-primary/30 text-primary">
                            <div class="w-2 h-2 rounded-full bg-primary animate-pulse" />
                            {device()}
                        </div>
                    </Show>
                </div>

                <Show when={!isLoading()} fallback={
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
                                    <Show when={decisionGrid()}>
                                        <For each={decisionGrid()!}>
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
                                    <For each={points()}>
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
                                    <div class="stat-value text-3xl text-primary">{epoch()}</div>
                                </div>
                                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-4">
                                    <div class="stat-title text-xs opacity-60">Loss</div>
                                    <div class="stat-value text-3xl text-secondary">
                                        {loss() !== null ? loss()!.toFixed(4) : '—'}
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
                                                if (params()) {
                                                    optState = initOptimizer(params()!, learningRate(), 'adam')
                                                }
                                            }}
                                            disabled={isTraining()}
                                        />
                                        <input
                                            class="join-item btn btn-sm btn-outline px-6"
                                            type="radio"
                                            name="optimizer"
                                            aria-label="SGD"
                                            checked={optimizerType() === 'sgd'}
                                            onClick={() => {
                                                setOptimizerType('sgd')
                                                if (params()) {
                                                    optState = initOptimizer(params()!, learningRate(), 'sgd')
                                                }
                                            }}
                                            disabled={isTraining()}
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
                                        disabled={isTraining()}
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
                                        disabled={isTraining()}
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
                                        disabled={isTraining()}
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
                                        disabled={isTraining()}
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
                                    class={`btn btn-primary flex-1 shadow-lg shadow-primary/20 flex flex-row items-center gap-2 ${isTraining() ? 'cursor-not-allowed opacity-50' : ''}`}
                                    onClick={(e) => {
                                        // Use e.target to verify we didn't click the input
                                        const target = e.target as HTMLElement;
                                        if (target.tagName !== 'INPUT' && !isTraining()) {
                                            runTraining();
                                        }
                                    }}
                                    role="button"
                                    tabindex="0"
                                >
                                    <Show when={isTraining()} fallback={
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
                                    disabled={isTraining()}
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
