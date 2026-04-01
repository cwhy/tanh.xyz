import { createEffect, createMemo, createSignal, For, Index, Show, onMount, onCleanup, type JSX } from 'solid-js'
import { SingleStageFullLayout } from '../../components/demo-layouts/SingleStageFull'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { createDamStore, type RecallFrame } from './store'

// ── Canvas rendering helpers ───────────────────────────────────────────

function renderPixelsToCanvas(
    canvas: HTMLCanvasElement,
    pixels: Float32Array,
    width: number,
    height: number,
    normalize = false
) {
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const imageData = ctx.createImageData(width, height)

    let min = 0
    let max = 1
    if (normalize) {
        min = Infinity
        max = -Infinity
        for (let i = 0; i < pixels.length; i++) {
            if (pixels[i] < min) min = pixels[i]
            if (pixels[i] > max) max = pixels[i]
        }
        if (max - min < 1e-6) { min = 0; max = 1 }
    }

    for (let i = 0; i < width * height; i++) {
        const val = normalize
            ? ((pixels[i] - min) / (max - min))
            : pixels[i]
        const v = Math.max(0, Math.min(255, Math.round(val * 255)))
        imageData.data[i * 4 + 0] = v
        imageData.data[i * 4 + 1] = v
        imageData.data[i * 4 + 2] = v
        imageData.data[i * 4 + 3] = 255
    }
    ctx.putImageData(imageData, 0, 0)
}

// ── Small image preview ────────────────────────────────────────────────

function ImagePreview(props: {
    pixels: Float32Array | null
    size?: number
    label?: string
    normalize?: boolean
    class?: string
}): JSX.Element {
    let canvas!: HTMLCanvasElement
    const size = () => props.size ?? 28

    createEffect(() => {
        if (props.pixels && canvas) {
            renderPixelsToCanvas(canvas, props.pixels, 28, 28, props.normalize)
        }
    })

    return (
        <div class={`relative inline-block ${props.class ?? ''}`}>
            <canvas
                ref={canvas}
                width={28}
                height={28}
                style={{
                    width: `${size()}px`,
                    height: `${size()}px`,
                    'image-rendering': 'pixelated',
                    'border-radius': '4px',
                }}
            />
            <Show when={props.label}>
                <div class="absolute left-0.5 top-0.5 rounded bg-black/60 px-1 py-0.5 text-[8px] text-white font-mono leading-none">
                    {props.label}
                </div>
            </Show>
        </div>
    )
}

// ── Hidden neuron activation bar ───────────────────────────────────────

function HiddenBar(props: {
    hVal: number
    hBinary: number
    index: number
    isActive: boolean
}): JSX.Element {
    return (
        <div class="flex items-center gap-1 group" title={`h${props.index} = ${props.hVal.toFixed(3)}`}>
            <span class="text-[9px] font-mono w-4 text-right opacity-50">{props.index}</span>
            <div class="flex-1 h-3 rounded-sm overflow-hidden" style={{ background: 'rgba(0,0,0,0.15)' }}>
                <div
                    class="h-full rounded-sm transition-all duration-150"
                    style={{
                        width: `${Math.max(0, Math.min(100, Math.abs(props.hVal) * 100))}%`,
                        background: props.isActive
                            ? 'linear-gradient(90deg, #10b981, #34d399)'
                            : 'linear-gradient(90deg, #6b7280, #9ca3af)',
                        opacity: props.isActive ? 1 : 0.4,
                    }}
                />
            </div>
            <span class={`text-[9px] font-mono w-3 ${props.isActive ? 'text-emerald-400 font-bold' : 'opacity-30'}`}>
                {props.isActive ? '1' : '0'}
            </span>
        </div>
    )
}

// ── MSE timeline ───────────────────────────────────────────────────────

function MsePlot(props: { frames: RecallFrame[]; currentStep: number; onStepClick: (step: number) => void }): JSX.Element {
    const maxMSE = createMemo(() => {
        if (props.frames.length === 0) return 1
        return Math.max(...props.frames.map(f => f.mseToTarget)) * 1.1
    })

    return (
        <div class="w-full h-full relative" style={{ 'min-height': '60px' }}>
            <svg viewBox={`0 0 ${props.frames.length} 100`} class="w-full h-full" preserveAspectRatio="none">
                {/* Area fill */}
                <path
                    d={(() => {
                        if (props.frames.length < 2) return ''
                        const pts = props.frames.map((f, i) =>
                            `${i},${100 - (f.mseToTarget / maxMSE()) * 100}`
                        )
                        return `M0,100 L${pts.join(' L')} L${props.frames.length - 1},100 Z`
                    })()}
                    fill="rgba(16, 185, 129, 0.15)"
                />
                {/* Line */}
                <polyline
                    points={props.frames.map((f, i) =>
                        `${i},${100 - (f.mseToTarget / maxMSE()) * 100}`
                    ).join(' ')}
                    fill="none"
                    stroke="#10b981"
                    stroke-width="1.5"
                    vector-effect="non-scaling-stroke"
                />
                {/* Current step marker */}
                <Show when={props.frames.length > 0}>
                    <line
                        x1={props.currentStep}
                        y1="0"
                        x2={props.currentStep}
                        y2="100"
                        stroke="#f59e0b"
                        stroke-width="1"
                        vector-effect="non-scaling-stroke"
                        stroke-dasharray="3,2"
                    />
                    <circle
                        cx={props.currentStep}
                        cy={100 - ((props.frames[props.currentStep]?.mseToTarget ?? 0) / maxMSE()) * 100}
                        r="3"
                        fill="#f59e0b"
                        vector-effect="non-scaling-stroke"
                    />
                </Show>
                {/* Click target (transparent rects for each step) */}
                <For each={props.frames}>
                    {(_, i) => (
                        <rect
                            x={i()}
                            y="0"
                            width="1"
                            height="100"
                            fill="transparent"
                            style={{ cursor: 'pointer' }}
                            onClick={() => props.onStepClick(i())}
                        />
                    )}
                </For>
            </svg>
        </div>
    )
}

// ── Training loss plot ─────────────────────────────────────────────────

function TrainLossPlot(props: { losses: number[] }): JSX.Element {
    const maxLoss = createMemo(() => {
        if (props.losses.length === 0) return 1
        return Math.max(...props.losses) * 1.05
    })

    return (
        <div class="w-full" style={{ height: '80px' }}>
            <svg viewBox={`0 0 ${Math.max(props.losses.length, 1)} 100`} class="w-full h-full" preserveAspectRatio="none">
                <polyline
                    points={props.losses.map((l, i) =>
                        `${i},${100 - (l / maxLoss()) * 100}`
                    ).join(' ')}
                    fill="none"
                    stroke="#818cf8"
                    stroke-width="1.5"
                    vector-effect="non-scaling-stroke"
                />
            </svg>
        </div>
    )
}

// ── Main Demo Component ────────────────────────────────────────────────

export function DenseAssociativeMemoryDemo() {
    const store = createDamStore()
    const { state } = store
    const [isAutoPlaying, setIsAutoPlaying] = createSignal(false)
    const [playSpeed, setPlaySpeed] = createSignal(150) // ms per frame

    onMount(() => { void store.initialize() })

    // Auto-play timer
    let playTimer: ReturnType<typeof setInterval> | null = null

    createEffect(() => {
        if (isAutoPlaying()) {
            playTimer = setInterval(() => {
                const next = state.currentRecallStep + 1
                if (next >= state.recallFrames.length) {
                    setIsAutoPlaying(false)
                } else {
                    store.setCurrentStep(next)
                }
            }, playSpeed())
        } else {
            if (playTimer) clearInterval(playTimer)
            playTimer = null
        }
    })

    onCleanup(() => { if (playTimer) clearInterval(playTimer) })

    const currentFrame = createMemo<RecallFrame | null>(() => {
        const frames = state.recallFrames
        const step = state.currentRecallStep
        return frames.length > 0 ? frames[step] ?? null : null
    })

    const handleRecall = (idx: number) => {
        setIsAutoPlaying(false)
        store.runRecall(idx)
        // Start auto-play from step 0
        store.setCurrentStep(0)
        setIsAutoPlaying(true)
    }

    // ── Visualization Panel ────────────────────────────────────────────

    const VisualizationPanel = () => (
        <div class="h-full p-4 md:p-6 overflow-y-auto">
            <Show
                when={!state.isLoading}
                fallback={
                    <div class="flex flex-col items-center justify-center h-full gap-3">
                        <span class="loading loading-spinner loading-lg" style={{ color: '#2d2d2d' }} />
                        <span style={{ color: '#2d2d2d', opacity: 0.6 }}>
                            {state.loadingMessage}
                            <br />
                            <span class="text-sm">Progress: {(state.loadingProgress * 100).toFixed(0)}%</span>
                        </span>
                    </div>
                }
            >
                <Show when={state.error}>
                    <div class="p-3 bg-red-100 border-2 border-red-400 text-red-800 rounded-lg mb-4 text-sm">
                        {state.error}
                    </div>
                </Show>

                <div class="flex flex-col gap-5">
                    {/* Recall visualization */}
                    <Show when={state.recallFrames.length > 0}>
                        <div class="flex flex-wrap gap-6">
                            {/* Main recall image + controls */}
                            <div class="flex flex-col gap-3 items-center">
                                <div class="flex gap-4 items-end">
                                    <div class="text-center">
                                        <div class="text-xs font-mono mb-1 opacity-60" style={{ color: '#2d2d2d' }}>Original</div>
                                        <ImagePreview pixels={state.originalImage} size={112} />
                                    </div>
                                    <div class="text-center">
                                        <div class="text-xs font-mono mb-1 opacity-60" style={{ color: '#2d2d2d' }}>Noisy Cue</div>
                                        <ImagePreview pixels={state.noisyCueImage} size={112} />
                                    </div>
                                    <div class="text-center">
                                        <div class="text-xs font-mono mb-1" style={{ color: '#f59e0b' }}>
                                            Step {state.currentRecallStep} / {state.recallFrames.length - 1}
                                        </div>
                                        <div style={{
                                            border: '3px solid #f59e0b',
                                            'border-radius': '6px',
                                            padding: '2px',
                                        }}>
                                            <ImagePreview pixels={currentFrame()?.visible ?? null} size={112} />
                                        </div>
                                    </div>
                                </div>

                                {/* Step slider + playback */}
                                <div class="flex items-center gap-2 w-full max-w-md">
                                    <button
                                        class="btn btn-xs btn-ghost"
                                        onClick={() => {
                                            if (isAutoPlaying()) {
                                                setIsAutoPlaying(false)
                                            } else {
                                                if (state.currentRecallStep >= state.recallFrames.length - 1) {
                                                    store.setCurrentStep(0)
                                                }
                                                setIsAutoPlaying(true)
                                            }
                                        }}
                                        style={{ color: '#2d2d2d' }}
                                    >
                                        {isAutoPlaying() ? '⏸' : '▶'}
                                    </button>
                                    <input
                                        type="range"
                                        min={0}
                                        max={state.recallFrames.length - 1}
                                        value={state.currentRecallStep}
                                        onInput={(e) => {
                                            setIsAutoPlaying(false)
                                            store.setCurrentStep(parseInt(e.currentTarget.value))
                                        }}
                                        class="flex-1"
                                    />
                                    <span class="text-xs font-mono opacity-60 w-20 text-right" style={{ color: '#2d2d2d' }}>
                                        MSE: {(currentFrame()?.mseToTarget ?? 0).toFixed(4)}
                                    </span>
                                </div>

                                {/* MSE plot */}
                                <div class="w-full max-w-md" style={{ height: '60px' }}>
                                    <MsePlot
                                        frames={state.recallFrames}
                                        currentStep={state.currentRecallStep}
                                        onStepClick={(s) => {
                                            setIsAutoPlaying(false)
                                            store.setCurrentStep(s)
                                        }}
                                    />
                                </div>
                            </div>

                            {/* Hidden neuron activations */}
                            <div class="flex-1 min-w-[200px] max-w-[280px]">
                                <div class="text-xs font-semibold mb-2 opacity-70" style={{ color: '#2d2d2d' }}>
                                    Hidden Neurons ({currentFrame()?.hActivated.reduce((a, b) => a + (b > 0.05 ? 1 : 0), 0) ?? 0}/{store.numHidden()} active)
                                </div>
                                <div class="flex flex-col gap-0.5 max-h-[300px] overflow-y-auto pr-1">
                                    <Show when={currentFrame()}>
                                        {(frame) => (
                                            <Index each={Array.from(frame().hActivated)}>
                                                {(ha, i) => (
                                                    <HiddenBar
                                                        hVal={ha()}
                                                        hBinary={frame().hBinary[i]}
                                                        index={i}
                                                        isActive={ha() > 0.05}
                                                    />
                                                )}
                                            </Index>
                                        )}
                                    </Show>
                                </div>
                            </div>
                        </div>
                    </Show>

                    {/* Basic memories grid */}
                    <Show when={state.basicMemoryImages.length > 0}>
                        <div>
                            <div class="text-sm font-semibold" style={{ color: '#2d2d2d' }}>
                                Basic Memories (ξ columns) — {state.basicMemoryImages.length} hidden neurons
                            </div>
                            
                            <div class="my-3 p-3 rounded-lg text-xs leading-relaxed" style={{ background: 'rgba(56, 189, 248, 0.1)', color: '#0369a1', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
                                <strong class="block mb-1 flex items-center gap-1.5">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                    Theory Note: Feature Compositionality
                                </strong>
                                Unlike standard Hopfield networks that store entire patterns as separate attractors, this high-capacity Dense Associative Memory learns a distributed, compositional representation. The basic memories ($\xi$ columns) resemble fractional pencil strokes or digit fragments rather than full numbers. During recall, the network reconstructs a complete memory by activating a sparse combination of these features, granting it exponential storage capacity.
                            </div>

                            <div class="flex flex-wrap gap-1.5">
                                <Index each={state.basicMemoryImages}>
                                    {(pixels, i) => {
                                        const isActive = () => currentFrame()?.hActivated[i] ?? 0
                                        return (
                                            <div style={{
                                                opacity: state.recallFrames.length > 0
                                                    ? (isActive() > 0.5 ? 1 : 0.3)
                                                    : 1,
                                                transition: 'opacity 0.2s',
                                                border: isActive() > 0.5 ? '2px solid #10b981' : '2px solid transparent',
                                                'border-radius': '4px',
                                            }}>
                                                <ImagePreview
                                                    pixels={pixels()}
                                                    size={42}
                                                    label={String(i)}
                                                    normalize={true}
                                                />
                                            </div>
                                        )
                                    }}
                                </Index>
                            </div>
                        </div>
                    </Show>

                    {/* Stored memories grid (pick any to recall) */}
                    <Show when={state.numMemoriesStored > 0 && !state.isTraining}>
                        <div>
                            <div class="text-sm font-semibold mb-2" style={{ color: '#2d2d2d' }}>
                                Stored Memories — click to recall (showing first 100)
                            </div>
                            <div class="flex flex-wrap gap-1">
                                <Index each={Array.from({ length: Math.min(100, state.numMemoriesStored) }, (_, i) => i)}>
                                    {(idx) => (
                                        <button
                                            onClick={() => handleRecall(idx())}
                                            style={{
                                                border: state.selectedCueIndex === idx()
                                                    ? '2px solid #f59e0b'
                                                    : '2px solid transparent',
                                                'border-radius': '4px',
                                                padding: '0',
                                                background: 'none',
                                                cursor: 'pointer',
                                            }}
                                            title={`Memory #${idx()} (label: ${state.memoryLabels[idx()]})`}
                                        >
                                            <ImagePreview
                                                pixels={store.getMemoryPixels(idx())}
                                                size={34}
                                                label={String(state.memoryLabels[idx()])}
                                            />
                                        </button>
                                    )}
                                </Index>
                            </div>
                        </div>
                    </Show>
                </div>
            </Show>
        </div>
    )

    // ── Control Panel ──────────────────────────────────────────────────

    const ControlPanel = () => (
        <div class="h-full overflow-y-auto p-4 md:p-6 space-y-4" style={{ color: '#2d2d2d' }}>
            {/* Stats */}
            <div class="grid grid-cols-3 gap-2">
                <div class="p-2 rounded-lg text-center" style={{ background: 'rgba(0,0,0,0.05)' }}>
                    <div class="text-[10px] opacity-50">Memories</div>
                    <div class="text-lg font-bold" style={{ color: '#6366f1' }}>
                        {state.numMemoriesStored || '—'}
                    </div>
                </div>
                <div class="p-2 rounded-lg text-center" style={{ background: 'rgba(0,0,0,0.05)' }}>
                    <div class="text-[10px] opacity-50">Epochs</div>
                    <div class="text-lg font-bold" style={{ color: '#10b981' }}>
                        {state.trainEpochsDone || '—'}
                    </div>
                </div>
                <div class="p-2 rounded-lg text-center" style={{ background: 'rgba(0,0,0,0.05)' }}>
                    <div class="text-[10px] opacity-50">θ (threshold)</div>
                    <div class="text-lg font-bold" style={{ color: '#f59e0b' }}>
                        {state.learnedTheta.toFixed(3)}
                    </div>
                </div>
            </div>

            {/* Training loss */}
            <Show when={state.trainLosses.length > 0}>
                <div class="p-2 rounded-lg" style={{ background: 'rgba(0,0,0,0.05)' }}>
                    <div class="text-xs font-semibold mb-1 opacity-60">Training Loss</div>
                    <TrainLossPlot losses={state.trainLosses} />
                    <div class="text-xs font-mono text-right opacity-50 mt-1">
                        {state.trainLosses.length > 0
                            ? `Latest: ${state.trainLosses[state.trainLosses.length - 1].toFixed(5)}`
                            : ''}
                    </div>
                </div>
            </Show>

            {/* Network config */}
            <div class="space-y-3">
                <div class="text-xs font-bold uppercase tracking-wider opacity-40">Network</div>

                <ConfigSlider
                    label="Hidden Neurons"
                    value={store.numHidden()}
                    min={10}
                    max={200}
                    step={10}
                    onChange={v => store.setNumHidden(Math.round(v))}
                    disabled={state.isTraining}
                    minLabel="10"
                    maxLabel="200"
                />

                <ConfigSlider
                    label="Memories to Store"
                    value={store.numMemories()}
                    min={100}
                    max={60000}
                    step={100}
                    onChange={v => store.setNumMemories(Math.round(v))}
                    disabled={state.isTraining}
                    minLabel="100"
                    maxLabel="60000"
                />

                <ConfigSlider
                    label="Training Epochs"
                    value={store.trainEpochs()}
                    min={10}
                    max={2000}
                    step={50}
                    onChange={v => store.setTrainEpochs(Math.round(v))}
                    disabled={state.isTraining}
                    minLabel="10"
                    maxLabel="2000"
                />

                <ConfigSlider
                    label="Learning Rate"
                    value={store.learningRate()}
                    min={0.001}
                    max={0.02}
                    step={0.001}
                    onChange={store.setLearningRate}
                    disabled={state.isTraining}
                    minLabel="0.001"
                    maxLabel="0.02"
                    valueFormatter={v => v.toFixed(3)}
                />
            </div>

            {/* Train button */}
            <div class="flex gap-2">
                <button
                    class="btn flex-1 shadow-md"
                    style={{
                        background: '#6366f1',
                        color: 'white',
                        border: '2px solid #4f46e5',
                    }}
                    onClick={() => void store.startTraining()}
                    disabled={state.isTraining || state.isLoading}
                >
                    <Show when={state.isTraining} fallback="Learn Memories">
                        <span class="loading loading-spinner loading-sm" />
                        Training…
                    </Show>
                </button>
                <Show when={state.isTraining}>
                    <button
                        class="btn btn-outline"
                        style={{ 'border-color': '#2d2d2d', color: '#2d2d2d' }}
                        onClick={() => store.stopTraining()}
                    >
                        Stop
                    </button>
                </Show>
            </div>

            {/* Recall config */}
            <Show when={state.basicMemoryImages.length > 0}>
                <div class="space-y-3">
                    <div class="text-xs font-bold uppercase tracking-wider opacity-40">Recall</div>

                    <ConfigSlider
                        label="Noise Level (σ)"
                        value={store.noiseLevel()}
                        min={0}
                        max={2}
                        step={0.05}
                        onChange={store.setNoiseLevel}
                        minLabel="0"
                        maxLabel="2"
                        valueFormatter={v => v.toFixed(2)}
                    />

                    <ConfigSlider
                        label="Recall Threshold (θ)"
                        value={store.recallTheta() ?? state.learnedTheta}
                        min={0}
                        max={1}
                        step={0.005}
                        onChange={v => store.setRecallTheta(v)}
                        minLabel="0"
                        maxLabel="1"
                        valueFormatter={v => v.toFixed(3)}
                        helperText={`Learned: ${state.learnedTheta.toFixed(3)}`}
                    />

                    <ConfigSlider
                        label="τ_v / τ_h Ratio"
                        value={store.tauRatio()}
                        min={2}
                        max={50}
                        step={1}
                        onChange={v => store.setTauRatio(Math.round(v))}
                        minLabel="2"
                        maxLabel="50"
                    />

                    <ConfigSlider
                        label="Recall Steps"
                        value={store.recallSteps()}
                        min={5}
                        max={100}
                        step={5}
                        onChange={v => store.setRecallSteps(Math.round(v))}
                        minLabel="5"
                        maxLabel="100"
                    />

                    <ConfigSlider
                        label="Activation β (0=hard Θ)"
                        value={store.recallBeta()}
                        min={0}
                        max={50}
                        step={1}
                        onChange={v => store.setRecallBeta(Math.round(v))}
                        minLabel="0"
                        maxLabel="50"
                    />

                    <ConfigSlider
                        label="Playback Speed (ms)"
                        value={playSpeed()}
                        min={20}
                        max={500}
                        step={10}
                        onChange={v => setPlaySpeed(Math.round(v))}
                        minLabel="20"
                        maxLabel="500"
                    />

                    <button
                        class="btn w-full shadow-md"
                        style={{
                            background: '#f59e0b',
                            color: '#2d2d2d',
                            border: '2px solid #d97706',
                        }}
                        onClick={() => handleRecall(state.selectedCueIndex)}
                        disabled={state.isTraining || state.isRecalling}
                    >
                        Recall Selected ({state.selectedCueIndex})
                    </button>

                    <button
                        class="btn btn-outline w-full"
                        style={{ 'border-color': '#2d2d2d', color: '#2d2d2d' }}
                        onClick={() => {
                            const randomIdx = Math.floor(Math.random() * state.numMemoriesStored)
                            handleRecall(randomIdx)
                        }}
                        disabled={state.isTraining || state.isRecalling || state.numMemoriesStored === 0}
                    >
                        Random Recall
                    </button>
                </div>
            </Show>
        </div>
    )

    return (
        <SingleStageFullLayout
            title="Dense Associative Memory"
            subtitle="Biologically plausible memory with threshold nonlinearity and exponential capacity. Based on Kafraj, Krotov & Latham (2026)."
            backHref="/"
            backLabel="Back"
            config={ControlPanel}
            stage={VisualizationPanel}
        />
    )
}

export function DenseAssociativeMemoryPage() {
    return <DenseAssociativeMemoryDemo />
}
