import { For, Show, createMemo, onMount, type JSX } from 'solid-js'
import { Activity, FastForward, Pause, Play, RotateCcw, Search, Shuffle, StepForward } from 'lucide-solid'
import { SingleStageFullLayout } from '../../components/demo-layouts/SingleStageFull'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { MnistImage } from '../../components/mnist/MnistImage'
import { DIGIT_COUNT, type PatchAgent, type SheafAdmmSnapshot } from './algorithm'
import { createSheafAdmmMnistStore, type ResidualPoint } from './store'

const panelStyle = {
    'border-radius': '10px 7px 11px 8px / 8px 11px 7px 10px',
    'box-shadow': '3px 3px 0 0 rgba(45,45,45,0.10)',
}

const strongPanelStyle = {
    'border-radius': '12px 8px 14px 9px / 9px 14px 8px 12px',
    'box-shadow': '5px 5px 0 0 rgba(45,45,45,0.14)',
}

function digitColor(value: number): string {
    const clamped = Math.max(0, Math.min(1, value))
    const hue = 210 - clamped * 170
    return `hsl(${hue} 70% ${64 - clamped * 18}%)`
}

function formatResidual(value: number | undefined): string {
    if (value === undefined) return '—'
    if (value < 0.001) return value.toExponential(1)
    return value.toFixed(3)
}

function StatCard(props: { label: string; value: JSX.Element | string; tone?: string }): JSX.Element {
    return (
        <div class="border-2 border-base-content bg-base-100 p-2" style={panelStyle}>
            <div class="text-xs text-base-content/55">{props.label}</div>
            <div class={`text-xl font-bold leading-tight ${props.tone ?? 'text-base-content'}`}>{props.value}</div>
        </div>
    )
}

function AgentDigitPanel(props: {
    pixels: Float32Array
    agents: PatchAgent[]
    snapshot: SheafAdmmSnapshot
}): JSX.Element {
    const edgePressure = createMemo(() => {
        const pressure = Array.from({ length: props.agents.length }, () => 0)
        for (const edge of props.snapshot.edges) {
            pressure[edge.from] = Math.max(pressure[edge.from], edge.mismatch)
            pressure[edge.to] = Math.max(pressure[edge.to], edge.mismatch)
        }
        return pressure
    })

    return (
        <div class="border-2 border-base-content bg-base-100 p-4 min-h-0" style={strongPanelStyle}>
            <div class="mb-3 flex items-center justify-between gap-3">
                <div>
                    <h2 class="font-heading text-2xl leading-none">Local MNIST Agents</h2>
                    <p class="text-sm text-base-content/60">49 patch views, each too small to decide alone.</p>
                </div>
                <div class="border-2 border-dashed border-base-content/50 px-2 py-1 text-sm" style={panelStyle}>
                    patch 4x4
                </div>
            </div>

            <div class="flex flex-col items-center gap-4">
                <div class="relative h-[320px] w-[320px] overflow-hidden border-2 border-base-content bg-black" style={panelStyle}>
                    <MnistImage pixels={props.pixels} size={320} class="block" />
                    <svg class="absolute inset-0 h-full w-full" viewBox="0 0 280 280" aria-label="MNIST image patch agents">
                        <For each={props.agents}>
                            {(agent) => {
                                const x = agent.col * 40
                                const y = agent.row * 40
                                const pressure = () => Math.min(1, edgePressure()[agent.id] * 2.4)
                                const best = () => props.snapshot.z[agent.id]?.[props.snapshot.prediction] ?? 0
                                return (
                                    <g>
                                        <rect
                                            x={x + 1}
                                            y={y + 1}
                                            width={38}
                                            height={38}
                                            fill={digitColor((best() + 2) / 4)}
                                            opacity={0.13 + pressure() * 0.28}
                                            stroke={pressure() > 0.35 ? '#ff4d4d' : 'rgba(255,255,255,0.5)'}
                                            stroke-width={pressure() > 0.35 ? 2.2 : 1}
                                        />
                                    </g>
                                )
                            }}
                        </For>
                    </svg>
                </div>

                <div class="grid w-full grid-cols-3 gap-2 text-center text-sm">
                    <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                        local guess <span class="font-mono font-bold">{props.snapshot.localPrediction}</span>
                    </div>
                    <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                        consensus <span class="font-mono font-bold text-secondary">{props.snapshot.prediction}</span>
                    </div>
                    <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                        agents <span class="font-mono font-bold">{props.agents.length}</span>
                    </div>
                </div>
            </div>
        </div>
    )
}

function SheafNetworkPanel(props: {
    agents: PatchAgent[]
    snapshot: SheafAdmmSnapshot
}): JSX.Element {
    const maxMismatch = createMemo(() => Math.max(0.001, ...props.snapshot.edges.map(edge => edge.mismatch)))
    const nodeRadius = 6

    return (
        <div class="border-2 border-base-content bg-base-100 p-4 min-h-0" style={strongPanelStyle}>
            <div class="mb-3 flex items-center justify-between gap-3">
                <div>
                    <h2 class="font-heading text-2xl leading-none">Sheaf Consensus Graph</h2>
                    <p class="text-sm text-base-content/60">Edges compare learned projections, not full private state.</p>
                </div>
                <Activity class="h-6 w-6 text-accent" />
            </div>

            <svg viewBox="0 0 360 360" class="h-[390px] w-full max-w-[460px] mx-auto" aria-label="Sheaf consensus network">
                <rect x="8" y="8" width="344" height="344" fill="#fdfbf7" stroke="#e5e0d8" stroke-dasharray="5 6" />
                <For each={props.snapshot.edges}>
                    {(edge) => {
                        const from = () => props.agents[edge.from]
                        const to = () => props.agents[edge.to]
                        const x1 = () => 36 + from().col * 48
                        const y1 = () => 36 + from().row * 48
                        const x2 = () => 36 + to().col * 48
                        const y2 = () => 36 + to().row * 48
                        const intensity = () => Math.max(0.05, edge.mismatch / maxMismatch())
                        return (
                            <line
                                x1={x1()}
                                y1={y1()}
                                x2={x2()}
                                y2={y2()}
                                stroke={intensity() > 0.55 ? '#ff4d4d' : '#2d5da1'}
                                stroke-width={1 + intensity() * 4}
                                opacity={0.25 + intensity() * 0.6}
                                stroke-linecap="round"
                            />
                        )
                    }}
                </For>
                <For each={props.agents}>
                    {(agent) => {
                        const stateValue = () => props.snapshot.z[agent.id]?.[props.snapshot.prediction] ?? 0
                        return (
                            <circle
                                cx={36 + agent.col * 48}
                                cy={36 + agent.row * 48}
                                r={nodeRadius + Math.max(0, agent.ink) * 5}
                                fill={digitColor((stateValue() + 2) / 4)}
                                stroke="#2d2d2d"
                                stroke-width="1.6"
                            />
                        )
                    }}
                </For>
            </svg>

            <div class="grid grid-cols-2 gap-2 text-sm">
                <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                    primal residual <span class="font-mono text-primary">{formatResidual(props.snapshot.primalResidual)}</span>
                </div>
                <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                    sheaf residual <span class="font-mono text-accent">{formatResidual(props.snapshot.consensusResidual)}</span>
                </div>
            </div>
        </div>
    )
}

function ProbabilityPanel(props: {
    label: number | null
    snapshot: SheafAdmmSnapshot
}): JSX.Element {
    return (
        <div class="border-2 border-base-content bg-base-100 p-4 min-h-0" style={strongPanelStyle}>
            <div class="mb-3 flex items-center justify-between gap-3">
                <div>
                    <h2 class="font-heading text-2xl leading-none">Global Digit Vote</h2>
                    <p class="text-sm text-base-content/60">Average z-state after local negotiation.</p>
                </div>
                <div class="border-2 border-base-content bg-[#fff9c4] px-3 py-1 text-lg font-bold" style={panelStyle}>
                    {props.snapshot.prediction}
                </div>
            </div>

            <div class="space-y-2">
                <For each={Array.from({ length: DIGIT_COUNT }, (_, digit) => digit)}>
                    {(digit) => {
                        const value = () => props.snapshot.probabilities[digit] ?? 0
                        const isPrediction = () => digit === props.snapshot.prediction
                        const isLabel = () => digit === props.label
                        return (
                            <div class="grid grid-cols-[24px_minmax(0,1fr)_48px] items-center gap-2 text-sm">
                                <div class={`font-mono font-bold ${isPrediction() ? 'text-secondary' : ''}`}>{digit}</div>
                                <div class="h-5 overflow-hidden border border-base-content/20 bg-base-200" style={panelStyle}>
                                    <div
                                        class="h-full"
                                        style={{
                                            width: `${Math.max(2, value() * 100)}%`,
                                            background: isPrediction() ? '#2d5da1' : isLabel() ? '#ff4d4d' : '#e5e0d8',
                                        }}
                                    />
                                </div>
                                <div class="text-right font-mono text-xs">{(value() * 100).toFixed(1)}%</div>
                            </div>
                        )
                    }}
                </For>
            </div>

            <div class="mt-4 grid grid-cols-2 gap-2 text-sm">
                <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                    label <span class="font-mono font-bold text-accent">{props.label ?? '—'}</span>
                </div>
                <div class="border border-base-content/20 bg-base-200 px-2 py-1" style={panelStyle}>
                    dual memory <span class="font-mono text-secondary">{formatResidual(props.snapshot.dualEnergy)}</span>
                </div>
            </div>
        </div>
    )
}

function ResidualChart(props: { points: ResidualPoint[] }): JSX.Element {
    const width = 900
    const height = 140
    const padding = 18
    const pathFor = (key: 'primal' | 'consensus') => {
        const points = props.points
        if (points.length === 0) return ''
        const maxY = Math.max(0.001, ...points.flatMap(point => [point.primal, point.consensus]))
        return points.map((point, index) => {
            const x = padding + (index / Math.max(1, points.length - 1)) * (width - padding * 2)
            const y = height - padding - (point[key] / maxY) * (height - padding * 2)
            return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
        }).join(' ')
    }

    return (
        <div class="border-2 border-base-content bg-base-100 p-3" style={strongPanelStyle}>
            <div class="mb-2 flex items-center justify-between">
                <h3 class="font-heading text-xl leading-none">Residual Trace</h3>
                <div class="flex gap-4 text-xs">
                    <span class="font-mono text-primary">primal</span>
                    <span class="font-mono text-accent">sheaf</span>
                </div>
            </div>
            <svg viewBox={`0 0 ${width} ${height}`} class="h-[120px] w-full" aria-label="ADMM residual trace">
                <rect x="0" y="0" width={width} height={height} fill="#fdfbf7" />
                <For each={[0, 1, 2, 3]}>
                    {(tick) => (
                        <line
                            x1={padding}
                            x2={width - padding}
                            y1={padding + tick * 32}
                            y2={padding + tick * 32}
                            stroke="#e5e0d8"
                            stroke-dasharray="5 7"
                        />
                    )}
                </For>
                <path d={pathFor('primal')} fill="none" stroke="#662200" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                <path d={pathFor('consensus')} fill="none" stroke="#ff4d4d" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
        </div>
    )
}

export function SheafAdmmMnistPage(): JSX.Element {
    return <SheafAdmmMnistDemo />
}

export function SheafAdmmMnistDemo(): JSX.Element {
    const store = createSheafAdmmMnistStore()
    const latest = createMemo(() => store.latest())
    const isComplete = createMemo(() => (latest()?.iteration ?? 0) >= store.maxIterations())

    onMount(() => {
        void store.initialize()
    })

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-3">
            <div class="grid grid-cols-2 gap-3">
                <StatCard label="Iteration" value={`${latest()?.iteration ?? 0}/${store.maxIterations()}`} tone="text-primary" />
                <StatCard label="Prediction" value={latest()?.prediction ?? '—'} tone="text-secondary" />
                <StatCard label="Primal" value={formatResidual(latest()?.primalResidual)} tone="text-accent" />
                <StatCard label="Sheaf" value={formatResidual(latest()?.consensusResidual)} tone="text-blue-700" />
            </div>

            <Show when={store.state.loadStatus === 'loading'}>
                <div class="border-2 border-dashed border-base-content/50 bg-base-100 p-3 text-sm" style={panelStyle}>
                    <div class="font-bold">Loading MNIST</div>
                    <Show when={store.state.loadProgress}>
                        {(progress) => (
                            <>
                                <div class="text-base-content/60">{progress().stage}: {progress().file}</div>
                                <progress class="progress progress-primary w-full" value={progress().completedFiles} max={progress().totalFiles} />
                            </>
                        )}
                    </Show>
                </div>
            </Show>

            <Show when={store.state.error}>
                <div class="alert alert-error text-sm">{store.state.error}</div>
            </Show>

            <div class="grid grid-cols-2 gap-x-3 gap-y-2">
                <ConfigSlider
                    label="Sample"
                    value={store.sampleIndex()}
                    min={0}
                    max={Math.max(1, store.datasetCount() - 1)}
                    step={1}
                    onChange={store.setSampleIndex}
                    disabled={store.state.loadStatus !== 'ready' || store.state.isRunning}
                    minLabel="0"
                    maxLabel={String(Math.max(1, store.datasetCount() - 1))}
                    valueClass="label-text-alt text-primary font-mono"
                    rangeClass="range range-primary range-sm"
                />

                <ConfigSlider
                    label="Iterations"
                    value={store.maxIterations()}
                    min={2}
                    max={40}
                    step={1}
                    onChange={store.setMaxIterations}
                    disabled={store.state.isRunning}
                    minLabel="2"
                    maxLabel="40"
                    valueClass="label-text-alt text-secondary font-mono"
                    rangeClass="range range-secondary range-sm"
                />

                <ConfigSlider
                    label="Diffusion"
                    value={store.diffusionSteps()}
                    min={1}
                    max={14}
                    step={1}
                    onChange={store.setDiffusionSteps}
                    disabled={store.state.isRunning}
                    minLabel="1"
                    maxLabel="14"
                    valueClass="label-text-alt text-info font-mono"
                    rangeClass="range range-info range-sm"
                />

                <ConfigSlider
                    label="Rho"
                    value={store.rho()}
                    min={0.2}
                    max={3}
                    step={0.1}
                    onChange={store.setRho}
                    disabled={store.state.isRunning}
                    minLabel="0.2"
                    maxLabel="3.0"
                    valueFormatter={(value) => value.toFixed(1)}
                    valueClass="label-text-alt text-primary font-mono"
                    rangeClass="range range-primary range-sm"
                />

                <ConfigSlider
                    label="Gamma"
                    value={store.gamma()}
                    min={0.01}
                    max={0.16}
                    step={0.01}
                    onChange={store.setGamma}
                    disabled={store.state.isRunning}
                    minLabel="0.01"
                    maxLabel="0.16"
                    valueFormatter={(value) => value.toFixed(2)}
                    valueClass="label-text-alt text-accent font-mono"
                    rangeClass="range range-accent range-sm"
                />

                <ConfigSlider
                    label="Delay"
                    value={store.runDelayMs()}
                    min={0}
                    max={800}
                    step={40}
                    onChange={store.setRunDelayMs}
                    disabled={store.state.isRunning}
                    minLabel="0"
                    maxLabel="800"
                    valueFormatter={(value) => `${Math.round(value)}ms`}
                    valueClass="label-text-alt text-base-content/70 font-mono"
                    rangeClass="range range-sm"
                />
            </div>

            <div class="grid grid-cols-2 gap-3 pt-1">
                <button
                    class="btn btn-sm btn-primary border-2 border-base-content gap-2 shadow-[3px_3px_0_0_#2d2d2d]"
                    onClick={() => store.runToLimit()}
                    disabled={store.state.loadStatus !== 'ready' || store.state.isRunning || isComplete()}
                >
                    <Show when={store.runDelayMs() <= 20} fallback={<Play class="h-4 w-4" />}>
                        <FastForward class="h-4 w-4" />
                    </Show>
                    Run
                </button>
                <button
                    class="btn btn-sm btn-outline border-2 gap-2"
                    onClick={() => store.stop()}
                    disabled={!store.state.isRunning}
                >
                    <Pause class="h-4 w-4" />
                    Stop
                </button>
                <button
                    class="btn btn-sm btn-outline border-2 gap-2"
                    onClick={() => store.stepOnce()}
                    disabled={store.state.loadStatus !== 'ready' || store.state.isRunning || isComplete()}
                >
                    <StepForward class="h-4 w-4" />
                    Step
                </button>
                <button
                    class="btn btn-sm btn-outline border-2 gap-2"
                    onClick={() => store.resetCurrent()}
                    disabled={store.state.loadStatus !== 'ready' || store.state.isRunning}
                >
                    <RotateCcw class="h-4 w-4" />
                    Reset
                </button>
            </div>

            <div class="grid grid-cols-2 gap-3">
                <button
                    class="btn btn-sm btn-outline border-2 gap-2"
                    onClick={() => store.setSampleIndex(store.sampleIndex() + 1)}
                    disabled={store.state.loadStatus !== 'ready' || store.state.isRunning}
                >
                    <Shuffle class="h-4 w-4" />
                    Next
                </button>
                <button
                    class="btn btn-sm btn-outline border-2 gap-2"
                    onClick={() => store.pickNextMistake()}
                    disabled={store.state.loadStatus !== 'ready' || store.state.isRunning}
                >
                    <Search class="h-4 w-4" />
                    Stress
                </button>
            </div>

            <div class="border-2 border-dashed border-base-content/40 bg-base-100 p-2 text-xs text-base-content/65" style={panelStyle}>
                Each agent solves a local prototype objective. The z-update diffuses only projected logits across neighbor channels, while u stores the leftover disagreement.
            </div>
        </div>
    )

    const StagePanel = () => (
        <div class="min-h-0 pr-0 lg:h-full lg:pr-6">
            <Show
                when={store.state.loadStatus === 'ready' && latest()}
                fallback={
                    <div class="flex h-full flex-col items-center justify-center gap-4 text-center">
                        <span class="loading loading-spinner loading-lg text-primary" />
                        <span class="text-base-content/60">Preparing MNIST patch agents...</span>
                    </div>
                }
            >
                {(snapshot) => (
                    <div class="flex min-h-0 flex-col gap-4 lg:grid lg:h-full lg:grid-rows-[minmax(0,1fr)_170px]">
                        <div class="grid min-h-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(300px,0.95fr)_minmax(320px,1.1fr)_minmax(300px,0.95fr)]">
                            <AgentDigitPanel pixels={store.state.currentPixels} agents={store.agents()} snapshot={snapshot()} />
                            <SheafNetworkPanel agents={store.agents()} snapshot={snapshot()} />
                            <ProbabilityPanel label={store.state.currentLabel} snapshot={snapshot()} />
                        </div>
                        <ResidualChart points={store.residualHistory()} />
                    </div>
                )}
            </Show>
        </div>
    )

    return (
        <SingleStageFullLayout
            title="Sheaf-ADMM MNIST"
            subtitle="Limited-view digit agents negotiate a global class through ADMM-style consensus."
            backHref="/"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
