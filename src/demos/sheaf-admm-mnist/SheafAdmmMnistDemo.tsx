import { For, Show, createEffect, createMemo, createSignal, onMount, type JSX } from 'solid-js'
import { Pause, Play, RotateCcw, StepForward } from 'lucide-solid'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { MnistImage } from '../../components/mnist/MnistImage'
import { type PatchAgent, type SheafAdmmSnapshot } from './algorithm'
import { createSheafAdmmMnistStore, type ResidualPoint } from './store'

const paper = '#fdfbf7'
const ink = '#2d2d2d'
const muted = '#e5e0d8'
const red = '#ff4d4d'
const blue = '#2d5da1'
const orange = '#f59e0b'

const wobbly = '10px 7px 11px 8px / 8px 11px 7px 10px'
const wobblyLarge = '14px 9px 13px 10px / 9px 14px 10px 13px'

function formatResidual(value: number | undefined): string {
    if (value === undefined) return '--'
    if (value < 0.001) return value.toExponential(1)
    return value.toFixed(3)
}

function argmax(values: number[]): number {
    let best = 0
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i
    }
    return best
}

function softmax(values: number[], temperature = 0.65): number[] {
    const scaled = values.map(value => value / temperature)
    const max = Math.max(...scaled)
    const exp = scaled.map(value => Math.exp(value - max))
    const sum = exp.reduce((acc, value) => acc + value, 0)
    return exp.map(value => value / Math.max(sum, 1e-9))
}

function confidence(snapshot: SheafAdmmSnapshot | null): number {
    if (!snapshot) return 0
    return Math.max(...snapshot.probabilities)
}

function edgeColor(mismatch: number): string {
    if (mismatch > 0.2) return red
    if (mismatch > 0.1) return orange
    if (mismatch > 0.05) return blue
    return '#39a86b'
}

function Latex(props: { expr: string; displayMode?: boolean; class?: string }): JSX.Element {
    let ref!: HTMLSpanElement

    createEffect(() => {
        katex.render(props.expr, ref, {
            displayMode: props.displayMode ?? false,
            throwOnError: false,
            strict: false,
        })
    })

    return <span ref={ref} class={props.class} />
}

function Panel(props: { title?: string; children: JSX.Element; class?: string; bodyClass?: string }): JSX.Element {
    return (
        <section
            class={`min-h-0 overflow-hidden border-2 border-[#2d2d2d] bg-white/70 ${props.class ?? ''}`}
            style={{
                'border-radius': wobblyLarge,
                'box-shadow': '2px 2px 0 rgba(45,45,45,0.08)',
            }}
        >
            <Show when={props.title}>
                <h2 class="px-4 pt-3 text-[18px] font-bold uppercase tracking-[0.02em] leading-none">
                    {props.title}
                </h2>
            </Show>
            <div class={props.bodyClass ?? 'p-4'}>
                {props.children}
            </div>
        </section>
    )
}

function SidebarSection(props: { title: string; children: JSX.Element }): JSX.Element {
    return (
        <div class="border-t border-[#2d2d2d]/20 pt-3">
            <h3 class="mb-2 text-[14px] font-bold uppercase tracking-[0.06em]">{props.title}</h3>
            {props.children}
        </div>
    )
}

function StatusRow(props: { label: string; value: JSX.Element | string; tone?: string }): JSX.Element {
    return (
        <div class="grid grid-cols-[1fr_auto] items-baseline gap-3 text-[15px] leading-tight">
            <span>{props.label}</span>
            <span class={`font-mono text-[17px] font-bold ${props.tone ?? ''}`}>{props.value}</span>
        </div>
    )
}

function ControlSlider(props: {
    label: string
    value: number
    min: number
    max: number
    step: number
    disabled?: boolean
    onChange: (value: number) => void
    valueText?: string
}): JSX.Element {
    return (
        <label class="grid gap-1 text-[14px]">
            <div class="grid grid-cols-[1fr_auto] items-center gap-2">
                <span>{props.label}</span>
                <span
                    class="min-w-9 rounded border border-[#2d2d2d]/50 bg-white px-2 py-0.5 text-center font-mono text-[14px] font-bold"
                    style={{ 'border-radius': wobbly }}
                >
                    {props.valueText ?? props.value}
                </span>
            </div>
            <input
                type="range"
                value={props.value}
                min={props.min}
                max={props.max}
                step={props.step}
                disabled={props.disabled}
                class="range range-sm range-primary"
                onInput={(event) => props.onChange(Number.parseFloat(event.currentTarget.value))}
            />
        </label>
    )
}

function Sidebar(props: {
    latest: SheafAdmmSnapshot | null
    modelStatus: string
    modelMessage: string | null
    sampleIndex: number
    datasetCount: number
    maxIterations: number
    diffusionSteps: number
    rho: number
    gamma: number
    displayMode: 'states' | 'edges'
    isRunning: boolean
    isReady: boolean
    isComplete: boolean
    onSample: (index: number) => void
    onIterations: (value: number) => void
    onDiffusion: (value: number) => void
    onRho: (value: number) => void
    onGamma: (value: number) => void
    onDisplay: (value: 'states' | 'edges') => void
    onRun: () => void
    onStep: () => void
    onStop: () => void
    onReset: () => void
}): JSX.Element {
    return (
        <aside class="h-screen min-h-0 overflow-hidden border-r-2 border-[#2d2d2d] px-5 py-6">
            <h1 class="mb-4 whitespace-nowrap font-heading text-[26px] font-bold leading-none">Sheaf-ADMM MNIST</h1>
            <a href="/" class="mb-5 inline-flex items-center gap-2 text-[16px] hover:text-[#2d5da1]">
                <span class="text-xl leading-none">‹</span>
                Back to Demos
            </a>

            <SidebarSection title="Status">
                <div class="grid gap-2">
                    <StatusRow
                        label="Engine"
                        value={props.modelStatus === 'trained' ? 'trained' : 'fallback'}
                        tone={props.modelStatus === 'trained' ? 'text-[#2d9a55]' : 'text-[#f59e0b]'}
                    />
                    <StatusRow label="Iteration" value={`${props.latest?.iteration ?? 0} / ${props.maxIterations}`} tone="text-[#2d5da1]" />
                    <StatusRow label="Primal Residual" value={formatResidual(props.latest?.primalResidual)} tone="text-[#ff4d4d]" />
                    <StatusRow label="Cons. Residual" value={formatResidual(props.latest?.consensusResidual)} tone="text-[#2d5da1]" />
                    <StatusRow label="Predicted Digit" value={props.latest?.prediction ?? '--'} tone="text-[#2d9a55]" />
                    <StatusRow label="Confidence" value={`${(confidence(props.latest) * 100).toFixed(1)}%`} tone="text-[#2d9a55]" />
                </div>
            </SidebarSection>

            <SidebarSection title="Controls">
                <div class="grid gap-3">
                    <div class="grid grid-cols-[1fr_auto_auto] items-center gap-2 text-[14px]">
                        <span>Sample</span>
                        <button
                            class="h-7 w-9 border-2 border-[#2d2d2d] bg-white text-lg leading-none disabled:opacity-40"
                            style={{ 'border-radius': wobbly }}
                            disabled={!props.isReady || props.isRunning || props.sampleIndex <= 0}
                            onClick={() => props.onSample(props.sampleIndex - 1)}
                        >
                            ‹
                        </button>
                        <button
                            class="h-7 w-9 border-2 border-[#2d2d2d] bg-white text-lg leading-none disabled:opacity-40"
                            style={{ 'border-radius': wobbly }}
                            disabled={!props.isReady || props.isRunning || props.sampleIndex >= props.datasetCount - 1}
                            onClick={() => props.onSample(props.sampleIndex + 1)}
                        >
                            ›
                        </button>
                        <span class="col-span-3 text-center font-mono text-[13px]">
                            {props.sampleIndex} / {Math.max(0, props.datasetCount - 1).toLocaleString()}
                        </span>
                    </div>

                    <ControlSlider
                        label="Iterations (K)"
                        value={props.maxIterations}
                        min={2}
                        max={40}
                        step={1}
                        disabled={props.isRunning}
                        onChange={props.onIterations}
                    />
                    <ControlSlider
                        label="Diffusion Steps (T)"
                        value={props.diffusionSteps}
                        min={1}
                        max={14}
                        step={1}
                        disabled={props.isRunning}
                        onChange={props.onDiffusion}
                    />
                    <ControlSlider
                        label="Rho (ρ)"
                        value={props.rho}
                        min={0.2}
                        max={3}
                        step={0.1}
                        disabled={props.isRunning}
                        onChange={props.onRho}
                        valueText={props.rho.toFixed(1)}
                    />
                    <ControlSlider
                        label="Gamma (γ)"
                        value={props.gamma}
                        min={0.01}
                        max={0.16}
                        step={0.01}
                        disabled={props.isRunning}
                        onChange={props.onGamma}
                        valueText={props.gamma.toFixed(2)}
                    />

                    <fieldset class="grid gap-1.5 text-[14px]">
                        <legend>Display</legend>
                        <label class="flex items-center gap-2">
                            <input
                                type="radio"
                                name="sheaf-display-mode"
                                checked={props.displayMode === 'states'}
                                onChange={() => props.onDisplay('states')}
                            />
                            States (x, z, u)
                        </label>
                        <label class="flex items-center gap-2">
                            <input
                                type="radio"
                                name="sheaf-display-mode"
                                checked={props.displayMode === 'edges'}
                                onChange={() => props.onDisplay('edges')}
                            />
                            Edge Disagreement
                        </label>
                    </fieldset>

                    <div class="grid grid-cols-2 gap-2">
                        <button
                            class="flex h-10 items-center justify-center gap-2 border-2 border-[#2d2d2d] bg-[#2d5da1] text-[18px] text-white disabled:bg-[#e5e0d8] disabled:text-[#2d2d2d]/40"
                            style={{ 'border-radius': wobbly }}
                            disabled={!props.isReady || props.isRunning || props.isComplete}
                            onClick={() => props.onRun()}
                        >
                            <Play class="h-4 w-4" />
                            Run
                        </button>
                        <button
                            class="flex h-10 items-center justify-center gap-2 border-2 border-[#2d2d2d] bg-white text-[18px] disabled:opacity-40"
                            style={{ 'border-radius': wobbly }}
                            disabled={!props.isReady || props.isComplete}
                            onClick={() => props.isRunning ? props.onStop() : props.onStep()}
                        >
                            <Show when={props.isRunning} fallback={<StepForward class="h-4 w-4" />}>
                                <Pause class="h-4 w-4" />
                            </Show>
                            {props.isRunning ? 'Pause' : 'Step'}
                        </button>
                        <button
                            class="col-span-2 flex h-10 items-center justify-center gap-2 border-2 border-[#2d2d2d] bg-white text-[18px] disabled:opacity-40"
                            style={{ 'border-radius': wobbly }}
                            disabled={!props.isReady || props.isRunning}
                            onClick={() => props.onReset()}
                        >
                            <RotateCcw class="h-4 w-4" />
                            Reset
                        </button>
                    </div>
                </div>
            </SidebarSection>

            <p class="mt-5 text-[13px] leading-snug">
                {props.modelMessage ?? 'Sheaf-ADMM iterates local proposals (x), consensus projection (z), and dual accumulation (u).'}
            </p>
        </aside>
    )
}

function InputPanel(props: {
    pixels: Float32Array
    agents: PatchAgent[]
    snapshot: SheafAdmmSnapshot
}): JSX.Element {
    const gridSize = createMemo(() => Math.max(1, Math.round(Math.sqrt(props.agents.length || 1))))
    const cellSize = createMemo(() => 280 / gridSize())

    return (
        <Panel title="1) INPUT: MNIST DIGIT (28x28) -> PATCH AGENTS" class="min-h-0" bodyClass="min-h-0 p-4">
            <div class="flex h-full min-h-0 flex-col items-center justify-between gap-3">
                <div class="relative aspect-square w-full max-w-[310px] overflow-hidden bg-black">
                    <MnistImage pixels={props.pixels} size={310} class="block h-full w-full" />
                    <svg class="absolute inset-0 h-full w-full" viewBox="0 0 280 280" aria-label="MNIST patch grid">
                        <For each={props.agents}>
                            {(agent) => {
                                const x = () => agent.col * cellSize()
                                const y = () => agent.row * cellSize()
                                const state = () => props.snapshot.z[agent.id]?.[props.snapshot.prediction] ?? 0
                                return (
                                    <g>
                                        <rect
                                            x={x()}
                                            y={y()}
                                            width={cellSize()}
                                            height={cellSize()}
                                            fill={state() > 0 ? 'rgba(45,93,161,0.16)' : 'transparent'}
                                            stroke="rgba(255,255,255,0.42)"
                                            stroke-width="1"
                                        />
                                        <text x={x() + 4} y={y() + 11} fill="white" font-size="8" font-family="monospace">
                                            {agent.id}
                                        </text>
                                    </g>
                                )
                            }}
                        </For>
                    </svg>
                </div>
                <div class="grid w-full max-w-[310px] grid-cols-[auto_1fr_auto] items-center gap-2 text-[14px]">
                    <span>0</span>
                    <div class="h-3 border border-[#2d2d2d]/40 bg-gradient-to-r from-black to-white" />
                    <span>1</span>
                </div>
                <p class="text-center text-[16px] leading-snug">
                    Each trained agent sees a local 3x3 patch.<br />
                    No single agent can identify the digit.
                </p>
            </div>
        </Panel>
    )
}

function NetworkPanel(props: {
    agents: PatchAgent[]
    snapshot: SheafAdmmSnapshot
    diffusionSteps: number
    rho: number
    gamma: number
}): JSX.Element {
    const maxMismatch = createMemo(() => Math.max(0.001, ...props.snapshot.edges.map(edge => edge.mismatch)))
    const gridSize = createMemo(() => Math.max(1, Math.round(Math.sqrt(props.agents.length || 1))))
    const spacing = createMemo(() => gridSize() > 1 ? 320 / (gridSize() - 1) : 0)

    return (
        <Panel title="2) SHEAF CONSENSUS NETWORK" class="min-h-0" bodyClass="grid h-[calc(100%-34px)] min-h-0 grid-cols-[minmax(0,1fr)_122px] gap-3 p-4">
            <div class="flex min-h-0 flex-col justify-between gap-2">
                <svg viewBox="0 0 380 380" class="min-h-0 flex-1" aria-label="Sheaf consensus network">
                    <For each={props.snapshot.edges}>
                        {(edge) => {
                            const from = () => props.agents[edge.from]
                            const to = () => props.agents[edge.to]
                            const x1 = () => 30 + from().col * spacing()
                            const y1 = () => 30 + from().row * spacing()
                            const x2 = () => 30 + to().col * spacing()
                            const y2 = () => 30 + to().row * spacing()
                            const intensity = () => edge.mismatch / maxMismatch()
                            return (
                                <line
                                    x1={x1()}
                                    y1={y1()}
                                    x2={x2()}
                                    y2={y2()}
                                    stroke={edgeColor(edge.mismatch)}
                                    stroke-width={2 + intensity() * 2}
                                    opacity={0.72}
                                    stroke-linecap="round"
                                />
                            )
                        }}
                    </For>
                    <For each={props.agents}>
                        {(agent) => (
                            <g>
                                <circle
                                    cx={30 + agent.col * spacing()}
                                    cy={30 + agent.row * spacing()}
                                    r={gridSize() > 7 ? 10 : 14}
                                    fill={paper}
                                    stroke={ink}
                                    stroke-width="2"
                                />
                                <text
                                    x={30 + agent.col * spacing()}
                                    y={34 + agent.row * spacing()}
                                    text-anchor="middle"
                                    fill={ink}
                                    font-size={gridSize() > 7 ? '9' : '13'}
                                    font-family="monospace"
                                >
                                    {agent.id}
                                </text>
                            </g>
                        )}
                    </For>
                </svg>
                <div class="grid grid-cols-4 gap-2 rounded border border-[#2d2d2d]/25 bg-[#fdfbf7] px-3 py-2 text-center text-[15px]" style={{ 'border-radius': wobbly }}>
                    <span>ADMM iter: {props.snapshot.iteration}</span>
                    <span>T = {props.diffusionSteps}</span>
                    <span>γ = {props.gamma.toFixed(2)}</span>
                    <span>ρ = {props.rho.toFixed(1)}</span>
                </div>
            </div>
            <div class="flex min-w-0 flex-col justify-center gap-3 text-[14px]">
                <div class="font-bold">Edge Disagreement</div>
                <Latex expr="\\|F_{ij}z_i-F_{ji}z_j\\|_2" class="max-w-full overflow-hidden whitespace-nowrap text-[12px]" />
                <LegendLine color={red} label="> 0.20" />
                <LegendLine color={orange} label="0.10 - 0.20" />
                <LegendLine color={blue} label="0.05 - 0.10" />
                <LegendLine color="#39a86b" label="< 0.05" />
                <p class="mt-4 leading-snug">Edges show learned projection mismatch between neighbors.</p>
            </div>
        </Panel>
    )
}

function LegendLine(props: { color: string; label: string }): JSX.Element {
    return (
        <div class="grid grid-cols-[28px_1fr] items-center gap-2">
            <span class="h-1.5 rounded" style={{ background: props.color }} />
            <span>{props.label}</span>
        </div>
    )
}

function ProbabilityChart(props: { probabilities: number[]; prediction: number }): JSX.Element {
    const width = 420
    const height = 160
    const chartLeft = 35
    const chartBottom = 132
    const probabilities = createMemo(() => props.probabilities)
    const max = createMemo(() => Math.max(0.01, ...probabilities()))

    return (
        <svg viewBox={`0 0 ${width} ${height}`} class="h-[160px] w-full" aria-label="Class probabilities chart">
            <text x={width / 2} y="18" text-anchor="middle" font-size="15" fill={ink}>
                Mean decoded agent probabilities
            </text>
            <line x1={chartLeft} y1={chartBottom} x2={width - 12} y2={chartBottom} stroke={ink} stroke-width="1" />
            <line x1={chartLeft} y1="32" x2={chartLeft} y2={chartBottom} stroke={ink} stroke-width="1" />
            <text x="7" y="38" font-size="13">1.0</text>
            <text x="10" y="84" font-size="13">0.5</text>
            <text x="13" y="137" font-size="13">0.0</text>
            <For each={probabilities()}>
                {(prob, index) => {
                    const digit = () => index()
                    const barX = () => chartLeft + 18 + digit() * 35
                    const barH = () => Math.max(4, (prob / max()) * 82)
                    return (
                        <g>
                            <rect
                                x={barX()}
                                y={chartBottom - barH()}
                                width="13"
                                height={barH()}
                                fill={digit() === props.prediction ? '#d9e8ff' : '#edf2fb'}
                                stroke={digit() === props.prediction ? blue : '#b8c3d8'}
                                stroke-width={digit() === props.prediction ? 2 : 1}
                            />
                            <text x={barX() + 6.5} y="151" text-anchor="middle" font-size="13" fill={ink}>
                                {digit()}
                            </text>
                            <Show when={digit() === props.prediction}>
                                <text x={barX() + 6.5} y={chartBottom - barH() - 9} text-anchor="middle" font-size="13" fill={blue} font-weight="700">
                                    {(prob * 100).toFixed(1)}%
                                </text>
                            </Show>
                        </g>
                    )
                }}
            </For>
        </svg>
    )
}

interface AgentComparisonRow {
    agentId: number
    localPred: number
    consensusPred: number
    localTopProb: number
    consensusTopProb: number
    delta: number
}

function makeAgentComparisons(snapshot: SheafAdmmSnapshot): AgentComparisonRow[] {
    return snapshot.x.map((local, agentId) => {
        const localProb = softmax(local)
        const consensusProb = softmax(snapshot.z[agentId] ?? local)
        const localPred = argmax(localProb)
        const consensusPred = argmax(consensusProb)
        const delta = localProb.reduce((acc, value, index) => acc + Math.abs(value - consensusProb[index]), 0)
        return {
            agentId,
            localPred,
            consensusPred,
            localTopProb: localProb[localPred],
            consensusTopProb: consensusProb[consensusPred],
            delta,
        }
    }).sort((a, b) => b.delta - a.delta).slice(0, 10)
}

function AgentComparisonTable(props: { snapshot: SheafAdmmSnapshot }): JSX.Element {
    const rows = createMemo(() => makeAgentComparisons(props.snapshot))

    return (
        <div>
            <h3 class="mb-2 text-[16px]">Local vs Consensus (Top 10 Agents by L1 shift)</h3>
            <div class="grid grid-cols-[44px_54px_72px_80px_1fr] gap-x-2 gap-y-1 text-[13px] leading-tight">
                <span>Agent</span>
                <span>Local</span>
                <span>Local Prob</span>
                <span>Cons. Prob</span>
                <span>Δ (L1)</span>
                <For each={rows()}>
                    {(row) => (
                        <>
                            <span class="font-mono">{row.agentId}</span>
                            <span class="font-mono text-[#ff4d4d]">{row.localPred}</span>
                            <MiniBar value={row.localTopProb} color="#ef7d67" label={row.localTopProb.toFixed(2)} />
                            <MiniBar value={row.consensusTopProb} color="#66b579" label={row.consensusTopProb.toFixed(2)} />
                            <span class="font-mono font-bold text-[#ff4d4d]">{row.delta.toFixed(2)}</span>
                        </>
                    )}
                </For>
            </div>
        </div>
    )
}

function MiniBar(props: { value: number; color: string; label: string }): JSX.Element {
    return (
        <span class="grid grid-cols-[32px_1fr] items-center gap-1">
            <span class="font-mono">{props.label}</span>
            <span class="h-2.5 border border-[#2d2d2d]/20 bg-white">
                <span class="block h-full" style={{ width: `${Math.max(4, props.value * 100)}%`, background: props.color }} />
            </span>
        </span>
    )
}

function ProbabilityPanel(props: { snapshot: SheafAdmmSnapshot }): JSX.Element {
    const conf = () => confidence(props.snapshot)

    return (
        <Panel title="3) CLASS PROBABILITIES (DIGIT)" class="min-h-0" bodyClass="grid h-[calc(100%-34px)] min-h-0 grid-rows-[170px_minmax(0,1fr)_44px] gap-2 p-4">
            <ProbabilityChart probabilities={props.snapshot.probabilities} prediction={props.snapshot.prediction} />
            <AgentComparisonTable snapshot={props.snapshot} />
            <div class="grid grid-cols-2 items-center rounded border border-[#2d2d2d]/30 px-4 text-[17px]" style={{ 'border-radius': wobbly }}>
                <span>Predicted digit: <b class="text-[#2d9a55]">{props.snapshot.prediction}</b></span>
                <span>Confidence: <b class="text-[#2d9a55]">{(conf() * 100).toFixed(1)}%</b></span>
            </div>
        </Panel>
    )
}

function ResidualChart(props: { points: ResidualPoint[]; maxIterations: number; currentIteration: number }): JSX.Element {
    const width = 720
    const height = 235
    const left = 55
    const right = 20
    const top = 22
    const bottom = 42

    const maxY = createMemo(() => Math.max(0.001, ...props.points.flatMap(point => [point.primal, point.consensus])))
    const minY = 1e-4
    const yScale = (value: number) => {
        const logMax = Math.log10(maxY())
        const logMin = Math.log10(minY)
        const logValue = Math.log10(Math.max(minY, value))
        const t = (logValue - logMin) / Math.max(0.001, logMax - logMin)
        return height - bottom - t * (height - top - bottom)
    }
    const xScale = (iteration: number) => left + (iteration / Math.max(1, props.maxIterations)) * (width - left - right)
    const pathFor = (key: 'primal' | 'consensus') => props.points.map((point, index) => {
        const x = xScale(point.iteration)
        const y = yScale(point[key])
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    }).join(' ')

    return (
        <Panel title="RESIDUALS OVER ITERATIONS" class="min-h-0" bodyClass="h-[calc(100%-34px)] min-h-0 p-3">
            <svg viewBox={`0 0 ${width} ${height}`} class="h-full w-full" aria-label="Residuals over iterations">
                <rect x="0" y="0" width={width} height={height} fill="transparent" />
                <For each={[1, 0.1, 0.01, 0.001, 0.0001]}>
                    {(tick) => {
                        const y = yScale(tick)
                        return (
                            <g>
                                <line x1={left} x2={width - right} y1={y} y2={y} stroke={muted} stroke-dasharray="4 6" />
                                <text x="12" y={y + 4} font-size="12" fill={ink}>
                                    {tick === 1 ? '10^0' : `10^${Math.round(Math.log10(tick))}`}
                                </text>
                            </g>
                        )
                    }}
                </For>
                <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} stroke={ink} />
                <line x1={left} x2={left} y1={top} y2={height - bottom} stroke={ink} />
                <For each={[0, 5, 10, 15, 20, 25, 30, 35, 40].filter(tick => tick <= props.maxIterations)}>
                    {(tick) => (
                        <text x={xScale(tick)} y={height - 16} text-anchor="middle" font-size="12" fill={ink}>{tick}</text>
                    )}
                </For>
                <path d={pathFor('primal')} fill="none" stroke="#d6452f" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                <path d={pathFor('consensus')} fill="none" stroke="#1461c9" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                <line x1={xScale(props.currentIteration)} x2={xScale(props.currentIteration)} y1={top} y2={height - bottom} stroke={ink} stroke-dasharray="5 5" />
                <text x={xScale(props.currentIteration) + 6} y={top + 12} font-size="12" fill={ink}>current</text>
                <circle cx={width - 210} cy="22" r="4" fill="#d6452f" />
                <text x={width - 200} y="27" font-size="13" fill={ink}>Primal Residual</text>
                <circle cx={width - 80} cy="22" r="4" fill="#1461c9" />
                <text x={width - 70} y="27" font-size="13" fill={ink}>Consensus Residual</text>
                <text x={width / 2} y={height - 2} text-anchor="middle" font-size="14" fill={ink}>Iteration (k)</text>
            </svg>
        </Panel>
    )
}

function ExplanationPanel(): JSX.Element {
    return (
        <Panel
            title="WHAT'S HAPPENING"
            class="min-h-0"
            bodyClass="grid h-[calc(100%-34px)] min-h-0 content-start gap-3 overflow-y-auto p-4 pr-3 text-[14px] leading-snug"
        >
            <p><b class="text-[#d6452f]">x-update:</b> each agent solves a local objective for <Latex expr="x_i^{k+1}" /> given <Latex expr="(z^k,u^k)" />.</p>
            <p><b class="text-[#1461c9]">z-update:</b> the trained model uses a CG projection to reduce <Latex expr="F_{ij}z_i-F_{ji}z_j" />.</p>
            <p><b class="text-[#f59e0b]">u-update:</b> accumulates disagreement with <Latex expr="u_i^{k+1}=u_i^k+x_i-z_i" />.</p>
            <p>The loaded checkpoint decodes every agent state and averages the agent softmax probabilities for the global digit.</p>
        </Panel>
    )
}

function StateLegendPanel(): JSX.Element {
    return (
        <Panel title="STATE LEGEND (per agent i)" class="min-h-0" bodyClass="grid min-h-0 gap-3 p-4 text-[14px] leading-snug">
            <LegendDot color="#d6452f" expr="x_i^k" text="Local proposal (primal)" />
            <LegendDot color="#1461c9" expr="z_i^k" text="Consensus state (after diffusion)" />
            <LegendDot color="#f59e0b" expr="u_i^k" text="Dual variable (accumulated error)" />
            <p>All states live in <Latex expr="R^{d_v}" />. Restriction maps <Latex expr="F_{ij}" /> project to edge spaces.</p>
        </Panel>
    )
}

function LegendDot(props: { color: string; expr: string; text: string }): JSX.Element {
    return (
        <div class="grid grid-cols-[18px_42px_1fr] items-center gap-2">
            <span class="h-3 w-3 rounded-full" style={{ background: props.color }} />
            <Latex expr={props.expr} />
            <span>{props.text}</span>
        </div>
    )
}

export function SheafAdmmMnistPage(): JSX.Element {
    return <SheafAdmmMnistDemo />
}

export function SheafAdmmMnistDemo(): JSX.Element {
    const store = createSheafAdmmMnistStore()
    const [displayMode, setDisplayMode] = createSignal<'states' | 'edges'>('states')
    const latest = createMemo(() => store.latest())
    const isComplete = createMemo(() => (latest()?.iteration ?? 0) >= store.maxIterations())

    onMount(() => {
        void store.initialize()
    })

    return (
        <div
            class="h-screen overflow-hidden text-[#2d2d2d]"
            style={{
                'background-color': paper,
                'background-image': `radial-gradient(${muted} 1px, transparent 1px)`,
                'background-size': '18px 18px',
            }}
        >
            <div class="grid h-screen min-w-[1280px] grid-cols-[280px_minmax(1000px,1fr)] border-2 border-[#2d2d2d] bg-transparent">
                <Sidebar
                    latest={latest()}
                    modelStatus={store.state.modelStatus}
                    modelMessage={store.state.modelMessage}
                    sampleIndex={store.sampleIndex()}
                    datasetCount={store.datasetCount()}
                    maxIterations={store.maxIterations()}
                    diffusionSteps={store.diffusionSteps()}
                    rho={store.rho()}
                    gamma={store.gamma()}
                    displayMode={displayMode()}
                    isRunning={store.state.isRunning}
                    isReady={store.state.loadStatus === 'ready'}
                    isComplete={isComplete()}
                    onSample={store.setSampleIndex}
                    onIterations={store.setMaxIterations}
                    onDiffusion={store.setDiffusionSteps}
                    onRho={store.setRho}
                    onGamma={store.setGamma}
                    onDisplay={setDisplayMode}
                    onRun={store.runToLimit}
                    onStep={store.stepOnce}
                    onStop={store.stop}
                    onReset={store.resetCurrent}
                />

                <main class="grid h-screen min-h-0 grid-rows-[620px_minmax(0,1fr)] gap-3 p-4">
                    <Show
                        when={store.state.loadStatus === 'ready' && latest()}
                        fallback={
                            <div class="flex items-center justify-center text-[24px]">
                                <Show when={store.state.error} fallback="Preparing MNIST patch agents...">
                                    {store.state.error}
                                </Show>
                            </div>
                        }
                    >
                        {(snapshot) => (
                            <>
                                <div class="grid min-h-0 grid-cols-[1fr_1.48fr_1.28fr] gap-2">
                                    <InputPanel pixels={store.state.currentPixels} agents={store.agents()} snapshot={snapshot()} />
                                    <NetworkPanel
                                        agents={store.agents()}
                                        snapshot={snapshot()}
                                        diffusionSteps={store.diffusionSteps()}
                                        rho={store.rho()}
                                        gamma={store.gamma()}
                                    />
                                    <ProbabilityPanel snapshot={snapshot()} />
                                </div>
                                <div class="grid min-h-0 grid-cols-[1.9fr_0.78fr_0.9fr] gap-0">
                                    <ResidualChart
                                        points={store.residualHistory()}
                                        maxIterations={store.maxIterations()}
                                        currentIteration={snapshot().iteration}
                                    />
                                    <ExplanationPanel />
                                    <StateLegendPanel />
                                </div>
                            </>
                        )}
                    </Show>
                </main>
            </div>
        </div>
    )
}
