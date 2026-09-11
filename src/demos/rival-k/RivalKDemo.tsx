import { For, Show, createSignal, onCleanup, onMount } from 'solid-js'
import { createRivalKStore } from './store'
import type { BasicDatasetDistribution } from '../../lib/datasets/basic'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { SingleStageLayout } from '../../components/demo-layouts/SingleStage'

const distributionOptions: Array<{ label: string; value: BasicDatasetDistribution }> = [
    { label: 'Two Moons', value: 'two-moons' },
    { label: 'Spiral', value: 'spiral' },
    { label: 'Checkerboard', value: 'checkerboard' },
    { label: 'Clusters', value: 'clusters' },
    { label: 'Uniform', value: 'uniform' },
]

const maturityBadge: Record<string, { bg: string; text: string; label: string }> = {
    kid: { bg: 'bg-sky-100', text: 'text-sky-700', label: 'kid' },
    k2: { bg: 'bg-amber-100', text: 'text-amber-700', label: 'K2' },
    adult: { bg: 'bg-emerald-100', text: 'text-emerald-700', label: 'adult' },
}

export function RivalKPage() {
    return <RivalKDemo />
}

export function RivalKDemo() {
    const store = createRivalKStore()
    const [pointCount, setPointCount] = createSignal(100)
    const [distribution, setDistribution] = createSignal<BasicDatasetDistribution>('two-moons')
    const [selectedClusterId, setSelectedClusterId] = createSignal<number | null>(null)

    const canvasSize = 760
    const [plotSize, setPlotSize] = createSignal(canvasSize)
    let stageCanvasContainerRef: HTMLDivElement | undefined

    onMount(() => {
        const updatePlotSize = () => {
            if (!stageCanvasContainerRef) return
            const { clientWidth, clientHeight } = stageCanvasContainerRef
            const size = Math.max(1, Math.floor(Math.min(clientWidth, clientHeight)))
            setPlotSize(size)
        }

        updatePlotSize()

        const resizeObserver = new ResizeObserver(updatePlotSize)
        resizeObserver.observe(stageCanvasContainerRef!)

        onCleanup(() => {
            resizeObserver.disconnect()
        })
    })

    function handleCanvasClick(e: MouseEvent) {
        const svg = e.currentTarget as SVGSVGElement
        const ctm = svg.getScreenCTM()
        if (!ctm) return

        const pt = svg.createSVGPoint()
        pt.x = e.clientX
        pt.y = e.clientY
        const local = pt.matrixTransform(ctm.inverse())

        const x = Math.max(0, Math.min(1, local.x / canvasSize))
        const y = Math.max(0, Math.min(1, 1 - local.y / canvasSize))
        store.addSinglePoint(x, y)
    }

    // Are we currently in the deferred-replay phase?
    const isInReplayPhase = () =>
        store.currentPointIndex() >= store.pendingPoints().length &&
        store.replayQueueLength() > 0

    // Pending points that haven't been consumed yet (initial batch only)
    const remainingPendingPoints = () =>
        store.pendingPoints().slice(store.currentPointIndex())

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-5">
            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Points</div>
                    <div class="stat-value text-2xl text-primary">{store.points().length}</div>
                </div>
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Clusters</div>
                    <div class="stat-value text-2xl text-secondary">{store.clusters().length}</div>
                </div>
            </div>

            <ConfigSlider
                label="K (Neighbor Count)"
                value={store.k()}
                min={1}
                max={20}
                step={1}
                onChange={(value) => store.setK(Math.round(value))}
                disabled={store.isStreaming()}
                minLabel="1"
                maxLabel="20"
                helperText="Clusters mature at K+2 points. Lower K = tighter medoid."
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="Stream Speed"
                value={store.streamSpeed()}
                min={50}
                max={1000}
                step={50}
                onChange={(value) => store.setStreamSpeed(Math.round(value))}
                minLabel="Fast (50ms)"
                maxLabel="Slow (1000ms)"
                valueFormatter={(value) => `${value}ms`}
                valueClass="label-text-alt text-accent font-mono"
                rangeClass="range range-accent range-sm"
            />

            <ConfigSlider
                label="Point Count"
                value={pointCount()}
                min={20}
                max={1000}
                step={10}
                onChange={(value) => setPointCount(Math.round(value))}
                disabled={store.isStreaming()}
                valueClass="label-text-alt text-info font-mono"
                rangeClass="range range-info range-sm"
            />

            <ConfigRadioGroup
                label="Distribution"
                name="distribution"
                value={distribution()}
                options={distributionOptions}
                onChange={setDistribution}
                disabled={store.isStreaming()}
            />

            <div class="form-control">
                <label class="label cursor-pointer justify-start gap-3">
                    <input
                        type="checkbox"
                        class="toggle toggle-warning"
                        checked={store.cancelReplay()}
                        onChange={(e) => store.setCancelReplay(e.currentTarget.checked)}
                        disabled={store.isStreaming()}
                    />
                    <span class="label-text">Defer Evicted Points</span>
                </label>
                <span class="text-xs text-base-content/50 pl-1">
                    When on, evicted points replay in additional rounds until none remain
                </span>
            </div>

            <div class="flex flex-col gap-3 pt-1">
                <button
                    class="btn btn-primary shadow-lg shadow-primary/20"
                    onClick={() => store.generateRandomPoints(pointCount(), distribution())}
                    disabled={store.isStreaming()}
                >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                    Generate {pointCount()} Points
                </button>

                <div class="flex gap-3">
                    <Show
                        when={!store.isStreaming()}
                        fallback={
                            <button class="btn btn-warning flex-1" onClick={() => store.stopStreaming()}>
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                Pause
                            </button>
                        }
                    >
                        <button
                            class="btn btn-success flex-1"
                            onClick={() => store.startStreaming()}
                            disabled={store.currentPointIndex() >= store.pendingPoints().length && store.replayQueueLength() === 0}
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Stream
                        </button>
                    </Show>

                    <button
                        class="btn btn-outline flex-1"
                        onClick={() => store.stepOnce()}
                        disabled={store.isStreaming() || (store.currentPointIndex() >= store.pendingPoints().length && store.replayQueueLength() === 0)}
                    >
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                        </svg>
                        Step
                    </button>
                </div>

                <button class="btn btn-outline btn-error" onClick={() => store.reset()}>
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Reset
                </button>
            </div>
        </div>
    )

    const StagePanel = () => (
        <div class="card h-full bg-base-200/60 backdrop-blur-xl shadow-2xl border border-base-content/5">
            <div class="card-body h-full min-h-0 p-4 md:p-6 space-y-4">
                <div
                    ref={stageCanvasContainerRef}
                    class="relative min-h-0 grow rounded-2xl overflow-hidden border border-base-content/10 bg-base-300/50 flex items-center justify-center"
                    style={{ 'min-height': '400px' }}
                >
                    <svg
                        width={plotSize()}
                        height={plotSize()}
                        viewBox={`0 0 ${canvasSize} ${canvasSize}`}
                        preserveAspectRatio="xMidYMid meet"
                        class="block cursor-crosshair"
                        onClick={handleCanvasClick}
                    >
                        <defs>
                            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                                <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(0,0,0,0.05)" stroke-width="1" />
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#grid)" />

                        {/* Remaining pending points (initial batch, not yet consumed) */}
                        <For each={remainingPendingPoints()}>
                            {(p) => (
                                <circle
                                    cx={p.x * canvasSize}
                                    cy={(1 - p.y) * canvasSize}
                                    r={3}
                                    fill="rgba(0,0,0,0.1)"
                                    style={{ 'pointer-events': 'none' }}
                                />
                            )}
                        </For>

                        {/* Deferred points in the current replay queue — shown as orange hollow circles */}
                        <For each={store.replayQueue()}>
                            {(p) => (
                                <circle
                                    cx={p.x * canvasSize}
                                    cy={(1 - p.y) * canvasSize}
                                    r={5}
                                    fill="none"
                                    stroke="#f97316"
                                    stroke-width={1.5}
                                    stroke-dasharray="3 2"
                                    style={{ 'pointer-events': 'none' }}
                                />
                            )}
                        </For>

                        {/* K-th nearest neighbor distance circles */}
                        <For each={store.clusters()}>
                            {(c) => {
                                if (c.medoidId === null || c.kNNDist === 0) return null
                                const medoidState = store.points().find(p => p.point.id === c.medoidId)
                                if (!medoidState) return null

                                return (
                                    <circle
                                        cx={medoidState.point.x * canvasSize}
                                        cy={(1 - medoidState.point.y) * canvasSize}
                                        r={c.kNNDist * canvasSize}
                                        fill="none"
                                        stroke={c.color}
                                        stroke-width="1.5"
                                        stroke-dasharray="4 4"
                                        stroke-opacity="0.6"
                                        style={{
                                            'pointer-events': 'none',
                                            'opacity': (selectedClusterId() === null || selectedClusterId() === c.id) ? 1 : 0.1,
                                            'transition': 'opacity 0.2s',
                                        }}
                                    />
                                )
                            }}
                        </For>

                        {/* Clustered points */}
                        <For each={store.points()}>
                            {(p) => {
                                const isSelected = () => selectedClusterId() === null || selectedClusterId() === p.clusterId
                                return (
                                    <g style={{ 'opacity': isSelected() ? 1 : 0.1, 'transition': 'opacity 0.2s' }}>
                                        <Show when={p.isMedoid}>
                                            <circle
                                                cx={p.point.x * canvasSize}
                                                cy={(1 - p.point.y) * canvasSize}
                                                r={14}
                                                fill="none"
                                                stroke={p.medoidColor}
                                                stroke-width={3}
                                            />
                                        </Show>
                                        <Show when={p.point.id === store.lastAddedPointId()}>
                                            <circle
                                                cx={p.point.x * canvasSize}
                                                cy={(1 - p.point.y) * canvasSize}
                                                r={12}
                                                fill="none"
                                                stroke="#10b981"
                                                stroke-width={3}
                                            />
                                        </Show>
                                        <Show when={p.point.id === store.lastEvictedPointId()}>
                                            <circle
                                                cx={p.point.x * canvasSize}
                                                cy={(1 - p.point.y) * canvasSize}
                                                r={14}
                                                fill="none"
                                                stroke="#f97316"
                                                stroke-width={3}
                                            />
                                        </Show>
                                        <circle
                                            cx={p.point.x * canvasSize}
                                            cy={(1 - p.point.y) * canvasSize}
                                            r={p.isMedoid ? 8 : 6}
                                            fill={p.color}
                                            stroke={p.isMedoid ? p.medoidColor : 'rgba(0,0,0,0.2)'}
                                            stroke-width={p.isMedoid ? 2 : 1}
                                        />
                                    </g>
                                )
                            }}
                        </For>
                    </svg>
                </div>

                {/* Deferred batch progress bar (replaces old progress bar) */}
                <Show when={store.cancelReplay() && (isInReplayPhase() || store.replayRound() > 0)}>
                    <div class="space-y-1">
                        <div class="flex justify-between text-xs text-base-content/60">
                            <span class="font-medium">
                                {store.replayQueueLength() > 0
                                    ? `Replay Round ${store.replayRound() + 1}`
                                    : 'Replay Complete'}
                            </span>
                            <span class={store.replayQueueLength() > 0 ? 'text-warning' : 'text-success'}>
                                {store.replayQueueLength() > 0
                                    ? `${store.replayRoundProcessed()} / ${store.replayRoundTotal()}`
                                    : 'done'}
                            </span>
                        </div>
                        <Show when={store.replayRoundTotal() > 0}>
                            <progress
                                class={`progress w-full ${store.replayQueueLength() > 0 ? 'progress-warning' : 'progress-success'}`}
                                value={store.replayRoundProcessed()}
                                max={store.replayRoundTotal()}
                            />
                        </Show>
                    </div>
                </Show>

                {/* Initial batch progress (only while consuming pending points) */}
                <Show when={store.pendingPoints().length > 0 && store.currentPointIndex() < store.pendingPoints().length}>
                    <div>
                        <div class="flex justify-between text-xs text-base-content/60 mb-1">
                            <span>Initial Points</span>
                            <span>
                                {store.currentPointIndex()} / {store.pendingPoints().length}
                            </span>
                        </div>
                        <progress
                            class="progress progress-primary w-full"
                            value={store.currentPointIndex()}
                            max={store.pendingPoints().length}
                        />
                    </div>
                </Show>

                <Show when={store.clusters().length > 0}>
                    <div class="flex flex-wrap gap-2 text-sm">
                        <For each={store.clusters()}>
                            {(cluster) => {
                                const badge = () => maturityBadge[cluster.maturity]
                                const minEvenness = Math.min(...store.clusters().filter(c => c.size > 1).map(c => c.evenness))
                                const isWorst = cluster.size > 1 && cluster.evenness === minEvenness && minEvenness < 1
                                const isSelected = selectedClusterId() === cluster.id

                                return (
                                    <span
                                        class={`flex items-center gap-2 px-2 py-1 rounded-lg border cursor-pointer select-none transition-colors ${isSelected ? 'ring ring-primary ring-offset-1 ring-offset-base-300 ' : ''
                                            }${isWorst ? 'border-error bg-error/10' : 'border-base-content/5 bg-base-300/50 hover:bg-base-300'
                                            }`}
                                        onClick={() => setSelectedClusterId(prev => prev === cluster.id ? null : cluster.id)}
                                    >
                                        <span class="w-4 h-4 rounded-full" style={{ 'background-color': cluster.color }} />
                                        <span class="text-base-content/70">
                                            C{cluster.id} ({cluster.size})
                                        </span>
                                        <span class={`text-[10px] font-bold px-1.5 py-0.5 rounded-full border ${badge().bg} ${badge().text}`}>
                                            {badge().label}
                                        </span>
                                        <span class={`text-[10px] font-mono ${isWorst ? 'text-error font-bold' : 'text-base-content/50'}`}>
                                            even {cluster.evenness.toFixed(2)}
                                        </span>
                                    </span>
                                )
                            }}
                        </For>
                    </div>
                </Show>
            </div>
        </div>
    )

    return (
        <SingleStageLayout
            title="2D Rival-K Clustering"
            subtitle="Online clustering with only K — clusters self-regulate via evenness-based eviction."
            backHref="/demo"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
