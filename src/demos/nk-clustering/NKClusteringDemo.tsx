import { For, Show, createSignal, onCleanup, onMount } from 'solid-js'
import { createNKClusteringStore } from './store'
import type { BasicDatasetDistribution } from '../../lib/datasets/basic'
import type { CenterType } from './algorithm'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { SingleStageLayout } from '../../components/demo-layouts/single-stage'

const distributionOptions: Array<{ label: string; value: BasicDatasetDistribution }> = [
    { label: 'Two Moons', value: 'two-moons' },
    { label: 'Spiral', value: 'spiral' },
    { label: 'Checkerboard', value: 'checkerboard' },
    { label: 'Clusters', value: 'clusters' },
    { label: 'Uniform', value: 'uniform' },
]

const centerTypeOptions: Array<{ label: string; value: CenterType }> = [
    { label: 'Medoid', value: 'medoid' },
    { label: 'Centroid', value: 'centroid' },
]

export function NKClusteringPage() {
    return <NKClusteringDemo />
}

export function NKClusteringDemo() {
    const store = createNKClusteringStore()
    const [pointCount, setPointCount] = createSignal(100)
    const [distribution, setDistribution] = createSignal<BasicDatasetDistribution>('two-moons')

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

            <Show when={store.cancelReplay() && store.replayQueueLength() > 0}>
                <div class="stat bg-warning/10 rounded-xl border border-warning/20 p-3">
                    <div class="stat-title text-xs opacity-60">Replay Queue</div>
                    <div class="stat-value text-xl text-warning">{store.replayQueueLength()}</div>
                </div>
            </Show>

            <ConfigSlider
                label="N (Max Cluster Size)"
                value={store.n()}
                min={3}
                max={50}
                step={1}
                onChange={(value) => store.setN(Math.round(value))}
                disabled={store.isStreaming()}
                minLabel="3"
                maxLabel="50"
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="K (Neighbors for Medoid)"
                value={store.k()}
                min={1}
                max={Math.max(1, store.n() - 1)}
                step={1}
                onChange={(value) => store.setK(Math.round(value))}
                disabled={store.isStreaming()}
                minLabel="1"
                maxLabel={String(Math.max(1, store.n() - 1))}
                helperText="K must be less than N. Lower K = denser medoid selection."
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
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
                max={600}
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

            <ConfigRadioGroup
                label="Center Type"
                name="centerType"
                value={store.centerType()}
                options={centerTypeOptions}
                onChange={store.setCenterType}
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

                        <For each={store.pendingPoints().slice(store.currentPointIndex())}>
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

                        <For each={store.points()}>
                            {(p) => (
                                <g>
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
                            )}
                        </For>
                    </svg>
                </div>

                <Show when={store.pendingPoints().length > 0}>
                    <div class="space-y-2">
                        <div>
                            <div class="flex justify-between text-xs text-base-content/60 mb-1">
                                <span>Progress</span>
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
                        <Show when={store.cancelReplay() && (store.replayQueueLength() > 0 || store.currentPointIndex() >= store.pendingPoints().length)}>
                            <div>
                                <div class="flex justify-between text-xs text-base-content/60 mb-1">
                                    <span>Deferred Points</span>
                                    <span class={store.replayQueueLength() > 0 ? 'text-warning' : 'text-success'}>
                                        {store.replayQueueLength() > 0 ? `${store.replayQueueLength()} remaining` : 'done'}
                                    </span>
                                </div>
                                <div class="h-2 w-full bg-base-300 rounded-full overflow-hidden">
                                    <div
                                        class={`h-full transition-all duration-300 ${store.replayQueueLength() > 0 ? 'bg-warning' : 'bg-success'}`}
                                        style={{ width: store.replayQueueLength() > 0 ? '100%' : '0%' }}
                                    />
                                </div>
                            </div>
                        </Show>
                    </div>
                </Show>

                <Show when={store.clusters().length > 0}>
                    <div class="flex flex-wrap gap-2 text-sm">
                        <For each={store.clusters()}>
                            {(cluster) => (
                                <span class="flex items-center gap-2 px-2 py-1 rounded-lg bg-base-300/50">
                                    <span class="w-4 h-4 rounded-full" style={{ 'background-color': cluster.color }} />
                                    <span class="text-base-content/70">
                                        C{cluster.id} ({cluster.size}/{store.n()})
                                    </span>
                                </span>
                            )}
                        </For>
                    </div>
                </Show>
            </div>
        </div>
    )

    return (
        <SingleStageLayout
            title="NK Clustering"
            subtitle="Online clustering with bounded cluster sizes. N = max size, K = neighbors for medoid."
            backHref="/"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
