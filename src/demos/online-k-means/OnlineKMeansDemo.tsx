import { For, Show, createSignal, onCleanup, onMount } from 'solid-js'
import { createOnlineKMeansStore } from './store'
import type { AlgorithmType, DistanceMetric } from './algorithm'
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

const algorithmOptions: Array<{ label: string; value: AlgorithmType }> = [
    { label: 'K-Means', value: 'online-k-means' },
    { label: 'Hard WTA', value: 'hard-wta' },
    { label: 'SoftHebb', value: 'softhebb' },
]

const metricOptions: Array<{ label: string; value: DistanceMetric }> = [
    { label: 'Euclidean', value: 'euclidean' },
    { label: 'Angular', value: 'angular' },
]

// ── MathML Equation Strings ───────────────────────────────────────

function getEquationsHtml(algorithm: AlgorithmType, metric: DistanceMetric): string {
    const label = (text: string) =>
        `<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.05em;opacity:0.5;margin-top:0.5rem;">${text}</div>`

    const eq = (mathml: string) =>
        `<div style="margin:0.15rem 0 0.15rem 0.25rem;"><math xmlns="http://www.w3.org/1998/Math/MathML">${mathml}</math></div>`

    // Shared fragments
    const ksup = '<msup><mi>k</mi><mo>∗</mo></msup>'
    const eta = '<mi>η</mi>'
    const wk = '<msub><mi mathvariant="bold">w</mi><mi>k</mi></msub>'
    const wkstar = '<msub><mi mathvariant="bold">w</mi>' + ksup + '</msub>'
    const xbold = '<mi mathvariant="bold">x</mi>'
    const xhat = '<mover><mi mathvariant="bold">x</mi><mo>^</mo></mover>'
    const yk = '<msub><mi>y</mi><mi>k</mi></msub>'
    const uk = '<msub><mi>u</mi><mi>k</mi></msub>'
    const bias = '<msub><mi>w</mi><mrow><mn>0</mn><mi>k</mi></mrow></msub>'
    const delta = '<mi mathvariant="normal">Δ</mi>'

    // Bias update (shared by all SoftHebb variants)
    const biasUpdate = eq(
        `<mrow>${delta}${bias}<mo>=</mo>${eta}<mo>⋅</mo>` +
        `<msup><mi>e</mi><mrow><mo>−</mo>${bias}</mrow></msup>` +
        `<mo>⋅</mo><mo>(</mo>${yk}<mo>−</mo>` +
        `<msup><mi>e</mi><mrow>${bias}</mrow></msup>` +
        `<mo>)</mo></mrow>`
    )

    // ── K-Means ──
    if (algorithm === 'online-k-means') {
        if (metric === 'euclidean') {
            return (
                label('Assign') +
                eq(`<mrow>${ksup}<mo>=</mo><munder><mo>arg min</mo><mi>k</mi></munder>` +
                    `<mo>‖</mo>${xbold}<mo>−</mo>${wk}<mo>‖</mo></mrow>`) +
                label('Update') +
                eq(`<mrow>${wkstar}<mo>←</mo><mfrac><mn>1</mn><mi>n</mi></mfrac>` +
                    `<munderover><mo>∑</mo><mi>i</mi><mi>n</mi></munderover>${xbold}</mrow>`)
            )
        } else {
            return (
                label('Assign') +
                eq(`<mrow>${ksup}<mo>=</mo><munder><mo>arg max</mo><mi>k</mi></munder>` +
                    `<mo>(</mo>${wk}<mo>⋅</mo>${xhat}<mo>)</mo></mrow>`) +
                label('Update') +
                eq(`<mrow>${wkstar}<mo>←</mo><mi>normalize</mi><mo>(</mo>` +
                    `<munderover><mo>∑</mo><mi>i</mi><mi>n</mi></munderover>${xhat}<mo>)</mo></mrow>`)
            )
        }
    }

    // ── Hard WTA ──
    if (algorithm === 'hard-wta') {
        if (metric === 'euclidean') {
            return (
                label('Assign') +
                eq(`<mrow>${ksup}<mo>=</mo><munder><mo>arg min</mo><mi>k</mi></munder>` +
                    `<mo>‖</mo>${xbold}<mo>−</mo>${wk}<mo>‖</mo></mrow>`) +
                label('Update (winner only)') +
                eq(`<mrow>${delta}${wkstar}<mo>=</mo>${eta}` +
                    `<mo>(</mo>${xbold}<mo>−</mo>${wkstar}<mo>)</mo></mrow>`)
            )
        } else {
            return (
                label('Preactivation') +
                eq(`<mrow>${uk}<mo>=</mo>${wk}<mo>⋅</mo>${xhat}</mrow>`) +
                label('Assign') +
                eq(`<mrow>${ksup}<mo>=</mo><munder><mo>arg max</mo><mi>k</mi></munder><mspace width="0.2em"/>${uk}</mrow>`) +
                label('Update (winner only)') +
                eq(`<mrow>${delta}${wkstar}<mo>=</mo>${eta}` +
                    `<mo>(</mo>${xhat}<mo>−</mo><mi>u</mi><mo>⋅</mo>${wkstar}<mo>)</mo></mrow>`)
            )
        }
    }

    // ── SoftHebb ──
    if (metric === 'euclidean') {
        return (
            label('Similarity') +
            eq(`<mrow>${uk}<mo>=</mo><mo>−</mo><mo>‖</mo>${xbold}<mo>−</mo>${wk}<msup><mo>‖</mo><mn>2</mn></msup></mrow>`) +
            label('Soft competition') +
            eq(`<mrow>${yk}<mo>=</mo><msub><mi>softmax</mi><mi>b</mi></msub>` +
                `<mo>(</mo>${uk}<mo>+</mo>${bias}<mo>)</mo></mrow>`) +
            label('Weight update (all neurons)') +
            eq(`<mrow>${delta}${wk}<mo>=</mo>${eta}<mo>⋅</mo>${yk}` +
                `<mo>⋅</mo><mo>(</mo>${xbold}<mo>−</mo>${wk}<mo>)</mo></mrow>`) +
            label('Bias update') +
            biasUpdate
        )
    } else {
        return (
            label('Preactivation') +
            eq(`<mrow>${uk}<mo>=</mo>${wk}<mo>⋅</mo>${xhat}</mrow>`) +
            label('Soft competition') +
            eq(`<mrow>${yk}<mo>=</mo><msub><mi>softmax</mi><mi>b</mi></msub>` +
                `<mo>(</mo>${uk}<mo>+</mo>${bias}<mo>)</mo></mrow>`) +
            label('Weight update (all neurons)') +
            eq(`<mrow>${delta}${wk}<mo>=</mo>${eta}<mo>⋅</mo>${yk}` +
                `<mo>⋅</mo><mo>(</mo>${xhat}<mo>−</mo>${uk}<mo>⋅</mo>${wk}<mo>)</mo></mrow>`) +
            label('Bias update') +
            biasUpdate
        )
    }
}

// ── Component ─────────────────────────────────────────────────────

export function OnlineKMeansPage() {
    return <OnlineKMeansDemo />
}

export function OnlineKMeansDemo() {
    const store = createOnlineKMeansStore()
    const [pointCount, setPointCount] = createSignal(200)
    const [distribution, setDistribution] = createSignal<BasicDatasetDistribution>('clusters')
    const [selectedClusterId, setSelectedClusterId] = createSignal<number | null>(null)

    const canvasSize = 760
    const [plotSize, setPlotSize] = createSignal(canvasSize)
    let stageCanvasContainerRef: HTMLDivElement | undefined

    onMount(() => {
        const updatePlotSize = () => {
            if (!stageCanvasContainerRef) return
            const { clientWidth, clientHeight } = stageCanvasContainerRef
            setPlotSize(Math.max(1, Math.floor(Math.min(clientWidth, clientHeight))))
        }
        updatePlotSize()
        const ro = new ResizeObserver(updatePlotSize)
        ro.observe(stageCanvasContainerRef!)
        onCleanup(() => ro.disconnect())
    })

    function handleCanvasClick(e: MouseEvent) {
        const svg = e.currentTarget as SVGSVGElement
        const ctm = svg.getScreenCTM()
        if (!ctm) return
        const pt = svg.createSVGPoint()
        pt.x = e.clientX
        pt.y = e.clientY
        const local = pt.matrixTransform(ctm.inverse())
        store.addSinglePoint(
            Math.max(0, Math.min(1, local.x / canvasSize)),
            Math.max(0, Math.min(1, 1 - local.y / canvasSize)),
        )
    }

    const remainingPendingPoints = () => store.pendingPoints().slice(store.currentPointIndex())
    const isAngular = () => store.distanceMetric() === 'angular'

    const pageTitle = () => {
        const prefix = {
            'online-k-means': 'Online K-Means',
            'hard-wta': 'Hard WTA',
            'softhebb': 'SoftHebb',
        }[store.algorithm()]
        return `2D ${prefix}`
    }

    const pageSubtitle = () => {
        const metricLabel = isAngular() ? 'Angular (cosine)' : 'Euclidean'
        const algDesc = {
            'online-k-means': 'running mean',
            'hard-wta': 'η-rate, winner only',
            'softhebb': 'η-rate, soft competition',
        }[store.algorithm()]
        return `${metricLabel} distance · ${algDesc}`
    }

    function angularBoundaryLine(angle: number) {
        const ox = 0, oy = canvasSize
        const len = canvasSize * 2
        return { ox, oy, ex: ox + Math.cos(angle) * len, ey: oy - Math.sin(angle) * len }
    }

    const equationsHtml = () => getEquationsHtml(store.algorithm(), store.distanceMetric())

    // ── Config Panel ──

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-4">
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

            <ConfigRadioGroup
                label="Algorithm"
                name="algorithm"
                value={store.algorithm()}
                options={algorithmOptions}
                onChange={(v) => store.setAlgorithm(v)}
                disabled={store.isStreaming()}
            />

            <ConfigRadioGroup
                label="Distance Metric"
                name="metric"
                value={store.distanceMetric()}
                options={metricOptions}
                onChange={(v) => store.setDistanceMetric(v)}
                disabled={store.isStreaming()}
            />

            {/* Live update equations */}
            <div
                class="rounded-xl border border-base-content/8 bg-base-300/40 px-3 py-2"
                style={{ 'font-size': '0.9rem', 'line-height': '1.5' }}
                innerHTML={equationsHtml()}
            />

            <ConfigSlider
                label="K (Number of Clusters)"
                value={store.k()}
                min={2}
                max={20}
                step={1}
                onChange={(v) => store.setK(Math.round(v))}
                disabled={store.isStreaming()}
                minLabel="2"
                maxLabel="20"
                helperText="First K points seed the initial centroids."
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <Show when={store.algorithm() !== 'online-k-means'}>
                <ConfigSlider
                    label="Learning Rate (η)"
                    value={store.learningRate()}
                    min={0.001}
                    max={0.1}
                    step={0.001}
                    onChange={(v) => store.setLearningRate(v)}
                    minLabel="0.001"
                    maxLabel="0.1"
                    valueFormatter={(v) => v.toFixed(3)}
                    helperText="Controls how fast centroids move toward inputs."
                    valueClass="label-text-alt text-warning font-mono"
                    rangeClass="range range-warning range-sm"
                />
            </Show>

            <Show when={store.algorithm() === 'softhebb'}>
                <ConfigSlider
                    label="Softmax Base (b)"
                    value={store.softmaxBase()}
                    min={2}
                    max={1000}
                    step={1}
                    onChange={(v) => store.setSoftmaxBase(v)}
                    minLabel="2 (soft)"
                    maxLabel="1000 (hard)"
                    valueFormatter={(v) => v === Math.E ? 'e ≈ 2.72' : String(Math.round(v))}
                    helperText="Higher = more winner-take-all. b=e is standard softmax."
                    valueClass="label-text-alt text-secondary font-mono"
                    rangeClass="range range-secondary range-sm"
                />
            </Show>

            <ConfigSlider
                label="Stream Speed"
                value={store.streamSpeed()}
                min={10}
                max={1000}
                step={10}
                onChange={(v) => store.setStreamSpeed(Math.round(v))}
                minLabel="Fast (10ms)"
                maxLabel="Slow (1000ms)"
                valueFormatter={(v) => `${v}ms`}
                valueClass="label-text-alt text-accent font-mono"
                rangeClass="range range-accent range-sm"
            />

            <ConfigSlider
                label="Point Count"
                value={pointCount()}
                min={20}
                max={1000}
                step={10}
                onChange={(v) => setPointCount(Math.round(v))}
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
                            disabled={store.currentPointIndex() >= store.pendingPoints().length}
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
                        disabled={store.isStreaming() || store.currentPointIndex() >= store.pendingPoints().length}
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

    // ── Stage Panel ──

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
                            <pattern id="grid-kmeans" width="50" height="50" patternUnits="userSpaceOnUse">
                                <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(0,0,0,0.05)" stroke-width="1" />
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#grid-kmeans)" />

                        {/* Pending points */}
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

                        {/* Euclidean: radius circles */}
                        <Show when={!isAngular()}>
                            <For each={store.clusters()}>
                                {(c) => {
                                    if (c.radius === 0) return null
                                    return (
                                        <circle
                                            cx={c.centroidX * canvasSize}
                                            cy={(1 - c.centroidY) * canvasSize}
                                            r={c.radius * canvasSize}
                                            fill="none"
                                            stroke={c.color}
                                            stroke-width="1.5"
                                            stroke-dasharray="4 4"
                                            stroke-opacity="0.4"
                                            style={{
                                                'pointer-events': 'none',
                                                'opacity': (selectedClusterId() === null || selectedClusterId() === c.id) ? 1 : 0.1,
                                                'transition': 'opacity 0.2s',
                                            }}
                                        />
                                    )
                                }}
                            </For>
                        </Show>

                        {/* Angular: sector boundaries + direction rays */}
                        <Show when={isAngular()}>
                            <For each={store.angularBoundaries()}>
                                {(angle) => {
                                    const { ox, oy, ex, ey } = angularBoundaryLine(angle)
                                    return (
                                        <line
                                            x1={ox} y1={oy} x2={ex} y2={ey}
                                            stroke="rgba(0,0,0,0.15)"
                                            stroke-width="1.5"
                                            stroke-dasharray="6 4"
                                            style={{ 'pointer-events': 'none' }}
                                        />
                                    )
                                }}
                            </For>
                            <For each={store.clusters()}>
                                {(c) => {
                                    const ox = 0, oy = canvasSize
                                    const len = canvasSize * 1.5
                                    const isSelected = () => selectedClusterId() === null || selectedClusterId() === c.id
                                    return (
                                        <line
                                            x1={ox} y1={oy}
                                            x2={ox + c.centroidX * len}
                                            y2={oy - c.centroidY * len}
                                            stroke={c.color}
                                            stroke-width="1"
                                            stroke-opacity="0.35"
                                            style={{
                                                'pointer-events': 'none',
                                                'opacity': isSelected() ? 1 : 0.1,
                                                'transition': 'opacity 0.2s',
                                            }}
                                        />
                                    )
                                }}
                            </For>
                        </Show>

                        {/* Points */}
                        <For each={store.points()}>
                            {(p) => {
                                const isSelected = () => selectedClusterId() === null || selectedClusterId() === p.clusterId
                                return (
                                    <g style={{ 'opacity': isSelected() ? 1 : 0.1, 'transition': 'opacity 0.2s' }}>
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
                                        <circle
                                            cx={p.point.x * canvasSize}
                                            cy={(1 - p.point.y) * canvasSize}
                                            r={5}
                                            fill={p.color}
                                            stroke="rgba(0,0,0,0.15)"
                                            stroke-width={0.5}
                                        />
                                    </g>
                                )
                            }}
                        </For>

                        {/* Centroids */}
                        <For each={store.clusters()}>
                            {(c) => {
                                const cx = c.centroidX * canvasSize
                                const cy = (1 - c.centroidY) * canvasSize
                                const isSelected = () => selectedClusterId() === null || selectedClusterId() === c.id
                                const s = 10
                                return (
                                    <g style={{ 'opacity': isSelected() ? 1 : 0.15, 'transition': 'opacity 0.2s' }}>
                                        <polygon
                                            points={`${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`}
                                            fill={c.centroidColor}
                                            stroke="white"
                                            stroke-width={2}
                                        />
                                        <line x1={cx - 4} y1={cy} x2={cx + 4} y2={cy} stroke="white" stroke-width={1.5} />
                                        <line x1={cx} y1={cy - 4} x2={cx} y2={cy + 4} stroke="white" stroke-width={1.5} />
                                    </g>
                                )
                            }}
                        </For>
                    </svg>
                </div>

                {/* Progress */}
                <Show when={store.pendingPoints().length > 0 && store.currentPointIndex() < store.pendingPoints().length}>
                    <div>
                        <div class="flex justify-between text-xs text-base-content/60 mb-1">
                            <span>Points Streamed</span>
                            <span>{store.currentPointIndex()} / {store.pendingPoints().length}</span>
                        </div>
                        <progress
                            class="progress progress-primary w-full"
                            value={store.currentPointIndex()}
                            max={store.pendingPoints().length}
                        />
                    </div>
                </Show>

                {/* Cluster list */}
                <Show when={store.clusters().length > 0}>
                    <div class="flex flex-wrap gap-2 text-sm">
                        <For each={store.clusters()}>
                            {(cluster) => {
                                const isSelected = selectedClusterId() === cluster.id
                                return (
                                    <span
                                        class={`flex items-center gap-2 px-2 py-1 rounded-lg border cursor-pointer select-none transition-colors ${isSelected ? 'ring ring-primary ring-offset-1 ring-offset-base-300 ' : ''}border-base-content/5 bg-base-300/50 hover:bg-base-300`}
                                        onClick={() => setSelectedClusterId(prev => prev === cluster.id ? null : cluster.id)}
                                    >
                                        <span class="w-4 h-4 rounded-full" style={{ 'background-color': cluster.color }} />
                                        <span class="text-base-content/70">
                                            C{cluster.id} ({cluster.size})
                                            <Show when={isAngular()}>
                                                <span class="text-xs opacity-50 ml-1">
                                                    {(cluster.angle * 180 / Math.PI).toFixed(1)}°
                                                </span>
                                            </Show>
                                            <Show when={store.algorithm() === 'softhebb'}>
                                                <span class="text-xs opacity-50 ml-1" title="Cluster bias">
                                                    b:{cluster.bias.toFixed(3)}
                                                </span>
                                            </Show>
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
            title={pageTitle()}
            subtitle={pageSubtitle()}
            backHref="/"
            backLabel="Back to Demos"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
