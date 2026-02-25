import { For, Show, createMemo, createSignal, type JSX } from 'solid-js'
import { SingleStageFullLayout } from '../../components/demo-layouts/SingleStageFull'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { MnistImage } from '../../components/mnist/MnistImage'
import { createMnistNKStore, type ClusterDisplayInfo } from './store'

// ---------------------------------------------------------------------------
// Cluster card
// ---------------------------------------------------------------------------

function ClusterCard(props: {
    cluster: ClusterDisplayInfo
    getPixels: (dataIndex: number) => Float32Array
    imageSize: number
}): JSX.Element {
    return (
        <div
            class="rounded-2xl bg-white/80 backdrop-blur-sm shadow-sm border p-3 flex flex-col gap-2"
            style={{ 'border-color': props.cluster.color, 'border-width': '1.5px' }}
        >
            {/* Header */}
            <div class="flex items-center gap-2 flex-wrap min-w-0">
                <div
                    class="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ background: props.cluster.color }}
                />
                <span
                    class="font-bold text-sm"
                    style={{ color: '#2d2d2d', 'font-family': "'Patrick Hand', cursive" }}
                >
                    #{props.cluster.id}
                </span>
                <span
                    class="text-xs font-mono opacity-60"
                    style={{ color: '#2d2d2d' }}
                >
                    even-ness {props.cluster.evenness.toFixed(2)}
                </span>
                <span
                    class="text-xs opacity-50"
                    style={{ color: '#2d2d2d' }}
                >
                    {props.cluster.pointCount}/{props.cluster.maxSize}
                </span>
                <Show when={props.cluster.isTopHalf}>
                    <span class="text-xs font-bold px-1.5 py-0.5 rounded-full bg-amber-100 text-amber-700 border border-amber-200">
                        top half
                    </span>
                </Show>
                <Show when={props.cluster.isLastOne}>
                    <span class="text-xs font-bold px-1.5 py-0.5 rounded-full bg-red-100 text-red-700 border border-red-200">
                        last-1
                    </span>
                </Show>
                <Show when={props.cluster.isSaturated}>
                    <span class="text-xs font-bold px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-500 border border-gray-200">
                        saturated
                    </span>
                </Show>
            </div>

            {/* Image grid */}
            <div class="flex flex-wrap gap-1">
                <For each={props.cluster.pointsSortedByDist}>
                    {(entry) => {
                        const pixels = createMemo(() => props.getPixels(entry.dataIndex))
                        const ring = entry.isMedoid
                            ? `3px solid ${props.cluster.color}`
                            : entry.isKNN
                                ? '3px solid #3b82f6'
                                : '1.5px solid rgba(0,0,0,0.08)'

                        return (
                            <div
                                class="rounded overflow-hidden flex-shrink-0"
                                style={{
                                    outline: ring,
                                    'outline-offset': entry.isMedoid || entry.isKNN ? '-1px' : '0',
                                    'border-radius': '4px',
                                }}
                                title={
                                    entry.isMedoid
                                        ? `#${entry.dataIndex} — mode (center)`
                                        : entry.isKNN
                                            ? `#${entry.dataIndex} — k-th nearest neighbour`
                                            : `#${entry.dataIndex} — dist ${entry.distToMode.toFixed(3)}`
                                }
                            >
                                <MnistImage
                                    pixels={pixels()}
                                    size={props.imageSize}
                                />
                            </div>
                        )
                    }}
                </For>
            </div>

            {/* Legend */}
            <div class="flex gap-3 text-xs opacity-50" style={{ color: '#2d2d2d' }}>
                <span style={{ outline: `2px solid ${props.cluster.color}`, padding: '0 4px', 'border-radius': '2px' }}>
                    mode
                </span>
                <span style={{ outline: '2px solid #3b82f6', padding: '0 4px', 'border-radius': '2px' }}>
                    k-NN
                </span>
            </div>
        </div>
    )
}

// ---------------------------------------------------------------------------
// Main demo
// ---------------------------------------------------------------------------

export function MnistNkClusteringPage(): JSX.Element {
    return <MnistNkClusteringDemo />
}

export function MnistNkClusteringDemo(): JSX.Element {
    const store = createMnistNKStore()

    // Image size in pixels (display size, not canvas pixels)
    const imageSize = 28

    const [sortByEvenness, setSortByEvenness] = createSignal(true)

    const displayedClusters = createMemo(() => {
        const clusters = store.clusters()
        if (sortByEvenness()) {
            return [...clusters].sort((a, b) => b.evenness - a.evenness)
        }
        return clusters
    })

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-5">
            {/* Stats */}
            <Show when={store.loadStatus() === 'ready'}>
                <div class="grid grid-cols-2 gap-3">
                    <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                        <div class="stat-title text-xs opacity-60">Processed</div>
                        <div class="stat-value text-2xl text-primary">
                            {store.processedCount()}<span class="text-sm text-base-content/40">/{store.totalCount()}</span>
                        </div>
                    </div>
                    <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                        <div class="stat-title text-xs opacity-60">Clusters</div>
                        <div class="stat-value text-2xl text-secondary">{store.clusters().length}</div>
                    </div>
                </div>


            </Show>

            {/* Loading progress */}
            <Show when={store.loadStatus() === 'loading'}>
                <div class="rounded-xl bg-base-300/50 border border-base-content/5 p-3 space-y-2">
                    <div class="text-sm font-semibold opacity-70">Loading MNIST…</div>
                    <Show when={store.loadProgress()}>
                        {(p) => (
                            <div class="space-y-1">
                                <div class="text-xs opacity-50">
                                    {p().stage}: {p().file}
                                </div>
                                <progress
                                    class="progress progress-primary w-full"
                                    value={p().completedFiles}
                                    max={p().totalFiles}
                                />
                            </div>
                        )}
                    </Show>
                </div>
            </Show>

            {/* JAX device badge */}
            <Show when={store.jaxDevice()}>
                <div class="text-xs opacity-40 font-mono">
                    JAX backend: {store.jaxDevice()}
                </div>
            </Show>

            {/* Parameters */}
            <ConfigSlider
                label="N (Max Cluster Size)"
                value={store.n()}
                min={3}
                max={60}
                step={1}
                onChange={(value) => store.setN(Math.round(value))}
                disabled={store.isStreaming() || store.loadStatus() === 'loading'}
                minLabel="3"
                maxLabel="60"
                valueClass="label-text-alt text-primary font-mono"
                rangeClass="range range-primary range-sm"
            />

            <ConfigSlider
                label="K (Neighbors for Mode)"
                value={store.k()}
                min={1}
                max={Math.max(1, store.n() - 1)}
                step={1}
                onChange={(value) => store.setK(Math.round(value))}
                disabled={store.isStreaming() || store.loadStatus() === 'loading'}
                minLabel="1"
                maxLabel={String(Math.max(1, store.n() - 1))}
                helperText="K must be less than N. Also used for even-ness."
                valueClass="label-text-alt text-secondary font-mono"
                rangeClass="range range-secondary range-sm"
            />

            <ConfigSlider
                label="Image Count"
                value={store.subsetSize()}
                min={100}
                max={5000}
                step={100}
                onChange={(value) => store.setSubsetSize(Math.round(value))}
                disabled={store.isStreaming() || store.loadStatus() === 'loading'}
                minLabel="100"
                maxLabel="5000"
                helperText="Number of MNIST test images to cluster"
                valueClass="label-text-alt text-info font-mono"
                rangeClass="range range-info range-sm"
            />

            <ConfigSlider
                label="Stream Speed"
                value={store.streamSpeed()}
                min={50}
                max={1000}
                step={50}
                onChange={(value) => store.setStreamSpeed(Math.round(value))}
                minLabel="Fast (50ms)"
                maxLabel="Slow (1s)"
                valueFormatter={(v) => `${v}ms`}
                valueClass="label-text-alt text-accent font-mono"
                rangeClass="range range-accent range-sm"
            />

            {/* Top-half mechanism toggle */}
            <div class="form-control">
                <label class="label cursor-pointer justify-start gap-3">
                    <input
                        type="checkbox"
                        class="toggle toggle-warning"
                        checked={store.top1Enabled()}
                        onChange={(e) => store.setTop1Enabled(e.currentTarget.checked)}
                        disabled={store.isStreaming() || store.loadStatus() === 'loading'}
                    />
                    <span class="label-text">Top-Half Growth</span>
                </label>
                <span class="text-xs text-base-content/50 pl-1">
                    Top half by even-ness (≥ 2 clusters) grow permanently instead of evicting
                </span>
            </div>

            {/* Last-1 shrink toggle */}
            <div class="form-control">
                <label class="label cursor-pointer justify-start gap-3">
                    <input
                        type="checkbox"
                        class="toggle toggle-error"
                        checked={store.last1ShrinkEnabled()}
                        onChange={(e) => store.setLast1ShrinkEnabled(e.currentTarget.checked)}
                        disabled={store.isStreaming() || store.loadStatus() === 'loading'}
                    />
                    <span class="label-text">Last-1 Shrink</span>
                </label>
                <span class="text-xs text-base-content/50 pl-1">
                    When Top-1 grows: shrink the least-even cluster by 1 (skips clusters already at K+1)
                </span>
            </div>

            {/* Action buttons */}
            <div class="flex flex-col gap-3 pt-1">
                <button
                    class="btn btn-primary shadow-lg shadow-primary/20"
                    onClick={() => store.loadAndGenerate()}
                    disabled={store.isStreaming() || store.loadStatus() === 'loading'}
                >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <Show when={store.loadStatus() === 'idle'} fallback="Reload MNIST">
                        Load MNIST
                    </Show>
                </button>

                <div class="flex gap-2">
                    <Show
                        when={!store.isStreaming()}
                        fallback={
                            <button
                                class="btn btn-warning flex-1"
                                onClick={() => store.stopStreaming()}
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                Pause
                            </button>
                        }
                    >
                        <button
                            class="btn btn-secondary flex-1"
                            onClick={() => store.startStreaming()}
                            disabled={store.loadStatus() !== 'ready'}
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Stream
                        </button>
                    </Show>

                    <button
                        class="btn btn-ghost"
                        onClick={() => { void store.stepOnce() }}
                        disabled={store.isStreaming() || store.loadStatus() !== 'ready'}
                        title="Step one image"
                    >
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                        </svg>
                    </button>
                </div>

                <button
                    class="btn btn-ghost btn-sm"
                    onClick={() => store.reset()}
                    disabled={store.isStreaming()}
                >
                    Reset Clusters
                </button>
            </div>
        </div>
    )

    const StagePanel = () => (
        <div class="h-full min-h-0 overflow-y-auto pr-4">
            <Show
                when={store.clusters().length > 0}
                fallback={
                    <div
                        class="h-full flex flex-col items-center justify-center gap-4 opacity-40"
                        style={{ color: '#2d2d2d', 'font-family': "'Patrick Hand', cursive" }}
                    >
                        <svg class="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 6h16M4 12h16M4 18h7" />
                        </svg>
                        <Show
                            when={store.loadStatus() === 'ready'}
                            fallback={<p class="text-lg">Load MNIST to begin</p>}
                        >
                            <p class="text-lg">Press Stream or Step to cluster images</p>
                        </Show>
                    </div>
                }
            >
                {/* Progress bar + sort toggle */}
                <div class="mb-4 flex flex-col gap-2">
                    <div class="flex items-center justify-between">
                        <Show when={store.processedCount() < store.totalCount()}>
                            <div class="flex items-center gap-2 text-xs opacity-50" style={{ color: '#2d2d2d' }}>
                                <span>{store.processedCount()} / {store.totalCount()} images</span>
                                <span>·</span>
                                <span>{store.clusters().length} clusters</span>
                            </div>
                        </Show>
                        <Show when={store.processedCount() >= store.totalCount()}>
                            <div class="text-xs opacity-50" style={{ color: '#2d2d2d' }}>
                                {store.clusters().length} clusters
                            </div>
                        </Show>
                        <label class="flex items-center gap-1.5 cursor-pointer select-none text-xs opacity-60 hover:opacity-90" style={{ color: '#2d2d2d' }}>
                            <input
                                type="checkbox"
                                class="toggle toggle-xs"
                                checked={sortByEvenness()}
                                onChange={(e) => setSortByEvenness(e.currentTarget.checked)}
                            />
                            <span style={{ 'font-family': "'Patrick Hand', cursive" }}>sort by even-ness</span>
                        </label>
                    </div>
                    <Show when={store.processedCount() < store.totalCount()}>
                        <progress
                            class="progress progress-primary w-full h-1.5"
                            value={store.processedCount()}
                            max={store.totalCount()}
                        />
                    </Show>
                </div>

                {/* Cluster cards */}
                <div class="columns-[280px] gap-4 space-y-0">
                    <For each={displayedClusters()}>
                        {(cluster) => (
                            <div class="break-inside-avoid mb-4">
                                <ClusterCard
                                    cluster={cluster}
                                    getPixels={store.getImagePixels}
                                    imageSize={imageSize}
                                />
                            </div>
                        )}
                    </For>
                </div>
            </Show>
        </div>
    )

    return (
        <SingleStageFullLayout
            title="MNIST NK Clustering"
            subtitle="Online NK clustering on handwritten digits"
            backHref="/"
            backLabel="Home"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
