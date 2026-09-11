import { For, Show, createMemo, onCleanup, onMount } from 'solid-js'
import { SingleStageLayout } from '../../components/demo-layouts/SingleStage'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { CANONICAL_NEEDLE_MANIFEST_URL, decodeNeedleTokenPieces, tokenPiece } from './models'
import { EXAMPLE_DECODER_PREFIX, EXAMPLE_QUERY, EXAMPLE_TOOLS_JSON, createNeedleDemoStore, type NeedleWeightSource } from './store'

const weightSourceOptions: Array<{ label: string; value: NeedleWeightSource }> = [
    { label: 'Same Shape', value: 'deterministic' },
    { label: 'Manifest', value: 'manifest' },
]

function formatCount(value: number): string {
    return value.toLocaleString('en-US')
}

function formatMb(value: number): string {
    return `${value.toFixed(1)} MB`
}

export function NeedlePage() {
    return <NeedleDemo />
}

export function NeedleDemo() {
    const store = createNeedleDemoStore()
    const {
        state,
        config,
        seed,
        setSeed,
        weightSource,
        setWeightSource,
        manifestUrl,
        setManifestUrl,
        query,
        setQuery,
        toolsJson,
        setToolsJson,
        decoderPrefix,
        setDecoderPrefix,
        generatedTokens,
        vocabulary,
        resetGeneratedTokens,
        initialize,
        loadWeights,
        runProbe,
        dispose,
    } = store

    onMount(() => {
        void initialize()
    })

    onCleanup(() => {
        dispose()
    })

    const sourceLabel = createMemo(() => {
        if (state.activeSource === 'manifest') return 'Converted checkpoint'
        if (state.activeSource === 'deterministic') return 'Deterministic init'
        return 'None'
    })

    const generatedTokenText = createMemo(() => generatedTokens().join(' '))
    const generatedPieceText = createMemo(() => (
        generatedTokens()
            .map((token) => `${token}:${tokenPiece(token, vocabulary())}`)
            .join(' ')
    ))
    const generatedDecodedText = createMemo(() => decodeNeedleTokenPieces(generatedTokens(), vocabulary()))

    const ConfigPanel = () => (
        <div class="h-full overflow-y-auto px-2 py-1 space-y-5">
            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Parameters</div>
                    <div class="stat-value text-lg text-primary">{formatCount(state.parameterCount)}</div>
                </div>
                <div class="stat bg-base-300/50 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">BF16 Weights</div>
                    <div class="stat-value text-lg text-secondary">{formatMb(state.estimatedBf16Mb)}</div>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
                <div class="stat bg-base-300/40 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Device</div>
                    <div class="stat-value text-base text-blue-700">{state.device || '—'}</div>
                </div>
                <div class="stat bg-base-300/40 rounded-xl border border-base-content/5 p-3">
                    <div class="stat-title text-xs opacity-60">Weights</div>
                    <div class="stat-value text-base text-fuchsia-600">{sourceLabel()}</div>
                </div>
            </div>

            <ConfigRadioGroup
                label="Weight Source"
                name="needle-weight-source"
                value={weightSource()}
                options={weightSourceOptions}
                onChange={setWeightSource}
                controlsInline
                disabled={state.isInitializing || state.isLoadingWeights || state.isRunning}
            />

            <Show when={weightSource() === 'manifest'} fallback={
                <label class="form-control">
                    <div class="label pb-2 flex justify-between items-center">
                        <span class="label-text font-medium">Seed</span>
                        <span class="label-text-alt font-mono text-primary">{seed()}</span>
                    </div>
                    <input
                        class="input input-sm input-bordered"
                        type="number"
                        min={1}
                        max={2147483647}
                        step={1}
                        value={seed()}
                        disabled={state.isInitializing || state.isLoadingWeights || state.isRunning}
                        onInput={(event) => {
                            const value = Number.parseInt(event.currentTarget.value, 10)
                            if (Number.isFinite(value)) setSeed(Math.max(1, Math.min(2147483647, value)))
                        }}
                    />
                </label>
            }>
                <label class="form-control">
                    <div class="label pb-2">
                        <span class="label-text font-medium">Manifest URL</span>
                    </div>
                    <input
                        class="input input-sm input-bordered font-mono text-xs"
                        value={manifestUrl()}
                        placeholder={CANONICAL_NEEDLE_MANIFEST_URL}
                        disabled={state.isInitializing || state.isLoadingWeights || state.isRunning}
                        onInput={(event) => setManifestUrl(event.currentTarget.value)}
                    />
                </label>
            </Show>

            <label class="form-control">
                <div class="label pb-2">
                    <span class="label-text font-medium">Query</span>
                </div>
                <textarea
                    class="textarea textarea-bordered min-h-24 text-base leading-snug"
                    value={query()}
                    placeholder={EXAMPLE_QUERY}
                    disabled={state.isInitializing || state.isRunning}
                    onInput={(event) => setQuery(event.currentTarget.value)}
                />
            </label>

            <label class="form-control">
                <div class="label pb-2">
                    <span class="label-text font-medium">Tools JSON</span>
                </div>
                <textarea
                    class="textarea textarea-bordered min-h-28 font-mono text-xs leading-snug"
                    value={toolsJson()}
                    placeholder={EXAMPLE_TOOLS_JSON}
                    disabled={state.isInitializing || state.isRunning}
                    onInput={(event) => setToolsJson(event.currentTarget.value)}
                >
                    {toolsJson()}
                </textarea>
            </label>

            <label class="form-control">
                <div class="label pb-2">
                    <span class="label-text font-medium">Decoder Prefix</span>
                </div>
                <input
                    class="input input-sm input-bordered font-mono text-xs"
                    value={decoderPrefix()}
                    placeholder={EXAMPLE_DECODER_PREFIX}
                    disabled={state.isInitializing || state.isRunning}
                    onInput={(event) => setDecoderPrefix(event.currentTarget.value)}
                />
            </label>

            <div class="rounded-xl border border-base-content/10 bg-base-300/30 p-3 text-sm">
                <div class="flex justify-between gap-3">
                    <span class="text-base-content/60">Tokenizer</span>
                    <span class="font-mono text-right">{state.tokenizerStatus}</span>
                </div>
                <div class="flex justify-between gap-3">
                    <span class="text-base-content/60">Architecture</span>
                    <span class="font-mono text-right">{config.numEncoderLayers}e/{config.numDecoderLayers}d · d={config.dModel}</span>
                </div>
                <div class="flex justify-between gap-3">
                    <span class="text-base-content/60">Heads</span>
                    <span class="font-mono text-right">{config.numHeads}Q / {config.numKvHeads}KV</span>
                </div>
                <div class="flex justify-between gap-3">
                    <span class="text-base-content/60">Vocabulary</span>
                    <span class="font-mono text-right">{config.vocabSize}</span>
                </div>
            </div>

            <div class="flex gap-3 pt-1">
                <button
                    class="btn btn-secondary flex-1"
                    onClick={() => void loadWeights()}
                    disabled={state.isInitializing || state.isLoadingWeights || state.isRunning}
                >
                    <Show when={state.isLoadingWeights} fallback="Load Weights">
                        <span class="loading loading-spinner loading-sm" />
                        Loading
                    </Show>
                </button>
                <button
                    class="btn btn-primary flex-1 shadow-lg shadow-primary/20"
                    onClick={() => void runProbe()}
                    disabled={state.isInitializing || state.isLoadingWeights || state.isRunning}
                >
                    <Show when={state.isRunning} fallback="Run Forward">
                        <span class="loading loading-spinner loading-sm" />
                        Running
                    </Show>
                </button>
            </div>

            <Show when={state.error}>
                <div class="alert alert-error text-sm">{state.error}</div>
            </Show>
        </div>
    )

    const GeneratedTokensPanel = () => (
        <div class="min-h-0 flex-1 rounded-xl border border-base-content/10 bg-base-200 p-4">
            <div class="mb-3 flex items-center justify-between gap-3">
                <h2 class="font-heading text-2xl">Generated Tokens</h2>
                <button
                    class="btn btn-ghost btn-xs"
                    onClick={resetGeneratedTokens}
                    disabled={state.isInitializing || state.isRunning || generatedTokens().length === 0}
                >
                    Clear
                </button>
            </div>
            <div class="h-[calc(100%-3rem)] min-h-24 overflow-y-auto rounded-lg border border-base-content/10 bg-base-100 px-3 py-2 font-mono text-xs leading-relaxed text-base-content/80">
                <Show when={generatedTokens().length > 0} fallback={<span class="text-base-content/40">No generated tokens yet</span>}>
                    <div>{generatedTokenText()}</div>
                    <div class="mt-1 break-words text-primary">{generatedPieceText()}</div>
                    <div class="mt-1 break-words text-base-content/70">{generatedDecodedText() || 'No printable text yet'}</div>
                </Show>
            </div>
        </div>
    )

    const StagePanel = () => (
        <div class="h-full min-h-0 rounded-[2rem] border-[3px] border-base-content bg-base-100 p-5 shadow-[6px_6px_0px_0px_#2d2d2d] overflow-hidden">
            <div class="flex h-full min-h-0 flex-col gap-4">
                <div class="grid grid-cols-3 gap-3">
                    <div class="rounded-xl border border-base-content/10 bg-base-200 p-4">
                        <div class="text-xs uppercase tracking-wide text-base-content/50">Status</div>
                        <div class="mt-1 text-xl font-heading text-primary">{state.weightStatus}</div>
                    </div>
                    <div class="rounded-xl border border-base-content/10 bg-base-200 p-4">
                        <div class="text-xs uppercase tracking-wide text-base-content/50">Source Tokens</div>
                        <div class="mt-1 text-2xl font-mono text-secondary">{state.lastProbe?.srcLength ?? '—'}</div>
                    </div>
                    <div class="rounded-xl border border-base-content/10 bg-base-200 p-4">
                        <div class="text-xs uppercase tracking-wide text-base-content/50">Forward Time</div>
                        <div class="mt-1 text-2xl font-mono text-accent">
                            {state.lastProbe ? `${state.lastProbe.elapsedMs.toFixed(0)} ms` : '—'}
                        </div>
                    </div>
                </div>

                <div class="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_320px] gap-4">
                    <div class="min-h-0 rounded-xl border border-base-content/10 bg-base-200 p-4">
                        <div class="mb-3 flex items-center justify-between">
                            <h2 class="font-heading text-2xl">Tied Output Head</h2>
                            <span class="rounded bg-base-100 px-2 py-1 font-mono text-xs text-base-content/60">
                                next {state.lastProbe?.nextToken ?? '—'}
                            </span>
                        </div>
                        <Show when={state.lastProbe}>
                            <div class="mb-3 rounded-lg bg-base-100 px-3 py-2 text-sm">
                                <span class="text-base-content/60">Appended token </span>
                                <span class="font-mono text-primary">{state.lastProbe?.nextToken}</span>
                                <span class="text-base-content/60"> to decoder input</span>
                            </div>
                        </Show>
                        <Show
                            when={state.lastProbe}
                            fallback={
                                <div class="flex h-[calc(100%-3rem)] items-center justify-center rounded-lg border border-dashed border-base-content/20 text-base-content/50">
                                    No forward pass yet
                                </div>
                            }
                        >
                            {(probe) => (
                                <div class="space-y-3">
                                    <For each={probe().topTokens}>
                                        {(token, index) => {
                                            const width = () => {
                                                const top = probe().topTokens[0]?.logit ?? 1
                                                const min = probe().topTokens[probe().topTokens.length - 1]?.logit ?? 0
                                                const denom = Math.max(1e-6, top - min)
                                                return 18 + ((token.logit - min) / denom) * 82
                                            }
                                            return (
                                                <div>
                                                    <div class="mb-1 flex items-center justify-between gap-3 text-sm">
                                                        <span class="font-mono">#{index() + 1} token {token.token} {tokenPiece(token.token, vocabulary())}</span>
                                                        <span class="font-mono text-base-content/60">{token.logit.toFixed(4)}</span>
                                                    </div>
                                                    <div class="h-5 overflow-hidden rounded bg-base-300">
                                                        <div
                                                            class="h-full bg-primary"
                                                            style={{ width: `${width()}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            )
                                        }}
                                    </For>
                                </div>
                            )}
                        </Show>
                    </div>

                    <div class="min-h-0 flex flex-col gap-4">
                        <GeneratedTokensPanel />

                        <div class="shrink-0 rounded-xl border border-base-content/10 bg-base-200 p-4">
                            <h2 class="font-heading text-2xl">Run Shape</h2>
                            <div class="mt-4 space-y-3 text-sm">
                                <div class="flex justify-between gap-3">
                                    <span class="text-base-content/60">Encoder blocks</span>
                                    <span class="font-mono">{config.numEncoderLayers}</span>
                                </div>
                                <div class="flex justify-between gap-3">
                                    <span class="text-base-content/60">Decoder blocks</span>
                                    <span class="font-mono">{config.numDecoderLayers}</span>
                                </div>
                                <div class="flex justify-between gap-3">
                                    <span class="text-base-content/60">Decoder tokens</span>
                                    <span class="font-mono">{state.lastProbe?.tgtLength ?? '—'}</span>
                                </div>
                                <div class="flex justify-between gap-3">
                                    <span class="text-base-content/60">Output logits</span>
                                    <span class="font-mono">{config.vocabSize}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )

    return (
        <SingleStageLayout
            title="Needle SAN"
            subtitle="26M-scale encoder-decoder attention in jax-js"
            backHref="/demo"
            config={ConfigPanel}
            stage={StagePanel}
        />
    )
}
