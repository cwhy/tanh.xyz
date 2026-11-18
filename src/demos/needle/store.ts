import { createSignal, createStore } from 'solid-js'
import {
    DEFAULT_NEEDLE_MANIFEST_URL,
    NEEDLE_26M_CONFIG,
    disposeNeedleParams,
    estimateParameterBytes,
    initJax,
    initNeedleParams,
    loadNeedleTokenizer,
    loadNeedleParamsFromManifest,
    makeNeedleSourceTokens,
    makeNeedleAllowedTokenIds,
    makeNeedleTargetTokens,
    needleParamCount,
    runNeedleProbe,
    type NeedleConfig,
    type NeedleParams,
    type NeedleProbeResult,
    type NeedleTokenizer,
} from './models'

export type NeedleWeightSource = 'deterministic' | 'manifest'

export interface NeedleDemoState {
    isInitializing: boolean
    isLoadingWeights: boolean
    isRunning: boolean
    device: string
    error: string | null
    weightStatus: string
    activeSource: NeedleWeightSource | null
    parameterCount: number
    estimatedBf16Mb: number
    tokenizerStatus: string
    lastProbe: NeedleProbeResult | null
}

export const EXAMPLE_QUERY = "What's the weather in San Francisco?"
export const EXAMPLE_TOOLS_JSON = JSON.stringify([
    {
        name: 'get_weather',
        parameters: {
            location: { type: 'string', description: 'City or place, for example San Francisco' },
        },
    },
    {
        name: 'calculator',
        parameters: {
            input: { type: 'string', description: 'Math expression, for example 1 + 1' },
        },
    },
    {
        name: 'pro_talker',
        parameters: {
            prompt: { type: 'string', description: 'Plain-language response prompt' },
        },
    },
])
export const EXAMPLE_DECODER_PREFIX = '[{"name":"get_weather","arguments":{"location":"'

const DEFAULT_QUERY = EXAMPLE_QUERY
const DEFAULT_TOOLS = EXAMPLE_TOOLS_JSON
const DEFAULT_PREFIX = ''

export function createNeedleDemoStore() {
    const config: NeedleConfig = NEEDLE_26M_CONFIG
    const [state, setState] = createStore<NeedleDemoState>({
        isInitializing: true,
        isLoadingWeights: false,
        isRunning: false,
        device: '',
        error: null,
        weightStatus: 'No weights loaded',
        activeSource: null,
        parameterCount: needleParamCount(config),
        estimatedBf16Mb: estimateParameterBytes(config, 2) / 1_000_000,
        tokenizerStatus: 'Loading vocabulary...',
        lastProbe: null,
    })

    const [seed, setSeed] = createSignal(17)
    const [weightSource, setWeightSource] = createSignal<NeedleWeightSource>(
        DEFAULT_NEEDLE_MANIFEST_URL ? 'manifest' : 'deterministic'
    )
    const [manifestUrl, setManifestUrl] = createSignal(DEFAULT_NEEDLE_MANIFEST_URL)
    const [query, setQuery] = createSignal(DEFAULT_QUERY)
    const [toolsJson, setToolsJson] = createSignal(DEFAULT_TOOLS)
    const [decoderPrefix, setDecoderPrefix] = createSignal(DEFAULT_PREFIX)
    const [generatedTokens, setGeneratedTokens] = createSignal<number[]>([])
    const [vocabulary, setVocabulary] = createSignal<string[] | null>(null)
    const [tokenizer, setTokenizer] = createSignal<NeedleTokenizer | null>(null)

    let params: NeedleParams | null = null
    let activeConfig = config

    async function initialize() {
        try {
            setState(s => {
                s.isInitializing = true
                s.error = null
            })
            const device = await initJax()
            setState(s => {
                s.isInitializing = false
                s.device = device
            })

            try {
                const loadedTokenizer = await loadNeedleTokenizer()
                setTokenizer(loadedTokenizer)
                setVocabulary(loadedTokenizer.vocabulary)
                setState(s => {
                    s.tokenizerStatus = `Loaded real encoder (${loadedTokenizer.vocabulary.length.toLocaleString('en-US')} pieces)`
                })
            } catch (vocabError) {
                const message = vocabError instanceof Error ? vocabError.message : 'Failed to load Needle tokenizer'
                setState(s => {
                    s.tokenizerStatus = message
                })
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to initialize jax-js'
            setState(s => {
                s.isInitializing = false
                s.error = message
            })
        }
    }

    async function loadWeights() {
        if (state.isInitializing || state.isLoadingWeights) return
        const source = weightSource()
        const nextManifestUrl = manifestUrl().trim()
        if (source === 'manifest' && !nextManifestUrl) {
            setState(s => {
                s.error = 'Set a manifest URL for R2 or another public origin first.'
            })
            return
        }

        setState(s => {
            s.isLoadingWeights = true
            s.error = null
            s.lastProbe = null
            s.weightStatus = source === 'manifest' ? 'Fetching manifest and tensors...' : 'Allocating deterministic 26M-scale weights...'
        })
        setGeneratedTokens([])

        disposeNeedleParams(params)
        params = null

        try {
            if (source === 'manifest') {
                const loaded = await loadNeedleParamsFromManifest(nextManifestUrl, config)
                params = loaded.params
                activeConfig = loaded.config
                setState(s => {
                    s.activeSource = 'manifest'
                    s.parameterCount = needleParamCount(activeConfig)
                    s.estimatedBf16Mb = estimateParameterBytes(activeConfig, 2) / 1_000_000
                    s.weightStatus = 'Loaded converted Needle checkpoint tensors'
                })
            } else {
                activeConfig = config
                params = initNeedleParams(seed(), activeConfig)
                setState(s => {
                    s.activeSource = 'deterministic'
                    s.parameterCount = needleParamCount(activeConfig)
                    s.estimatedBf16Mb = estimateParameterBytes(activeConfig, 2) / 1_000_000
                    s.weightStatus = 'Loaded deterministic same-shape SAN weights'
                })
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to load Needle weights'
            setState(s => {
                s.error = message
                s.weightStatus = 'Weight load failed'
                s.activeSource = null
            })
        } finally {
            setState(s => {
                s.isLoadingWeights = false
            })
        }
    }

    async function runProbe() {
        if (state.isInitializing || state.isLoadingWeights || state.isRunning) return
        if (generatedTokens().at(-1) === activeConfig.eosTokenId) return
        if (!params) {
            await loadWeights()
            if (!params) return
        }

        try {
            setState(s => {
                s.isRunning = true
                s.error = null
            })
            const srcTokens = makeNeedleSourceTokens(query(), toolsJson(), activeConfig, tokenizer())
            const tgtTokens = makeNeedleTargetTokens(decoderPrefix(), activeConfig, generatedTokens(), tokenizer())
            const allowedTokenIds = makeNeedleAllowedTokenIds(
                toolsJson(),
                decoderPrefix(),
                generatedTokens(),
                activeConfig,
                vocabulary()
            )
            const result = await runNeedleProbe(params, srcTokens, tgtTokens, activeConfig, { allowedTokenIds })
            setState(s => {
                s.lastProbe = result
            })
            setGeneratedTokens((tokens) => [...tokens, result.nextToken])
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Needle forward pass failed'
            setState(s => {
                s.error = message
            })
        } finally {
            setState(s => {
                s.isRunning = false
            })
        }
    }

    function resetGeneratedTokens() {
        setGeneratedTokens([])
        setState(s => {
            s.lastProbe = null
        })
    }

    function setQueryInput(value: string) {
        setQuery(value)
        resetGeneratedTokens()
    }

    function setToolsJsonInput(value: string) {
        setToolsJson(value)
        resetGeneratedTokens()
    }

    function setDecoderPrefixInput(value: string) {
        setDecoderPrefix(value)
        resetGeneratedTokens()
    }

    function dispose() {
        disposeNeedleParams(params)
        params = null
    }

    return {
        state,
        config,
        seed,
        setSeed,
        weightSource,
        setWeightSource,
        manifestUrl,
        setManifestUrl,
        query,
        setQuery: setQueryInput,
        toolsJson,
        setToolsJson: setToolsJsonInput,
        decoderPrefix,
        setDecoderPrefix: setDecoderPrefixInput,
        generatedTokens,
        vocabulary,
        resetGeneratedTokens,
        initialize,
        loadWeights,
        runProbe,
        dispose,
    }
}

export type NeedleDemoStore = ReturnType<typeof createNeedleDemoStore>
