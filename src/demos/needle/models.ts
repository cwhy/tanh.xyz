import { nn, numpy as np, tree } from '@jax-js/jax'
import { initJax } from '../../lib/deeplearning/runtime'

export { initJax }

export interface NeedleConfig {
    vocabSize: number
    dModel: number
    numHeads: number
    numKvHeads: number
    numEncoderLayers: number
    numDecoderLayers: number
    maxSeqLen: number
    padTokenId: number
    eosTokenId: number
    toolCallTokenId: number
    toolsTokenId: number
    ropeTheta: number
}

export interface NeedleAttentionParams {
    qProj: np.Array
    kProj: np.Array
    vProj: np.Array
    outProj: np.Array
    qNormScale: np.Array
    kNormScale: np.Array
}

export interface NeedleEncoderLayerParams {
    attnNormScale: np.Array
    attnGate: np.Array
    selfAttn: NeedleAttentionParams
}

export interface NeedleDecoderLayerParams {
    selfNormScale: np.Array
    selfAttnGate: np.Array
    selfAttn: NeedleAttentionParams
    crossNormScale: np.Array
    crossAttnGate: np.Array
    crossAttn: NeedleAttentionParams
}

export interface NeedleParams {
    embedding: np.Array
    encoderLayers: NeedleEncoderLayerParams[]
    encoderFinalNormScale: np.Array
    decoderLayers: NeedleDecoderLayerParams[]
    decoderFinalNormScale: np.Array
    [key: string]: np.Array | NeedleEncoderLayerParams[] | NeedleDecoderLayerParams[]
}

export interface NeedleForwardResult {
    logits: np.Array
    encoderOut: np.Array
}

export interface NeedleProbeResult {
    nextToken: number
    topTokens: Array<{ token: number; logit: number }>
    elapsedMs: number
    srcLength: number
    tgtLength: number
}

export interface NeedleProbeOptions {
    allowedTokenIds?: ReadonlySet<number> | null
}

export interface NeedleTensorManifest {
    config?: Partial<NeedleConfig>
    baseUrl?: string
    tensors: Record<string, {
        shape: number[]
        dtype?: 'float32' | 'float16' | 'bfloat16'
        file?: string
    }>
}

export interface NeedleTokenizer {
    vocabulary: string[]
    scores: number[]
    encode: (text: string, maxTokens?: number) => number[]
}

export const NEEDLE_26M_CONFIG: NeedleConfig = {
    vocabSize: 8192,
    dModel: 512,
    numHeads: 8,
    numKvHeads: 4,
    numEncoderLayers: 12,
    numDecoderLayers: 8,
    maxSeqLen: 1024,
    padTokenId: 0,
    eosTokenId: 1,
    toolCallTokenId: 4,
    toolsTokenId: 5,
    ropeTheta: 10_000,
}

export const NEEDLE_R2_PARAMS_PREFIX = 'data/demos/needle/needle/params/v1/'
export const CANONICAL_NEEDLE_MANIFEST_URL = `https://media.tanh.xyz/${NEEDLE_R2_PARAMS_PREFIX}manifest.json`
export const CANONICAL_NEEDLE_VOCAB_URL = `https://media.tanh.xyz/${NEEDLE_R2_PARAMS_PREFIX}tokenizer/needle.vocab`
export const CANONICAL_NEEDLE_TOKENIZER_MODEL_URL = `https://media.tanh.xyz/${NEEDLE_R2_PARAMS_PREFIX}tokenizer/needle.model`
export const DEFAULT_NEEDLE_MANIFEST_URL = import.meta.env.VITE_NEEDLE_MODEL_MANIFEST_URL ?? CANONICAL_NEEDLE_MANIFEST_URL

const PARAM_INIT_STD = 0.02

interface SeededRng {
    next: () => number
    nextNormal: () => number
}

export function needleParamCount(config: NeedleConfig = NEEDLE_26M_CONFIG): number {
    const headDim = config.dModel / config.numHeads
    const kvDim = config.numKvHeads * headDim
    const attnParams = (config.dModel * config.dModel)
        + (config.dModel * kvDim)
        + (config.dModel * kvDim)
        + (config.dModel * config.dModel)
        + headDim
        + headDim
    const encoderLayerParams = config.dModel + 1 + attnParams
    const decoderLayerParams = (config.dModel + 1 + attnParams) * 2

    return (config.vocabSize * config.dModel)
        + (config.numEncoderLayers * encoderLayerParams)
        + config.dModel
        + (config.numDecoderLayers * decoderLayerParams)
        + config.dModel
}

export function estimateParameterBytes(config: NeedleConfig = NEEDLE_26M_CONFIG, bytesPerParam: number = 2): number {
    return needleParamCount(config) * bytesPerParam
}

export function makeNeedleSourceTokens(
    query: string,
    toolsJson: string,
    config: NeedleConfig = NEEDLE_26M_CONFIG,
    tokenizer: NeedleTokenizer | null = null
): Int32Array {
    const queryTokens = tokenizeText(query, config.vocabSize, 120, tokenizer)
    const toolsTokens = tokenizeText(normalizeNeedleToolsJson(toolsJson), config.vocabSize, 360, tokenizer)
    const merged = [...queryTokens, config.toolsTokenId, ...toolsTokens].slice(0, 512)
    return new Int32Array(merged.length > 0 ? merged : [config.eosTokenId])
}

export function makeNeedleTargetTokens(
    prefix: string,
    config: NeedleConfig = NEEDLE_26M_CONFIG,
    generatedTokens: readonly number[] = [],
    tokenizer: NeedleTokenizer | null = null
): Int32Array {
    const prefixTokens = tokenizeText(prefix, config.vocabSize, 96, tokenizer)
    const generated = generatedTokens
        .map((token) => Math.trunc(token))
        .filter((token) => token >= 0 && token < config.vocabSize)
    const baseTokens = prefixTokens.length > 0
        ? [config.eosTokenId, config.toolCallTokenId, ...prefixTokens]
        : [config.eosTokenId]
    const availableGeneratedSlots = Math.max(0, 128 - baseTokens.length)
    return new Int32Array([...baseTokens, ...generated.slice(-availableGeneratedSlots)].slice(0, 128))
}

export async function loadNeedleTokenizer(vocabUrl: string = CANONICAL_NEEDLE_VOCAB_URL): Promise<NeedleTokenizer> {
    const response = await fetch(vocabUrl)
    if (!response.ok) throw new Error(`Failed to fetch Needle tokenizer vocabulary: ${response.status}`)
    return createNeedleTokenizer(await response.text())
}

export async function loadNeedleVocabulary(vocabUrl: string = CANONICAL_NEEDLE_VOCAB_URL): Promise<string[]> {
    const response = await fetch(vocabUrl)
    if (!response.ok) throw new Error(`Failed to fetch Needle vocabulary: ${response.status}`)
    const text = await response.text()
    return text
        .split(/\r?\n/)
        .filter((line) => line.length > 0)
        .map((line) => line.split('\t')[0] ?? '')
}

export function createNeedleTokenizer(vocabText: string): NeedleTokenizer {
    const entries = vocabText
        .split(/\r?\n/)
        .filter((line) => line.length > 0)
        .map((line) => {
            const [piece = '', score = '0'] = line.split('\t')
            return { piece, score: Number.parseFloat(score) || 0 }
        })
    const vocabulary = entries.map((entry) => entry.piece)
    const scores = entries.map((entry) => entry.score)
    const pieceToId = new Map(vocabulary.map((piece, id) => [piece, id]))
    const byteTokenIds = new Map<number, number>()
    let maxPieceLength = 0

    for (const [id, piece] of vocabulary.entries()) {
        const byte = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece)
        if (byte) {
            byteTokenIds.set(Number.parseInt(byte[1], 16), id)
            continue
        }
        if (!isSpecialPiece(piece)) maxPieceLength = Math.max(maxPieceLength, piece.length)
    }

    const encode = (text: string, maxTokens = Number.POSITIVE_INFINITY): number[] => (
        encodeWithVocabulary(text, pieceToId, scores, byteTokenIds, maxPieceLength, maxTokens)
    )

    return { vocabulary, scores, encode }
}

export function tokenPiece(token: number, vocabulary: readonly string[] | null): string {
    const id = Math.trunc(token)
    return vocabulary?.[id] ?? `<token:${id}>`
}

export function decodeNeedleTokenPieces(tokens: readonly number[], vocabulary: readonly string[] | null): string {
    return tokens
        .map((token) => pieceToText(tokenPiece(token, vocabulary)))
        .join('')
        .trimStart()
}

export function makeNeedleAllowedTokenIds(
    toolsJson: string,
    prefix: string,
    generatedTokens: readonly number[],
    config: NeedleConfig = NEEDLE_26M_CONFIG,
    vocabulary: readonly string[] | null = null
): ReadonlySet<number> | null {
    if (!vocabulary) return null
    if (prefix.trim().length === 0 && generatedTokens.length === 0) {
        return new Set([config.toolCallTokenId])
    }

    const tools = parseNeedleTools(toolsJson)
    const visibleText = decodeNeedleConstraintText(prefix, generatedTokens, vocabulary)
    const constraint = inferNeedleTokenConstraint(visibleText, tools)
    if (constraint) return tokenIdsForConstraint(constraint, vocabulary, config)

    return defaultAllowedNeedleTokenIds(vocabulary, config, generatedTokens.length > 0 || prefix.trim().length > 0)
}

function pieceToText(piece: string): string {
    if (piece === '<pad>' || piece === '<s>' || piece === '</s>' || piece === '<unk>' || piece === '<tool_call>' || piece === '<tools>') {
        return ''
    }
    const byte = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece)
    if (byte) return String.fromCharCode(Number.parseInt(byte[1], 16))
    return piece.replace(/\u2581/g, ' ')
}

export function initNeedleParams(seed: number, config: NeedleConfig = NEEDLE_26M_CONFIG): NeedleParams {
    const rng = createSeededRng(seed)
    const headDim = config.dModel / config.numHeads
    const kvDim = config.numKvHeads * headDim
    const residualStd = PARAM_INIT_STD / Math.sqrt(2 * (config.numEncoderLayers + config.numDecoderLayers))

    const createAttention = (): NeedleAttentionParams => ({
        qProj: normalArray(rng, [config.dModel, config.dModel], PARAM_INIT_STD),
        kProj: normalArray(rng, [config.dModel, kvDim], PARAM_INIT_STD),
        vProj: normalArray(rng, [config.dModel, kvDim], PARAM_INIT_STD),
        outProj: normalArray(rng, [config.dModel, config.dModel], residualStd),
        qNormScale: np.zeros([headDim]),
        kNormScale: np.zeros([headDim]),
    })

    const encoderLayers: NeedleEncoderLayerParams[] = []
    for (let i = 0; i < config.numEncoderLayers; i++) {
        encoderLayers.push({
            attnNormScale: np.zeros([config.dModel]),
            attnGate: np.zeros([]),
            selfAttn: createAttention(),
        })
    }

    const decoderLayers: NeedleDecoderLayerParams[] = []
    for (let i = 0; i < config.numDecoderLayers; i++) {
        decoderLayers.push({
            selfNormScale: np.zeros([config.dModel]),
            selfAttnGate: np.zeros([]),
            selfAttn: createAttention(),
            crossNormScale: np.zeros([config.dModel]),
            crossAttnGate: np.zeros([]),
            crossAttn: createAttention(),
        })
    }

    return {
        embedding: normalArray(rng, [config.vocabSize, config.dModel], PARAM_INIT_STD),
        encoderLayers,
        encoderFinalNormScale: np.zeros([config.dModel]),
        decoderLayers,
        decoderFinalNormScale: np.zeros([config.dModel]),
    }
}

export async function loadNeedleParamsFromManifest(
    manifestUrl: string,
    fallbackConfig: NeedleConfig = NEEDLE_26M_CONFIG
): Promise<{ params: NeedleParams; config: NeedleConfig }> {
    const manifestResponse = await fetch(manifestUrl)
    if (!manifestResponse.ok) {
        throw new Error(`Failed to fetch Needle manifest: ${manifestResponse.status}`)
    }
    const manifest = await manifestResponse.json() as NeedleTensorManifest
    const config = { ...fallbackConfig, ...manifest.config }
    const manifestBase = new URL(manifestUrl, globalThis.location?.href)
    const baseUrl = manifest.baseUrl ? new URL(manifest.baseUrl, manifestBase).href : manifestBase.href.replace(/[^/]*$/, '')

    const load = async (name: string): Promise<np.Array> => {
        const entry = manifest.tensors[name]
        if (!entry) throw new Error(`Missing tensor "${name}" in Needle manifest`)
        const tensorUrl = new URL(entry.file ?? `${name}.f32.bin`, baseUrl).href
        const response = await fetch(tensorUrl)
        if (!response.ok) throw new Error(`Failed to fetch Needle tensor "${name}": ${response.status}`)
        const buffer = await response.arrayBuffer()
        const values = decodeTensorValues(buffer, entry.dtype)
        return np.array(values).reshape(entry.shape)
    }

    const loadAttention = async (prefix: string): Promise<NeedleAttentionParams> => ({
        qProj: await load(`${prefix}.qProj`),
        kProj: await load(`${prefix}.kProj`),
        vProj: await load(`${prefix}.vProj`),
        outProj: await load(`${prefix}.outProj`),
        qNormScale: await load(`${prefix}.qNormScale`),
        kNormScale: await load(`${prefix}.kNormScale`),
    })

    const encoderLayers: NeedleEncoderLayerParams[] = []
    for (let i = 0; i < config.numEncoderLayers; i++) {
        const prefix = `encoderLayers.${i}`
        encoderLayers.push({
            attnNormScale: await load(`${prefix}.attnNormScale`),
            attnGate: await load(`${prefix}.attnGate`),
            selfAttn: await loadAttention(`${prefix}.selfAttn`),
        })
    }

    const decoderLayers: NeedleDecoderLayerParams[] = []
    for (let i = 0; i < config.numDecoderLayers; i++) {
        const prefix = `decoderLayers.${i}`
        decoderLayers.push({
            selfNormScale: await load(`${prefix}.selfNormScale`),
            selfAttnGate: await load(`${prefix}.selfAttnGate`),
            selfAttn: await loadAttention(`${prefix}.selfAttn`),
            crossNormScale: await load(`${prefix}.crossNormScale`),
            crossAttnGate: await load(`${prefix}.crossAttnGate`),
            crossAttn: await loadAttention(`${prefix}.crossAttn`),
        })
    }

    return {
        config,
        params: {
            embedding: await load('embedding'),
            encoderLayers,
            encoderFinalNormScale: await load('encoderFinalNormScale'),
            decoderLayers,
            decoderFinalNormScale: await load('decoderFinalNormScale'),
        },
    }
}

export function disposeNeedleParams(params: NeedleParams | null) {
    if (!params) return
    try {
        tree.dispose(params)
    } catch {
        // jax-js owns arrays linearly; already-consumed params can be ignored here.
    }
}

export async function runNeedleProbe(
    params: NeedleParams,
    srcTokens: Int32Array,
    tgtTokens: Int32Array,
    config: NeedleConfig = NEEDLE_26M_CONFIG,
    options: NeedleProbeOptions = {}
): Promise<NeedleProbeResult> {
    const start = performance.now()
    const src = np.array(srcTokens, { dtype: np.int32 }).reshape([1, srcTokens.length])
    const tgt = np.array(tgtTokens, { dtype: np.int32 }).reshape([1, tgtTokens.length])
    const output = needleForward(tree.ref(params) as NeedleParams, src, tgt, config)
    const lastIndex = Math.max(0, tgtTokens.length - 1)
    const lastLogits = output.logits.ref.slice(0, lastIndex, [])
    const values = await lastLogits.jsAsync() as number[]
    output.encoderOut.dispose()
    output.logits.dispose()

    const filteredValues = options.allowedTokenIds?.size
        ? values.map((value, index) => options.allowedTokenIds?.has(index) ? value : Number.NEGATIVE_INFINITY)
        : values
    const topTokens = topK(filteredValues, 8).map(({ index, value }) => ({ token: index, logit: value }))
    return {
        nextToken: topTokens[0]?.token ?? 0,
        topTokens,
        elapsedMs: performance.now() - start,
        srcLength: srcTokens.length,
        tgtLength: tgtTokens.length,
    }
}

export function needleForward(
    params: NeedleParams,
    src: np.Array,
    tgt: np.Array,
    config: NeedleConfig = NEEDLE_26M_CONFIG
): NeedleForwardResult {
    const encoderOut = encodeText(params, src, config)
    const logits = decodeText(params, tgt, encoderOut.ref, src.shape[1], config)
    return { logits, encoderOut }
}

function encodeText(params: NeedleParams, src: np.Array, config: NeedleConfig): np.Array {
    const seqLen = src.shape[1]
    let x = embedTokens(src.ref, params.embedding.ref, config)
    const mask = makePaddingMask(src, config.padTokenId, seqLen)
    const rope = precomputeRope(config.dModel / config.numHeads, seqLen, config.ropeTheta)

    for (const layer of params.encoderLayers) {
        const residual = x
        const normed = zcRmsNorm(x.ref, layer.attnNormScale.ref)
        const attn = multiHeadAttention(normed, normed.ref, layer.selfAttn, mask.ref, rope, config)
        const gate = nn.sigmoid(layer.attnGate.ref)
        x = residual.add(attn.mul(gate))
    }

    mask.dispose()
    rope.cos.dispose()
    rope.sin.dispose()
    return zcRmsNorm(x, params.encoderFinalNormScale.ref)
}

function decodeText(
    params: NeedleParams,
    tgt: np.Array,
    encoderOut: np.Array,
    encoderLength: number,
    config: NeedleConfig
): np.Array {
    let x = embedTokens(tgt, params.embedding.ref, config)
    const selfMask = makeCausalMask(tgt.shape[1])
    const crossMask = np.ones([1, 1, tgt.shape[1], encoderLength], { dtype: np.bool })
    const rope = precomputeRope(config.dModel / config.numHeads, tgt.shape[1], config.ropeTheta)

    for (const layer of params.decoderLayers) {
        let residual = x
        let normed = zcRmsNorm(x.ref, layer.selfNormScale.ref)
        let attn = multiHeadAttention(normed, normed.ref, layer.selfAttn, selfMask.ref, rope, config)
        let gate = nn.sigmoid(layer.selfAttnGate.ref)
        x = residual.add(attn.mul(gate))

        residual = x
        normed = zcRmsNorm(x.ref, layer.crossNormScale.ref)
        attn = multiHeadAttention(normed, encoderOut.ref, layer.crossAttn, crossMask.ref, null, config)
        gate = nn.sigmoid(layer.crossAttnGate.ref)
        x = residual.add(attn.mul(gate))
    }

    selfMask.dispose()
    crossMask.dispose()
    rope.cos.dispose()
    rope.sin.dispose()
    const finalNorm = zcRmsNorm(x, params.decoderFinalNormScale.ref)
    return np.matmul(finalNorm, params.embedding.transpose([1, 0]))
}

function multiHeadAttention(
    qInput: np.Array,
    kvInput: np.Array,
    params: NeedleAttentionParams,
    mask: np.Array | null,
    rope: { cos: np.Array; sin: np.Array } | null,
    config: NeedleConfig
): np.Array {
    const batchSize = qInput.shape[0]
    const qLen = qInput.shape[1]
    const kvLen = kvInput.shape[1]
    const headDim = config.dModel / config.numHeads

    let q = np.matmul(qInput, params.qProj.ref).reshape([batchSize, qLen, config.numHeads, headDim])
    let k = np.matmul(kvInput.ref, params.kProj.ref).reshape([batchSize, kvLen, config.numKvHeads, headDim])
    let v = np.matmul(kvInput, params.vProj.ref).reshape([batchSize, kvLen, config.numKvHeads, headDim])

    q = zcRmsNorm(q, params.qNormScale.ref)
    k = zcRmsNorm(k, params.kNormScale.ref)

    if (rope) {
        q = applyRope(q, rope.cos.ref, rope.sin.ref)
        k = applyRope(k, rope.cos.ref, rope.sin.ref)
    }

    const repeats = config.numHeads / config.numKvHeads
    if (repeats > 1) {
        // Needle repeats each KV head adjacently. jax-js dotProductAttention's
        // built-in GQA path tiles KV heads, which pairs heads differently.
        k = np.repeat(k, repeats, 2)
        v = np.repeat(v, repeats, 2)
    }

    const attended = nn.dotProductAttention(
        q,
        k.ref,
        v,
        mask ? { mask } : undefined
    )
    const merged = attended.reshape([batchSize, qLen, config.dModel])
    return np.matmul(merged, params.outProj)
}

function embedTokens(tokens: np.Array, embedding: np.Array, config: NeedleConfig): np.Array {
    const tokenOneHot = nn.oneHot(tokens.astype(np.int32), config.vocabSize)
    return np.matmul(tokenOneHot, embedding).mul(Math.sqrt(config.dModel))
}

function zcRmsNorm(x: np.Array, scale: np.Array, eps: number = 1e-6): np.Array {
    const rms = np.sqrt(np.mean(np.square(x.ref), -1, { keepdims: true }).add(eps))
    return x.div(rms).mul(scale.add(1))
}

function precomputeRope(headDim: number, seqLen: number, theta: number): { cos: np.Array; sin: np.Array } {
    const half = headDim / 2
    const values = new Float32Array(seqLen * half)
    for (let pos = 0; pos < seqLen; pos++) {
        for (let i = 0; i < half; i++) {
            values[pos * half + i] = pos / Math.pow(theta, (2 * i) / headDim)
        }
    }
    const angles = np.array(values).reshape([seqLen, half])
    return {
        cos: np.cos(angles.ref),
        sin: np.sin(angles),
    }
}

function applyRope(x: np.Array, cos: np.Array, sin: np.Array): np.Array {
    const half = x.shape[3] / 2
    const x1 = x.ref.slice([], [], [], [0, half])
    const x2 = x.slice([], [], [], [half])
    const cos4 = cos.reshape([1, cos.shape[0], 1, half])
    const sin4 = sin.reshape([1, sin.shape[0], 1, half])
    const left = x1.ref.mul(cos4.ref).sub(x2.ref.mul(sin4.ref))
    const right = x2.mul(cos4).add(x1.mul(sin4))
    return np.concatenate([left, right], -1)
}

function makePaddingMask(tokens: np.Array, padTokenId: number, seqLen: number): np.Array {
    return tokens.notEqual(padTokenId).reshape([tokens.shape[0], 1, 1, seqLen])
}

function makeCausalMask(seqLen: number): np.Array {
    return np.tril(np.ones([seqLen, seqLen], { dtype: np.bool })).reshape([1, 1, seqLen, seqLen])
}

function normalArray(rng: SeededRng, shape: number[], std: number): np.Array {
    const size = shape.reduce((acc, value) => acc * value, 1)
    const values = new Float32Array(size)
    for (let i = 0; i < size; i++) values[i] = rng.nextNormal() * std
    return np.array(values).reshape(shape)
}

function createSeededRng(seed: number): SeededRng {
    let state = (seed >>> 0) || 1
    const next = () => {
        state ^= state << 13
        state ^= state >>> 17
        state ^= state << 5
        return ((state >>> 0) + 1) / 4294967297
    }
    const nextNormal = () => {
        const u1 = Math.max(next(), 1e-12)
        const u2 = next()
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }
    return { next, nextNormal }
}

function hashTextToTokens(text: string, vocabSize: number, maxTokens: number): number[] {
    const clean = text.trim()
    if (!clean) return []
    const tokens: number[] = []
    let hash = 2166136261
    for (let i = 0; i < clean.length && tokens.length < maxTokens; i++) {
        hash ^= clean.charCodeAt(i)
        hash = Math.imul(hash, 16777619) >>> 0
        if (i % 3 === 2 || /[\s{}[\]":,]/.test(clean[i])) {
            tokens.push(6 + (hash % Math.max(1, vocabSize - 6)))
            hash = (hash ^ (tokens.length * 2654435761)) >>> 0
        }
    }
    if (tokens.length === 0) tokens.push(6 + (hash % Math.max(1, vocabSize - 6)))
    return tokens
}

function tokenizeText(
    text: string,
    vocabSize: number,
    maxTokens: number,
    tokenizer: NeedleTokenizer | null
): number[] {
    if (!tokenizer) return hashTextToTokens(text, vocabSize, maxTokens)
    return tokenizer.encode(text, maxTokens).filter((token) => token >= 0 && token < vocabSize)
}

type NeedleToolSpec = {
    name: string
    parameters: string[]
}

type NeedleTokenConstraint =
    | { kind: 'literal'; current: string; expected: string }
    | { kind: 'choice'; partial: string; choices: string[] }

export function normalizeNeedleToolsJson(toolsJson: string): string {
    const trimmed = toolsJson.trim()
    if (!trimmed) return ''
    try {
        const parsed = JSON.parse(trimmed)
        const normalized = Array.isArray(parsed)
            ? parsed.map((tool) => (
                tool && typeof tool === 'object' && !Array.isArray(tool)
                    ? { ...tool, name: toSnakeCase(String((tool as { name?: unknown }).name ?? '')) }
                    : tool
            ))
            : parsed
        return JSON.stringify(normalized)
    } catch {
        return trimmed
    }
}

function parseNeedleTools(toolsJson: string): NeedleToolSpec[] {
    try {
        const parsed = JSON.parse(normalizeNeedleToolsJson(toolsJson))
        if (!Array.isArray(parsed)) return []
        return parsed
            .map((tool): NeedleToolSpec | null => {
                if (!tool || typeof tool !== 'object' || Array.isArray(tool)) return null
                const name = toSnakeCase(String((tool as { name?: unknown }).name ?? '')).trim()
                if (!name) return null
                const rawParameters = (tool as { parameters?: unknown }).parameters
                const parameters = rawParameters && typeof rawParameters === 'object' && !Array.isArray(rawParameters)
                    ? Object.keys(rawParameters).map(toSnakeCase).filter(Boolean)
                    : []
                return { name, parameters }
            })
            .filter((tool): tool is NeedleToolSpec => tool !== null)
    } catch {
        return []
    }
}

function toSnakeCase(value: string): string {
    return value
        .trim()
        .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
        .replace(/[\s.-]+/g, '_')
        .replace(/[^A-Za-z0-9_]/g, '')
        .replace(/_+/g, '_')
        .replace(/^_+|_+$/g, '')
        .toLowerCase()
}

function decodeNeedleConstraintText(
    prefix: string,
    generatedTokens: readonly number[],
    vocabulary: readonly string[]
): string {
    return `${prefix}${generatedTokens
        .map((token) => pieceToText(tokenPiece(token, vocabulary)))
        .join('')}`.trimStart()
}

function inferNeedleTokenConstraint(text: string, tools: readonly NeedleToolSpec[]): NeedleTokenConstraint | null {
    const clean = text.trimStart()
    const startLiteral = '[{"name":"'
    if (clean.length < startLiteral.length && startLiteral.startsWith(clean)) {
        return { kind: 'literal', current: clean, expected: startLiteral }
    }

    const unclosedName = unclosedJsonStringAfter(clean, '"name":"')
    if (unclosedName !== null) {
        const names = tools.map((tool) => tool.name)
        return names.length > 0 ? { kind: 'choice', partial: unclosedName, choices: names } : null
    }

    const selectedTool = lastClosedToolName(clean, tools)
    if (!selectedTool) return null

    const nameEnd = clean.lastIndexOf(`"name":"${selectedTool.name}"`) + `"name":"${selectedTool.name}"`.length
    const afterName = clean.slice(nameEnd)
    const argumentsLiteral = ',"arguments":{'
    if (argumentsLiteral.startsWith(afterName)) {
        return { kind: 'literal', current: afterName, expected: argumentsLiteral }
    }

    const argumentsStart = afterName.indexOf(argumentsLiteral)
    if (argumentsStart === -1) return null
    const insideArguments = afterName.slice(argumentsStart + argumentsLiteral.length)
    if (insideArguments.length === 0) return { kind: 'literal', current: '', expected: '"' }

    const unclosedParameter = unclosedParameterName(insideArguments)
    if (unclosedParameter !== null) {
        return selectedTool.parameters.length > 0
            ? { kind: 'choice', partial: unclosedParameter, choices: selectedTool.parameters }
            : null
    }

    return null
}

function unclosedJsonStringAfter(text: string, marker: string): string | null {
    const start = text.lastIndexOf(marker)
    if (start === -1) return null
    const valueStart = start + marker.length
    const value = text.slice(valueStart)
    return value.includes('"') ? null : value
}

function lastClosedToolName(text: string, tools: readonly NeedleToolSpec[]): NeedleToolSpec | null {
    let selected: NeedleToolSpec | null = null
    let selectedIndex = -1
    for (const tool of tools) {
        const marker = `"name":"${tool.name}"`
        const index = text.lastIndexOf(marker)
        if (index > selectedIndex) {
            selected = tool
            selectedIndex = index
        }
    }
    return selected
}

function unclosedParameterName(insideArguments: string): string | null {
    const startMarkers = ['{"', ',"']
    let bestStart = -1
    for (const marker of startMarkers) {
        const index = insideArguments.lastIndexOf(marker)
        if (index > bestStart) bestStart = index
    }

    const valueStart = bestStart === -1
        ? (insideArguments.startsWith('"') ? 0 : -1)
        : bestStart + 1
    if (valueStart === -1) return null

    const value = insideArguments.slice(valueStart + 1)
    if (value.includes('"')) return null
    return value
}

function tokenIdsForConstraint(
    constraint: NeedleTokenConstraint,
    vocabulary: readonly string[],
    config: NeedleConfig
): ReadonlySet<number> {
    const allowed = new Set<number>()
    for (let token = 0; token < Math.min(vocabulary.length, config.vocabSize); token++) {
        const text = pieceToText(tokenPiece(token, vocabulary))
        if (!text) continue
        if (constraint.kind === 'literal') {
            const candidate = appendNeedleTokenText(constraint.current, text)
            if (constraint.expected.startsWith(candidate) || candidate.startsWith(constraint.expected)) allowed.add(token)
        } else {
            const candidate = appendNeedleTokenText(constraint.partial, text)
            if (constraint.choices.some((choice) => choice.startsWith(candidate) || candidate.startsWith(choice))) {
                allowed.add(token)
            }
        }
    }
    return allowed.size > 0 ? allowed : defaultAllowedNeedleTokenIds(vocabulary, config, true)
}

function appendNeedleTokenText(current: string, tokenText: string): string {
    return current.length === 0 ? tokenText.trimStart() : current + tokenText
}

function defaultAllowedNeedleTokenIds(
    vocabulary: readonly string[],
    config: NeedleConfig,
    blockToolCall: boolean
): ReadonlySet<number> {
    const blocked = new Set([config.padTokenId, config.toolsTokenId, 2, 3])
    if (blockToolCall) blocked.add(config.toolCallTokenId)
    const allowed = new Set<number>()
    for (let token = 0; token < Math.min(vocabulary.length, config.vocabSize); token++) {
        if (!blocked.has(token)) allowed.add(token)
    }
    return allowed
}

function encodeWithVocabulary(
    text: string,
    pieceToId: ReadonlyMap<string, number>,
    scores: readonly number[],
    byteTokenIds: ReadonlyMap<number, number>,
    maxPieceLength: number,
    maxTokens: number
): number[] {
    const normalized = sentencePieceText(text)
    if (!normalized) return []

    const bestScores = new Array<number>(normalized.length + 1).fill(Number.NEGATIVE_INFINITY)
    const backpointers = new Array<{ start: number; tokens: number[] } | null>(normalized.length + 1).fill(null)
    bestScores[0] = 0

    for (let start = 0; start < normalized.length; start++) {
        if (!Number.isFinite(bestScores[start])) continue

        for (let length = 1; length <= maxPieceLength && start + length <= normalized.length; length++) {
            const piece = normalized.slice(start, start + length)
            const token = pieceToId.get(piece)
            if (token === undefined || isSpecialPiece(piece)) continue
            const score = bestScores[start] + (scores[token] ?? -10_000)
            const end = start + length
            if (score > bestScores[end]) {
                bestScores[end] = score
                backpointers[end] = { start, tokens: [token] }
            }
        }

        const fallback = byteFallbackTokens(normalized[start] ?? '', byteTokenIds)
        if (fallback.length > 0) {
            const end = start + 1
            const score = bestScores[start] - 10_000
            if (score > bestScores[end]) {
                bestScores[end] = score
                backpointers[end] = { start, tokens: fallback }
            }
        }
    }

    const tokens: number[] = []
    for (let cursor = normalized.length; cursor > 0;) {
        const pointer = backpointers[cursor]
        if (!pointer) return hashTextToTokens(text, scores.length, maxTokens)
        tokens.push(...pointer.tokens.slice().reverse())
        cursor = pointer.start
    }

    return tokens.reverse().slice(0, maxTokens)
}

function sentencePieceText(text: string): string {
    const clean = text
        .normalize('NFKC')
        .replace(/\s+/g, ' ')
        .trim()
    return clean ? `\u2581${clean.replace(/ /g, '\u2581')}` : ''
}

function byteFallbackTokens(text: string, byteTokenIds: ReadonlyMap<number, number>): number[] {
    const bytes = new TextEncoder().encode(text)
    const tokens: number[] = []
    for (const byte of bytes) {
        const token = byteTokenIds.get(byte)
        if (token === undefined) return []
        tokens.push(token)
    }
    return tokens
}

function isSpecialPiece(piece: string): boolean {
    return /^<[^>]+>$/.test(piece)
}

function topK(values: number[], k: number): Array<{ index: number; value: number }> {
    return values
        .map((value, index) => ({ index, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, k)
}

function decodeTensorValues(buffer: ArrayBuffer, dtype: NeedleTensorManifest['tensors'][string]['dtype']): Float32Array {
    if (dtype === 'float16') return float16ToFloat32(new Uint16Array(buffer))
    if (dtype === 'bfloat16') return bfloat16ToFloat32(new Uint16Array(buffer))
    return new Float32Array(buffer)
}

function float16ToFloat32(input: Uint16Array): Float32Array {
    const out = new Float32Array(input.length)
    for (let i = 0; i < input.length; i++) {
        const h = input[i]
        const sign = (h & 0x8000) ? -1 : 1
        const exp = (h >> 10) & 0x1f
        const frac = h & 0x03ff
        if (exp === 0) {
            out[i] = sign * Math.pow(2, -14) * (frac / 1024)
        } else if (exp === 31) {
            out[i] = frac ? Number.NaN : sign * Number.POSITIVE_INFINITY
        } else {
            out[i] = sign * Math.pow(2, exp - 15) * (1 + frac / 1024)
        }
    }
    return out
}

function bfloat16ToFloat32(input: Uint16Array): Float32Array {
    const out = new Float32Array(input.length)
    const view = new DataView(out.buffer)
    for (let i = 0; i < input.length; i++) {
        view.setUint32(i * 4, input[i] << 16, true)
    }
    return out
}
