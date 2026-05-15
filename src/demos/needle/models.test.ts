import { describe, expect, test } from 'vitest'
import { tree } from '@jax-js/jax'
import {
    initJax,
    createNeedleTokenizer,
    decodeNeedleTokenPieces,
    initNeedleParams,
    makeNeedleAllowedTokenIds,
    makeNeedleSourceTokens,
    makeNeedleTargetTokens,
    needleForward,
    needleParamCount,
    runNeedleProbe,
    tokenPiece,
    type NeedleConfig,
} from './models'

const TINY_CONFIG: NeedleConfig = {
    vocabSize: 32,
    dModel: 8,
    numHeads: 2,
    numKvHeads: 1,
    numEncoderLayers: 1,
    numDecoderLayers: 1,
    maxSeqLen: 32,
    padTokenId: 0,
    eosTokenId: 1,
    toolCallTokenId: 4,
    toolsTokenId: 5,
    ropeTheta: 10_000,
}

const TINY_STACK_CONFIG: NeedleConfig = {
    ...TINY_CONFIG,
    numEncoderLayers: 2,
    numDecoderLayers: 2,
}

describe('Needle jax-js model', () => {
    test('counts the SAN parameter tree at 26M scale', () => {
        expect(needleParamCount()).toBe(26_233_372)
        expect(needleParamCount(TINY_CONFIG)).toBe(899)
    })

    test('builds bounded browser-side probe tokens', () => {
        const source = makeNeedleSourceTokens(
            'What is the weather in San Francisco?',
            '[{"name":"get_weather","parameters":{"location":"string"}}]',
            TINY_CONFIG
        )
        const target = makeNeedleTargetTokens('', TINY_CONFIG)

        expect(source.length).toBeGreaterThan(1)
        expect(source).toContain(TINY_CONFIG.toolsTokenId)
        expect([...source].every(token => token >= 0 && token < TINY_CONFIG.vocabSize)).toBe(true)
        expect([...target]).toEqual([TINY_CONFIG.eosTokenId])
    })

    test('appends generated token ids to the decoder target', () => {
        const target = makeNeedleTargetTokens('', TINY_CONFIG, [7, 12, 31])

        expect([...target]).toEqual([
            TINY_CONFIG.eosTokenId,
            7,
            12,
            31,
        ])
    })

    test('decodes generated token ids through a SentencePiece vocabulary', () => {
        const vocabulary = ['<pad>', '</s>', '<s>', '<unk>', '<tool_call>', '<tools>', '<0x20>', '▁weather', 'ing']

        expect(tokenPiece(7, vocabulary)).toBe('▁weather')
        expect(decodeNeedleTokenPieces([7, 8], vocabulary)).toBe('weathering')
        expect(decodeNeedleTokenPieces([6, 7], vocabulary)).toBe('weather')
    })

    test('uses the Needle vocabulary encoder for source and decoder text', () => {
        const tokenizer = createNeedleTokenizer([
            '<pad>\t0',
            '</s>\t0',
            '<s>\t0',
            '<unk>\t0',
            '<tool_call>\t0',
            '<tools>\t0',
            '<0x20>\t0',
            '<0x5F>\t0',
            '▁weather\t-1',
            '▁[{"\t-1',
            'name\t-1',
            '":"\t-1',
            'get\t-1',
            '_\t-1',
            'weather\t-1',
        ].join('\n'))

        const source = makeNeedleSourceTokens('weather', '[{"name":"get_weather"}]', TINY_CONFIG, tokenizer)
        const target = makeNeedleTargetTokens('[{"name":"get_weather', TINY_CONFIG, [], tokenizer)

        expect([...source].slice(0, 2)).toEqual([8, TINY_CONFIG.toolsTokenId])
        expect([...target]).toEqual([TINY_CONFIG.eosTokenId, TINY_CONFIG.toolCallTokenId, 9, 10, 11, 12, 13, 14])
    })

    test('constrains generated tokens to upstream tool-call boundaries', () => {
        const vocabulary = [
            '<pad>',
            '</s>',
            '<s>',
            '<unk>',
            '<tool_call>',
            '<tools>',
            '▁[{"',
            'name',
            '":"',
            'get',
            '_',
            'weather',
            'calculator',
            '","',
            'arguments',
            '":{',
            'location',
            'input',
        ]
        const toolsJson = JSON.stringify([
            { name: 'get_weather', parameters: { location: { type: 'string' } } },
            { name: 'calculator', parameters: { input: { type: 'string' } } },
        ])

        const first = makeNeedleAllowedTokenIds(toolsJson, '', [], TINY_CONFIG, vocabulary)
        const name = makeNeedleAllowedTokenIds(toolsJson, '', [4, 6, 7, 8], TINY_CONFIG, vocabulary)
        const parameter = makeNeedleAllowedTokenIds(
            toolsJson,
            '[{"name":"calculator","arguments":{"',
            [],
            TINY_CONFIG,
            vocabulary
        )

        expect(first?.has(TINY_CONFIG.toolCallTokenId)).toBe(true)
        expect(first?.size).toBe(1)
        expect(name?.has(9)).toBe(true)
        expect(name?.has(12)).toBe(true)
        expect(name?.has(16)).toBe(false)
        expect(parameter?.has(17)).toBe(true)
        expect(parameter?.has(16)).toBe(false)
    })

    test('runs a tiny encoder-decoder forward pass with tied output logits', async () => {
        await initJax()
        const params = initNeedleParams(7, TINY_CONFIG)
        const source = makeNeedleSourceTokens('email jane hello', '[{"name":"send_email"}]', TINY_CONFIG)
        const target = makeNeedleTargetTokens('', TINY_CONFIG)

        const probe = await runNeedleProbe(params, source, target, TINY_CONFIG)

        expect(probe.srcLength).toBe(source.length)
        expect(probe.tgtLength).toBe(target.length)
        expect(probe.nextToken).toBeGreaterThanOrEqual(0)
        expect(probe.nextToken).toBeLessThan(TINY_CONFIG.vocabSize)
        expect(probe.topTokens).toHaveLength(8)
        tree.dispose(params)
    })

    test('produces logits with expected source and target dimensions', async () => {
        await initJax()
        const params = initNeedleParams(11, TINY_CONFIG)
        const src = makeNeedleSourceTokens('stock price please', '[{"name":"get_stock_price"}]', TINY_CONFIG)
        const tgt = makeNeedleTargetTokens('{"', TINY_CONFIG)

        const { numpy: np } = await import('@jax-js/jax')
        const output = needleForward(
            tree.ref(params),
            np.array(src, { dtype: np.int32 }).reshape([1, src.length]),
            np.array(tgt, { dtype: np.int32 }).reshape([1, tgt.length]),
            TINY_CONFIG
        )

        expect(output.encoderOut.shape).toEqual([1, src.length, TINY_CONFIG.dModel])
        expect(output.logits.shape).toEqual([1, tgt.length, TINY_CONFIG.vocabSize])
        output.encoderOut.dispose()
        output.logits.dispose()
        tree.dispose(params)
    })

    test('reuses causal masks and RoPE tables across repeated layers', async () => {
        await initJax()
        const params = initNeedleParams(19, TINY_STACK_CONFIG)
        const source = makeNeedleSourceTokens(
            'weather in singapore tomorrow morning',
            '[{"name":"get_weather"},{"name":"get_calendar"}]',
            TINY_STACK_CONFIG
        )
        const target = makeNeedleTargetTokens('', TINY_STACK_CONFIG)

        const probe = await runNeedleProbe(params, source, target, TINY_STACK_CONFIG)

        expect(probe.topTokens).toHaveLength(8)
        expect(probe.nextToken).toBeGreaterThanOrEqual(0)
        expect(probe.nextToken).toBeLessThan(TINY_STACK_CONFIG.vocabSize)
        tree.dispose(params)
    })
})
