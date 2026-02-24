export const ADDITION_NUM_DIGITS = 10
export const ADDITION_SUM_DIGITS = 11
export const ADDITION_PROMPT_LEN = ADDITION_NUM_DIGITS + 1 + ADDITION_NUM_DIGITS + 1 // 22
export const ADDITION_TARGET_LEN = ADDITION_SUM_DIGITS + 1 // 12 (11 digits + EOS)
export const ADDITION_LM_INPUT_LEN = ADDITION_PROMPT_LEN + ADDITION_SUM_DIGITS // 33
export const ADDITION_LM_FULL_LEN = ADDITION_PROMPT_LEN + ADDITION_TARGET_LEN // 34

export const ADDITION_TOKENS = {
    '+': 10,
    '=': 11,
    '<PAD>': 12,
    '<EOS>': 13,
} as const

export const ADDITION_VOCAB_SIZE = 14
export const ADDITION_INPUT_SIZE = ADDITION_PROMPT_LEN * ADDITION_VOCAB_SIZE
export const ADDITION_TARGET_SIZE = ADDITION_SUM_DIGITS * 10

const POW10 = Array.from({ length: ADDITION_NUM_DIGITS + 1 }, (_, i) => 10 ** i)

export interface SeededRng {
    next: () => number
}

export interface AdditionPair {
    a: number
    b: number
    sum: number
}

export interface CurriculumPhase {
    minDigits: number
    maxDigits: number
    steps: number
}

export interface EncodedAdditionBatch {
    pairs: AdditionPair[]
    inputs: Float32Array
    targets: Float32Array
    targetDigits: Uint8Array
    batchSize: number
}

export interface EncodedAdditionLmBatch {
    pairs: AdditionPair[]
    promptTokens: Int32Array
    inputTokens: Int32Array
    labelTokens: Int32Array
    labelMask: Float32Array
    targetDigits: Uint8Array
    batchSize: number
}

export const DEFAULT_ADDITION_CURRICULUM: CurriculumPhase[] = [
    { minDigits: 1, maxDigits: 3, steps: 2000 },
    { minDigits: 1, maxDigits: 6, steps: 5000 },
    { minDigits: 1, maxDigits: 10, steps: 47000 },
]

export function createSeededRng(seed: number): SeededRng {
    let state = (seed >>> 0) || 1
    return {
        next: () => {
            state ^= state << 13
            state ^= state >>> 17
            state ^= state << 5
            return ((state >>> 0) + 1) / 4294967297
        },
    }
}

export function pairKey(a: number, b: number): string {
    return `${a}:${b}`
}

export function getCurriculumPhase(
    step: number,
    phases: CurriculumPhase[] = DEFAULT_ADDITION_CURRICULUM
): CurriculumPhase {
    let cumulative = 0
    for (const phase of phases) {
        cumulative += Math.max(1, phase.steps)
        if (step < cumulative) return phase
    }
    return phases[phases.length - 1]
}

export function sampleAdditionPairs(options: {
    count: number
    minDigits: number
    maxDigits: number
    rng: SeededRng
    reserved?: Set<string>
}): AdditionPair[] {
    const { count, minDigits, maxDigits, rng, reserved } = options
    const pairs: AdditionPair[] = []
    const seen = new Set<string>()

    while (pairs.length < count) {
        const digits = minDigits + Math.floor(rng.next() * (maxDigits - minDigits + 1))
        const maxValue = POW10[Math.max(1, Math.min(ADDITION_NUM_DIGITS, digits))]
        const a = Math.floor(rng.next() * maxValue)
        const b = Math.floor(rng.next() * maxValue)
        const key = pairKey(a, b)
        if (seen.has(key) || reserved?.has(key)) continue
        seen.add(key)
        pairs.push({ a, b, sum: a + b })
    }

    return pairs
}

export function sampleCurriculumBatch(options: {
    batchSize: number
    step: number
    rng: SeededRng
    phases?: CurriculumPhase[]
    reserved?: Set<string>
}): EncodedAdditionBatch {
    const phase = getCurriculumPhase(options.step, options.phases)
    const pairs = sampleAdditionPairs({
        count: options.batchSize,
        minDigits: phase.minDigits,
        maxDigits: phase.maxDigits,
        rng: options.rng,
        reserved: options.reserved,
    })
    return encodeAdditionPairs(pairs)
}

export function sampleCurriculumLmBatch(options: {
    batchSize: number
    step: number
    rng: SeededRng
    phases?: CurriculumPhase[]
    reserved?: Set<string>
}): EncodedAdditionLmBatch {
    const phase = getCurriculumPhase(options.step, options.phases)
    const pairs = sampleAdditionPairs({
        count: options.batchSize,
        minDigits: phase.minDigits,
        maxDigits: phase.maxDigits,
        rng: options.rng,
        reserved: options.reserved,
    })
    return encodeAdditionLmPairs(pairs)
}

export function encodeAdditionPairs(pairs: AdditionPair[]): EncodedAdditionBatch {
    const batchSize = pairs.length
    const inputs = new Float32Array(batchSize * ADDITION_INPUT_SIZE)
    const targets = new Float32Array(batchSize * ADDITION_TARGET_SIZE)
    const targetDigits = new Uint8Array(batchSize * ADDITION_SUM_DIGITS)

    for (let i = 0; i < batchSize; i++) {
        encodePairIntoMlp(pairs[i], inputs, targets, targetDigits, i)
    }

    return {
        pairs,
        inputs,
        targets,
        targetDigits,
        batchSize,
    }
}

export function encodeAdditionLmPairs(pairs: AdditionPair[]): EncodedAdditionLmBatch {
    const batchSize = pairs.length
    const promptTokens = new Int32Array(batchSize * ADDITION_PROMPT_LEN)
    const inputTokens = new Int32Array(batchSize * ADDITION_LM_INPUT_LEN)
    const labelTokens = new Int32Array(batchSize * ADDITION_LM_INPUT_LEN)
    const labelMask = new Float32Array(batchSize * ADDITION_LM_INPUT_LEN)
    const targetDigits = new Uint8Array(batchSize * ADDITION_SUM_DIGITS)

    for (let i = 0; i < batchSize; i++) {
        encodePairIntoLm(pairs[i], promptTokens, inputTokens, labelTokens, labelMask, targetDigits, i)
    }

    return {
        pairs,
        promptTokens,
        inputTokens,
        labelTokens,
        labelMask,
        targetDigits,
        batchSize,
    }
}

function encodePairIntoMlp(
    pair: AdditionPair,
    inputs: Float32Array,
    targets: Float32Array,
    targetDigits: Uint8Array,
    batchIndex: number
): void {
    const promptTokens = promptTokenIds(pair.a, pair.b)
    const reversedDigits = reversedSumDigits(pair.sum)

    const inputBase = batchIndex * ADDITION_INPUT_SIZE
    for (let p = 0; p < ADDITION_PROMPT_LEN; p++) {
        const token = promptTokens[p]
        inputs[inputBase + p * ADDITION_VOCAB_SIZE + token] = 1
    }

    const targetBase = batchIndex * ADDITION_TARGET_SIZE
    const digitBase = batchIndex * ADDITION_SUM_DIGITS
    for (let p = 0; p < ADDITION_SUM_DIGITS; p++) {
        const digit = reversedDigits[p]
        targets[targetBase + p * 10 + digit] = 1
        targetDigits[digitBase + p] = digit
    }
}

function encodePairIntoLm(
    pair: AdditionPair,
    promptTokensOut: Int32Array,
    inputTokensOut: Int32Array,
    labelTokensOut: Int32Array,
    labelMaskOut: Float32Array,
    targetDigitsOut: Uint8Array,
    batchIndex: number
): void {
    const prompt = promptTokenIds(pair.a, pair.b)
    const reversedDigits11 = reversedSumDigits(pair.sum)
    const target = [...reversedDigits11, ADDITION_TOKENS['<EOS>']]

    const full = [...prompt, ...target]
    const x = full.slice(0, ADDITION_LM_INPUT_LEN)
    const y = full.slice(1)

    const promptBase = batchIndex * ADDITION_PROMPT_LEN
    const lmBase = batchIndex * ADDITION_LM_INPUT_LEN
    const digitBase = batchIndex * ADDITION_SUM_DIGITS

    for (let i = 0; i < ADDITION_PROMPT_LEN; i++) {
        promptTokensOut[promptBase + i] = prompt[i]
    }
    for (let i = 0; i < ADDITION_LM_INPUT_LEN; i++) {
        inputTokensOut[lmBase + i] = x[i]
        labelTokensOut[lmBase + i] = y[i]
        const masked = i < ADDITION_PROMPT_LEN - 1
        if (masked) {
            labelTokensOut[lmBase + i] = -100
            labelMaskOut[lmBase + i] = 0
        } else {
            labelMaskOut[lmBase + i] = 1
        }
    }

    for (let i = 0; i < ADDITION_SUM_DIGITS; i++) {
        targetDigitsOut[digitBase + i] = reversedDigits11[i]
    }
}

function promptTokenIds(a: number, b: number): number[] {
    const aDigits = String(a).padStart(ADDITION_NUM_DIGITS, '0')
    const bDigits = String(b).padStart(ADDITION_NUM_DIGITS, '0')
    const tokens: number[] = []

    for (let i = 0; i < aDigits.length; i++) tokens.push(Number(aDigits[i]))
    tokens.push(ADDITION_TOKENS['+'])
    for (let i = 0; i < bDigits.length; i++) tokens.push(Number(bDigits[i]))
    tokens.push(ADDITION_TOKENS['='])
    return tokens
}

function reversedSumDigits(sum: number): number[] {
    const sumText = String(sum).padStart(ADDITION_SUM_DIGITS, '0')
    const digits: number[] = []
    for (let i = ADDITION_SUM_DIGITS - 1; i >= 0; i--) digits.push(Number(sumText[i]))
    return digits
}
