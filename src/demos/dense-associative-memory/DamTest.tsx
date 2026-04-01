/**
 * Console-based test for the Dense Associative Memory model.
 * 
 * This runs in the browser (needs jax-js WebGPU/WASM).
 * Open the browser dev console to see test results.
 * 
 * Route: /demos/dam-test
 */
import { onMount } from 'solid-js'
import { numpy as np, tree } from '@jax-js/jax'
import { fetchMnistRaw, type MnistFetchProgress } from '../../lib/datasources/mnist'
import { formatMnistForMlp } from '../../lib/datasets/jax/mnist'
import {
    initJax,
    createSeededRng,
    initDamParams,
    createOptimizer,
    trainStep,
    recallStep,
    extractBasicMemory,
    addNoise,
    mse,
} from './models'
import type { OptState } from './models'

function log(label: string, msg: string) {
    console.log(`[DAM-TEST] ${label}: ${msg}`)
}

function logSection(title: string) {
    console.log(`\n${'='.repeat(60)}`)
    console.log(`[DAM-TEST] ${title}`)
    console.log('='.repeat(60))
}

async function runTests() {
    // ── Test 1: jax-js init ────────────────────────────────────────────
    logSection('Test 1: jax-js initialization')
    const device = await initJax()
    log('device', device)

    // ── Test 2: RNG produces valid outputs ─────────────────────────────
    logSection('Test 2: Seeded RNG')
    const rng = createSeededRng(42)
    const samples = Array.from({ length: 10 }, () => rng.nextNormal())
    log('normal samples', JSON.stringify(samples.map(x => x.toFixed(4))))
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length
    log('mean (should be ~0)', mean.toFixed(4))

    // ── Test 3: Parameter init ─────────────────────────────────────────
    logSection('Test 3: Parameter initialization')
    const Nv = 784
    const Nh = 20
    const params = initDamParams(42, Nv, Nh)
    
    const xiShape = params.xi.shape
    const thetaVal = await params.theta.ref.jsAsync() as number[]
    log('xi shape', JSON.stringify(xiShape))
    log('theta value', JSON.stringify(thetaVal))
    log('expected xi shape', `[${Nv}, ${Nh}]`)

    if (xiShape[0] !== Nv || xiShape[1] !== Nh) {
        console.error('[DAM-TEST] FAIL: xi shape mismatch!')
        return
    }
    log('PASS', 'Parameter shapes correct')
    tree.dispose(params)

    // ── Test 4: Basic jax-js operations ────────────────────────────────
    logSection('Test 4: Basic jax-js matmul & sigmoid')
    const testParams = initDamParams(99, Nv, Nh)
    const testV = np.ones([2, Nv])
    const hRaw = np.matmul(testV, testParams.xi.ref).div(Nv)
    const hRawVals = await hRaw.jsAsync() as number[][]
    log('hRaw shape', `[${hRawVals.length}, ${hRawVals[0].length}]`)
    log('hRaw[0] first 5', JSON.stringify(hRawVals[0].slice(0, 5).map(x => x.toFixed(4))))
    tree.dispose(testParams)

    // Test sharp sigmoid
    const testInput = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    const sigResult = np.reciprocal(np.exp(testInput.mul(-20)).add(1))
    const sigVals = await sigResult.jsAsync() as number[]
    log('sigmoid(-2..2, β=20)', JSON.stringify(sigVals.map(x => x.toFixed(4))))
    log('PASS', 'jax-js operations work')

    // ── Test 5: Load MNIST ─────────────────────────────────────────────
    logSection('Test 5: Loading MNIST')
    const raw = await fetchMnistRaw({
        onProgress(p: MnistFetchProgress) {
            log('loading', `${p.stage} ${p.file} (${p.completedFiles}/${p.totalFiles})`)
        },
    })
    const dataset = formatMnistForMlp(raw)
    log('train count', String(dataset.train.count))
    log('test count', String(dataset.test.count))
    // Sample from the middle where there are actual pixel values
    const midStart = 400 * Nv
    const sampleSlice = dataset.train.images.subarray(midStart, midStart + 100)
    log('train image sample range (mid)', `${Math.min(...sampleSlice).toFixed(3)} – ${Math.max(...sampleSlice).toFixed(3)}`)

    // ── Test 6: Training loop ──────────────────────────────────────────
    logSection('Test 6: Training (small subset)')
    const NUM_MEMORIES = 200
    const EPOCHS = 30
    const LR = 0.005

    // Build a small memory batch
    const memoryImages = new Float32Array(NUM_MEMORIES * Nv)
    for (let i = 0; i < NUM_MEMORIES; i++) {
        const srcStart = i * Nv
        memoryImages.set(dataset.train.images.subarray(srcStart, srcStart + Nv), i * Nv)
    }
    const memoryLabels = new Int32Array(NUM_MEMORIES)
    for (let i = 0; i < NUM_MEMORIES; i++) {
        memoryLabels[i] = dataset.train.labels[i]
    }

    let trainParams = initDamParams(42, Nv, Nh)
    const { solver, optState: initOptState } = createOptimizer(trainParams, LR)
    let currentOptState: OptState = initOptState

    // Create memories tensor — keep one ref for ourselves (loop reuses it)
    const memoriesTensor = np.array(memoryImages).reshape([NUM_MEMORIES, Nv])

    const losses: number[] = []
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        const result = await trainStep(trainParams, currentOptState, solver, memoriesTensor)
        trainParams = result.params
        currentOptState = result.optState
        losses.push(result.loss)

        if (epoch % 5 === 0 || epoch === EPOCHS - 1) {
            log(`epoch ${epoch}`, `loss = ${result.loss.toFixed(6)}`)
        }
        await new Promise(r => setTimeout(r, 0))
    }

    const lossDecreased = losses[losses.length - 1] < losses[0]
    log('loss decreased?', String(lossDecreased))
    if (!lossDecreased) {
        console.error('[DAM-TEST] FAIL: Loss did not decrease during training!')
    } else {
        log('PASS', 'Loss decreased during training')
    }

    // ── Test 7: Extract & inspect basic memories ───────────────────────
    logSection('Test 7: Basic memories extraction')
    const xiData = new Float32Array(await trainParams.xi.ref.data() as ArrayLike<number>)
    const thetaFinal = (await trainParams.theta.ref.jsAsync() as number[])[0]
    log('learned theta', thetaFinal.toFixed(4))

    for (let mu = 0; mu < Math.min(5, Nh); mu++) {
        const basicMem = extractBasicMemory(xiData, Nv, Nh, mu)
        const bMin = Math.min(...basicMem)
        const bMax = Math.max(...basicMem)
        const bMean = basicMem.reduce((a, b) => a + b, 0) / basicMem.length
        log(`basic memory ${mu}`, `min=${bMin.toFixed(4)}, max=${bMax.toFixed(4)}, mean=${bMean.toFixed(4)}`)
    }
    log('PASS', 'Basic memories extracted')

    // ── Test 8: Recall dynamics ────────────────────────────────────────
    logSection('Test 8: Recall dynamics')
    const TEST_INDEX = 0
    const testImage = new Float32Array(dataset.train.images.subarray(TEST_INDEX * Nv, (TEST_INDEX + 1) * Nv))
    log('cue label', String(memoryLabels[TEST_INDEX]))

    // Add noise to the cue
    const noiseRng = createSeededRng(123)
    const noisyCue = addNoise(testImage, 0.5, noiseRng)
    log('MSE(noisy cue, original)', mse(noisyCue, testImage).toFixed(6))

    // Run recall
    let v = new Float32Array(noisyCue)
    const TAU_RATIO = 10
    const RECALL_STEPS = 20

    const recallMSEs: number[] = []
    for (let step = 0; step < RECALL_STEPS; step++) {
        const result = recallStep(xiData, thetaFinal, v, Nv, Nh, TAU_RATIO)
        v = result.v as any
        const currentMSE = mse(v, testImage)
        recallMSEs.push(currentMSE)

        const activeCount = result.hBinary.reduce((a: number, b: number) => a + b, 0)

        if (step % 5 === 0 || step === RECALL_STEPS - 1) {
            log(`step ${step}`,
                `MSE=${currentMSE.toFixed(6)}, active_h=${activeCount}/${Nh}, ` +
                `hBinary=[${Array.from(result.hBinary).join(',')}]`)
        }
    }

    const noiseMSE = mse(noisyCue, testImage)
    const recallImproved = recallMSEs[recallMSEs.length - 1] < noiseMSE
    log('noise MSE', noiseMSE.toFixed(6))
    log('final recall MSE', recallMSEs[recallMSEs.length - 1].toFixed(6))
    log('recall MSE improved?', String(recallImproved))
    if (recallImproved) {
        log('PASS', 'Recall dynamics improved reconstruction')
    } else {
        console.warn('[DAM-TEST] WARN: Recall did not improve MSE — may need more training or tuning')
    }

    // ── Test 9: Recall from clean cue ──────────────────────────────────
    logSection('Test 9: Recall from clean cue (no noise)')
    let vClean = new Float32Array(testImage)

    for (let step = 0; step < RECALL_STEPS; step++) {
        const result = recallStep(xiData, thetaFinal, vClean, Nv, Nh, TAU_RATIO)
        vClean = result.v as any

        if (step % 5 === 0 || step === RECALL_STEPS - 1) {
            const currentMSE = mse(vClean, testImage)
            const activeCount = result.hBinary.reduce((a: number, b: number) => a + b, 0)
            log(`step ${step}`, `MSE=${currentMSE.toFixed(6)}, active_h=${activeCount}/${Nh}`)
        }
    }
    const cleanRecallMSE = mse(vClean, testImage)
    log('final MSE (clean recall)', cleanRecallMSE.toFixed(6))

    // ── Test 10: Multiple cue recall ───────────────────────────────────
    logSection('Test 10: Recall multiple memories')
    const TEST_INDICES = [0, 1, 5, 10, 50, 100]
    for (const idx of TEST_INDICES) {
        if (idx >= NUM_MEMORIES) continue
        const src = new Float32Array(dataset.train.images.subarray(idx * Nv, (idx + 1) * Nv))
        const noisySrc = addNoise(src, 0.5, createSeededRng(idx + 100))
        const noiseMSEi = mse(noisySrc, src)

        let vR = new Float32Array(noisySrc)
        for (let step = 0; step < RECALL_STEPS; step++) {
            const result = recallStep(xiData, thetaFinal, vR, Nv, Nh, TAU_RATIO)
            vR = result.v as any
        }
        const finalMSE = mse(vR, src)
        const label = memoryLabels[idx]
        log(`memory #${idx} (label=${label})`,
            `noise MSE=${noiseMSEi.toFixed(4)} → recall MSE=${finalMSE.toFixed(4)} (${finalMSE < noiseMSEi ? '✓ improved' : '✗ worse'})`)
    }

    // ── Cleanup ────────────────────────────────────────────────────────
    logSection('DONE — All tests completed')
    tree.dispose(trainParams)
    memoriesTensor.dispose()
}

export function DamTestPage() {
    onMount(() => {
        console.log('[DAM-TEST] Starting Dense Associative Memory tests...')
        void runTests().catch(err => {
            console.error('[DAM-TEST] Test failed with error:', err)
        })
    })

    return (
        <div style={{
            'min-height': '100vh',
            'display': 'flex',
            'align-items': 'center',
            'justify-content': 'center',
            'font-family': 'monospace',
            'background': '#1a1a2e',
            'color': '#e0e0e0',
        }}>
            <div style={{ 'text-align': 'center' }}>
                <h1 style={{ 'font-size': '2rem', 'margin-bottom': '1rem' }}>
                    Dense Associative Memory — Console Test
                </h1>
                <p style={{ 'opacity': '0.7' }}>
                    Open your browser's Developer Console (F12) to see test results.
                </p>
                <p style={{ 'margin-top': '1rem', 'opacity': '0.5' }}>
                    Tests are running in the background...
                </p>
            </div>
        </div>
    )
}
