export const MNIST_SIZE = 28
export const PATCH_SIZE = 4
export const AGENT_GRID = MNIST_SIZE / PATCH_SIZE
export const AGENT_COUNT = AGENT_GRID * AGENT_GRID
export const DIGIT_COUNT = 10

export interface SheafAdmmParams {
    rho: number
    gamma: number
    diffusionSteps: number
}

export interface PatchAgent {
    id: number
    row: number
    col: number
    ink: number
    target: number[]
    reliability: number[]
}

export interface SheafEdge {
    from: number
    to: number
    weight: number
    mismatch: number
}

export interface SheafAdmmSnapshot {
    iteration: number
    x: number[][]
    z: number[][]
    u: number[][]
    prediction: number
    localPrediction: number
    probabilities: number[]
    primalResidual: number
    consensusResidual: number
    dualEnergy: number
    edges: SheafEdge[]
}

export interface SheafAdmmSession {
    agents: PatchAgent[]
    edges: SheafEdge[]
    initial: SheafAdmmSnapshot
}

export interface MnistClassPrototypes {
    means: Float32Array
    counts: number[]
}

function zeros2D(rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0))
}

function clone2D(values: number[][]): number[][] {
    return values.map(row => [...row])
}

function argmax(values: number[]): number {
    let bestIndex = 0
    let bestValue = values[0] ?? Number.NEGATIVE_INFINITY
    for (let i = 1; i < values.length; i++) {
        if (values[i] > bestValue) {
            bestValue = values[i]
            bestIndex = i
        }
    }
    return bestIndex
}

function softmax(logits: number[], temperature = 1): number[] {
    const scaled = logits.map(value => value / temperature)
    const maxValue = Math.max(...scaled)
    const exp = scaled.map(value => Math.exp(value - maxValue))
    const sum = exp.reduce((acc, value) => acc + value, 0)
    return exp.map(value => value / Math.max(sum, 1e-9))
}

function meanLogits(values: number[][]): number[] {
    const out = Array.from({ length: DIGIT_COUNT }, () => 0)
    if (values.length === 0) return out
    for (const row of values) {
        for (let c = 0; c < DIGIT_COUNT; c++) out[c] += row[c]
    }
    for (let c = 0; c < DIGIT_COUNT; c++) out[c] /= values.length
    return out
}

function makeEdges(agents: PatchAgent[]): SheafEdge[] {
    const edges: SheafEdge[] = []
    for (const agent of agents) {
        const right = agent.col < AGENT_GRID - 1 ? agent.id + 1 : null
        const down = agent.row < AGENT_GRID - 1 ? agent.id + AGENT_GRID : null
        for (const neighbor of [right, down]) {
            if (neighbor === null) continue
            const other = agents[neighbor]
            const inkBalance = 1 - Math.abs(agent.ink - other.ink)
            edges.push({
                from: agent.id,
                to: other.id,
                weight: 0.45 + 0.55 * Math.max(0, Math.min(1, inkBalance)),
                mismatch: 0,
            })
        }
    }
    return edges
}

function patchMse(
    pixels: Float32Array,
    prototypes: MnistClassPrototypes,
    digit: number,
    agentRow: number,
    agentCol: number
): number {
    let sum = 0
    const protoOffset = digit * MNIST_SIZE * MNIST_SIZE
    for (let r = 0; r < PATCH_SIZE; r++) {
        for (let c = 0; c < PATCH_SIZE; c++) {
            const imageIndex = (agentRow * PATCH_SIZE + r) * MNIST_SIZE + agentCol * PATCH_SIZE + c
            const diff = pixels[imageIndex] - prototypes.means[protoOffset + imageIndex]
            sum += diff * diff
        }
    }
    return sum / (PATCH_SIZE * PATCH_SIZE)
}

function patchInk(pixels: Float32Array, agentRow: number, agentCol: number): number {
    let sum = 0
    for (let r = 0; r < PATCH_SIZE; r++) {
        for (let c = 0; c < PATCH_SIZE; c++) {
            const imageIndex = (agentRow * PATCH_SIZE + r) * MNIST_SIZE + agentCol * PATCH_SIZE + c
            sum += pixels[imageIndex]
        }
    }
    return sum / (PATCH_SIZE * PATCH_SIZE)
}

function computeResiduals(
    agents: PatchAgent[],
    edges: SheafEdge[],
    x: number[][],
    z: number[][],
    u: number[][]
): Pick<SheafAdmmSnapshot, 'primalResidual' | 'consensusResidual' | 'dualEnergy' | 'edges'> {
    let primal = 0
    let dual = 0
    for (let i = 0; i < AGENT_COUNT; i++) {
        for (let c = 0; c < DIGIT_COUNT; c++) {
            const diff = x[i][c] - z[i][c]
            primal += diff * diff
            dual += u[i][c] * u[i][c]
        }
    }

    let consensus = 0
    const nextEdges = edges.map(edge => {
        const fromAgent = agents[edge.from]
        const toAgent = agents[edge.to]
        let edgeSum = 0
        for (let c = 0; c < DIGIT_COUNT; c++) {
            const err = fromAgent.reliability[c] * z[edge.from][c] - toAgent.reliability[c] * z[edge.to][c]
            edgeSum += err * err
        }
        const mismatch = Math.sqrt(edgeSum / DIGIT_COUNT)
        consensus += edge.weight * edgeSum
        return { ...edge, mismatch }
    })

    return {
        primalResidual: Math.sqrt(primal / (AGENT_COUNT * DIGIT_COUNT)),
        consensusResidual: Math.sqrt(consensus / Math.max(1, edges.length * DIGIT_COUNT)),
        dualEnergy: Math.sqrt(dual / (AGENT_COUNT * DIGIT_COUNT)),
        edges: nextEdges,
    }
}

function makeSnapshot(
    iteration: number,
    agents: PatchAgent[],
    edges: SheafEdge[],
    x: number[][],
    z: number[][],
    u: number[][]
): SheafAdmmSnapshot {
    const logits = meanLogits(z)
    const localLogits = meanLogits(x)
    const residuals = computeResiduals(agents, edges, x, z, u)

    return {
        iteration,
        x: clone2D(x),
        z: clone2D(z),
        u: clone2D(u),
        prediction: argmax(logits),
        localPrediction: argmax(localLogits),
        probabilities: softmax(logits, 0.65),
        ...residuals,
    }
}

export function buildMnistPrototypes(
    images: Float32Array,
    labels: Int32Array,
    maxImages = labels.length
): MnistClassPrototypes {
    const means = new Float32Array(DIGIT_COUNT * MNIST_SIZE * MNIST_SIZE)
    const counts = Array.from({ length: DIGIT_COUNT }, () => 0)
    const limit = Math.min(labels.length, maxImages)

    for (let sample = 0; sample < limit; sample++) {
        const label = labels[sample]
        if (label < 0 || label >= DIGIT_COUNT) continue
        counts[label] += 1
        const sourceOffset = sample * MNIST_SIZE * MNIST_SIZE
        const targetOffset = label * MNIST_SIZE * MNIST_SIZE
        for (let p = 0; p < MNIST_SIZE * MNIST_SIZE; p++) {
            means[targetOffset + p] += images[sourceOffset + p]
        }
    }

    for (let digit = 0; digit < DIGIT_COUNT; digit++) {
        const divisor = Math.max(1, counts[digit])
        const offset = digit * MNIST_SIZE * MNIST_SIZE
        for (let p = 0; p < MNIST_SIZE * MNIST_SIZE; p++) {
            means[offset + p] /= divisor
        }
    }

    return { means, counts }
}

export function createSheafAdmmSession(
    pixels: Float32Array,
    prototypes: MnistClassPrototypes
): SheafAdmmSession {
    const agents: PatchAgent[] = []

    for (let row = 0; row < AGENT_GRID; row++) {
        for (let col = 0; col < AGENT_GRID; col++) {
            const id = row * AGENT_GRID + col
            const ink = patchInk(pixels, row, col)
            const target = Array.from({ length: DIGIT_COUNT }, (_, digit) => {
                const mse = patchMse(pixels, prototypes, digit, row, col)
                return -18 * mse + 0.45 * ink
            })
            const localProbs = softmax(target, 0.55)
            const reliability = localProbs.map(prob => (0.2 + 0.8 * Math.sqrt(prob)) * (0.35 + 0.65 * Math.sqrt(ink)))
            agents.push({ id, row, col, ink, target, reliability })
        }
    }

    const x = agents.map(agent => [...agent.target])
    const z = clone2D(x)
    const u = zeros2D(AGENT_COUNT, DIGIT_COUNT)
    const edges = makeEdges(agents)

    return {
        agents,
        edges,
        initial: makeSnapshot(0, agents, edges, x, z, u),
    }
}

export function stepSheafAdmm(
    agents: PatchAgent[],
    previous: SheafAdmmSnapshot,
    params: SheafAdmmParams
): SheafAdmmSnapshot {
    const rho = Math.max(0.01, params.rho)
    const gamma = Math.max(0.001, params.gamma)
    const x = zeros2D(AGENT_COUNT, DIGIT_COUNT)
    let z = zeros2D(AGENT_COUNT, DIGIT_COUNT)
    const u = zeros2D(AGENT_COUNT, DIGIT_COUNT)

    for (let i = 0; i < AGENT_COUNT; i++) {
        for (let c = 0; c < DIGIT_COUNT; c++) {
            x[i][c] = (agents[i].target[c] + rho * (previous.z[i][c] - previous.u[i][c])) / (1 + rho)
            z[i][c] = x[i][c] + previous.u[i][c]
        }
    }

    for (let step = 0; step < params.diffusionSteps; step++) {
        const grad = zeros2D(AGENT_COUNT, DIGIT_COUNT)
        for (const edge of previous.edges) {
            const fromAgent = agents[edge.from]
            const toAgent = agents[edge.to]
            for (let c = 0; c < DIGIT_COUNT; c++) {
                const fromRestriction = fromAgent.reliability[c]
                const toRestriction = toAgent.reliability[c]
                const err = fromRestriction * z[edge.from][c] - toRestriction * z[edge.to][c]
                grad[edge.from][c] += edge.weight * fromRestriction * err
                grad[edge.to][c] -= edge.weight * toRestriction * err
            }
        }

        z = z.map((row, i) => row.map((value, c) => value - gamma * grad[i][c]))
    }

    for (let i = 0; i < AGENT_COUNT; i++) {
        for (let c = 0; c < DIGIT_COUNT; c++) {
            u[i][c] = previous.u[i][c] + x[i][c] - z[i][c]
        }
    }

    return makeSnapshot(previous.iteration + 1, agents, previous.edges, x, z, u)
}

export function runSheafAdmm(
    session: SheafAdmmSession,
    params: SheafAdmmParams,
    iterations: number
): SheafAdmmSnapshot[] {
    const history = [session.initial]
    for (let i = 0; i < iterations; i++) {
        history.push(stepSheafAdmm(session.agents, history[history.length - 1], params))
    }
    return history
}
