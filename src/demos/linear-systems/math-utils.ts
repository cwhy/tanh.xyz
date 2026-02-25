import { numpy as np, jit } from '@jax-js/jax'

export type MatrixType = 'lattice' | 'random' | 'sparse' | 'watts-strogatz' | 'hubs'

export function generateMatrix(type: MatrixType, n: number, minVal: number, maxVal: number): number[][] {
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0))

    const randWeight = () => Math.random() * (maxVal - minVal) + minVal

    switch (type) {
        case 'lattice':
            // 1D Ring Lattice, connect to i-1, i+1, and self
            for (let i = 0; i < n; i++) {
                matrix[i][i] = randWeight()
                matrix[i][(i + 1) % n] = randWeight()
                matrix[i][(i - 1 + n) % n] = randWeight()
            }
            break
        case 'random':
            // Random values uniformly between minVal and maxVal
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    matrix[i][j] = randWeight()
                }
            }
            break
        case 'sparse':
            // Sparse random matrix (e.g. 80% zeros)
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (Math.random() > 0.8) {
                        matrix[i][j] = randWeight()
                    }
                }
            }
            break
        case 'watts-strogatz':
            const k = Math.max(1, Math.floor(n / 3))
            for (let i = 0; i < n; i++) {
                for (let j = 1; j <= k; j++) {
                    matrix[i][(i + j) % n] = randWeight()
                    matrix[i][(i - j + n) % n] = randWeight()
                }
            }
            const p = 0.25 // Slightly higher rewiring probability for small matrices
            for (let i = 0; i < n; i++) {
                for (let j = 1; j <= k; j++) {
                    if (Math.random() < p) {
                        const target = (i + j) % n
                        // If it was non-zero, disconnect it
                        if (matrix[i][target] !== 0) {
                            matrix[i][target] = 0
                            let newTarget = Math.floor(Math.random() * n)
                            while (newTarget === i || matrix[i][newTarget] !== 0) {
                                newTarget = (newTarget + 1) % n
                            }
                            matrix[i][newTarget] = randWeight()
                        }
                    }
                }
            }
            break
        case 'hubs':
            const blockSize = Math.max(2, Math.floor(n / 2))
            for (let i = 0; i < n; i++) {
                const blockIndex = Math.floor(i / blockSize)
                for (let j = 0; j < n; j++) {
                    if (Math.floor(j / blockSize) === blockIndex) {
                        matrix[i][j] = randWeight()
                    } else {
                        // Sparse cross-block connections (e.g. 10% chance)
                        if (Math.random() < 0.1) {
                            matrix[i][j] = randWeight()
                        }
                    }
                }
            }
            break
    }

    // Constrain to bounds
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            matrix[i][j] = Math.min(Math.max(matrix[i][j], minVal), maxVal)
        }
    }

    return matrix
}

export function computeSparsity(matrix: number[][], threshold = 1e-4): number {
    if (matrix.length === 0) return 0
    const n = matrix.length
    let zeros = 0
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (Math.abs(matrix[i][j]) < threshold) {
                zeros++
            }
        }
    }
    return zeros / (n * n)
}

export const stepSimulationJit = jit((x: np.Array, W: np.Array) => {
    // W is connection matrix NxN, x is Nx1
    // We want W @ x
    return np.matmul(W, x)
})
