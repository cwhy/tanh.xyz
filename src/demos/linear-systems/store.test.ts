import { expect, it } from 'vitest'
import { createRoot, flush } from 'solid-js'
import { createSimulationStore } from './store'
import { computeSparsity } from './math-utils'

it('isolates simulation state and calculates sparsity from the edited matrix', () => {
    let dispose!: () => void
    const [first, second] = createRoot(cleanup => {
        dispose = cleanup
        return [createSimulationStore(), createSimulationStore()]
    })
    try {
        first.setN(2)
        flush()
        first.setMatrixValue(0, 0, 0)
        first.setMatrixValue(0, 1, 0)
        flush()
        expect(first.state.sparsity).toBe(computeSparsity(first.state.W))
        expect(first.state.W[0]).toEqual([0, 0])
        expect(first.state.N).toBe(2)
        expect(second.state.N).toBe(6)
        expect(second.state.W).toEqual([])
    } finally {
        first.dispose()
        second.dispose()
        dispose()
    }
})
