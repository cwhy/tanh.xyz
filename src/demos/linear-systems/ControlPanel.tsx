import { For, Show, createSignal } from 'solid-js'
import { simulationStore } from './store'
import { ConfigSlider } from '../../components/ui-parts/configs/ConfigSlider'
import { ConfigRadioGroup } from '../../components/ui-parts/configs/ConfigRadioGroup'
import { type MatrixType } from './math-utils'
import { type StateNormType } from './store'

const matrixOptions: Array<{ label: string; value: MatrixType }> = [
    { label: 'Lattice', value: 'lattice' },
    { label: 'Random', value: 'random' },
    { label: 'Sparse', value: 'sparse' },
    { label: 'Small World', value: 'watts-strogatz' },
    { label: 'Hubs', value: 'hubs' }
]

const normOptions: Array<{ label: string; value: StateNormType }> = [
    { label: 'None', value: 'none' },
    { label: 'Step Clipping', value: 'clip' },
    { label: 'Step Min/Max', value: 'step-minmax' },
    { label: 'EMA Layer Norm', value: 'ema-layernorm' },
    { label: 'Softmax', value: 'softmax' },
    { label: 'Tanh', value: 'tanh' }
]

export function ControlPanel() {
    const { state, setN, setMatrixType, setMatrixValue, setInitialStateValue, randomizeInitialState, setGlobalBounds, setStateNorm, togglePlaying, resetSimulation } = simulationStore

    const [selectedCell, setSelectedCell] = createSignal<{ r: number, c: number } | null>(null)
    const [selectedVecIdx, setSelectedVecIdx] = createSignal<number | null>(null)

    return (
        <div class="p-4 md:p-6 grid grid-cols-1 md:grid-cols-3 gap-6">

            {/* Global Settings & Simulation */}
            <div class="space-y-4">
                <h3 class="font-bold text-lg mb-2" style={{ 'font-family': "'Kalam', cursive", color: '#2d2d2d' }}>Simulation Settings</h3>

                <div class="flex gap-2">
                    <button
                        class="btn btn-primary flex-1 shadow-[4px_4px_0px_0px_#2d2d2d] border-2 border-[#2d2d2d] active:shadow-none active:translate-x-[4px] active:translate-y-[4px] transition-all"
                        style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px' }}
                        onClick={togglePlaying}
                        disabled={state.isLoading}
                    >
                        {state.isPlaying ? 'Pause' : 'Start'}
                    </button>
                    <button
                        class="btn btn-outline flex-1 border-2 border-[#2d2d2d] shadow-[4px_4px_0px_0px_#2d2d2d] active:shadow-none active:translate-x-[4px] active:translate-y-[4px] transition-all"
                        style={{ 'border-radius': '15px 255px 15px 225px / 255px 15px 225px 15px', color: '#2d2d2d' }}
                        onClick={resetSimulation}
                        disabled={state.isLoading}
                    >
                        Reset State
                    </button>
                </div>

                <div class="stat bg-white border-2 border-[#2d2d2d] shadow-[3px_3px_0px_0px_#2d2d2d] p-3" style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px' }}>
                    <div class="stat-title text-xs font-bold" style={{ color: '#2d2d2d' }}>Step</div>
                    <div class="stat-value text-xl" style={{ color: '#ff4d4d' }}>{state.history.length - 1}</div>
                </div>

                <ConfigSlider
                    label="System Size (N)"
                    value={state.N}
                    min={2}
                    max={16}
                    step={1}
                    onChange={setN}
                    disabled={state.isPlaying || state.isLoading}
                    labelInline={true}
                />

                <ConfigRadioGroup
                    label="State Normalization"
                    name="stateNorm"
                    value={state.stateNorm}
                    options={normOptions}
                    onChange={(val) => setStateNorm(val as StateNormType)}
                    controlsInline
                />

                <div class="flex gap-2">
                    <div class="flex-1">
                        <label class="label pb-1"><span class="label-text font-medium">Global Min: {state.globalMin}</span></label>
                        <input
                            type="range"
                            min="-20" max="0" step="1"
                            value={state.globalMin}
                            onInput={(e) => setGlobalBounds(parseFloat(e.currentTarget.value), state.globalMax)}
                            class="range range-xs range-accent"
                            disabled={state.isPlaying || state.isLoading}
                        />
                    </div>
                    <div class="flex-1">
                        <label class="label pb-1"><span class="label-text font-medium">Global Max: {state.globalMax}</span></label>
                        <input
                            type="range"
                            min="0" max="20" step="1"
                            value={state.globalMax}
                            onInput={(e) => setGlobalBounds(state.globalMin, parseFloat(e.currentTarget.value))}
                            class="range range-xs range-secondary"
                            disabled={state.isPlaying || state.isLoading}
                        />
                    </div>
                </div>
            </div>

            {/* Matrix Editor */}
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <h3 class="font-bold text-lg" style={{ 'font-family': "'Kalam', cursive", color: '#2d2d2d' }}>Connection Matrix W</h3>
                    <div class="text-xs font-bold bg-[#ff9c9c]/30 px-2 py-1 border border-[#2d2d2d] shadow-[2px_2px_0px_0px_#2d2d2d]" style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px', color: '#2d2d2d' }}>
                        Sparsity: {(state.sparsity * 100).toFixed(1)}%
                    </div>
                </div>

                <ConfigRadioGroup
                    label="Initialize Presets"
                    name="matrixType"
                    value={state.matrixType}
                    options={matrixOptions}
                    onChange={(val) => { setMatrixType(val as MatrixType); setSelectedCell(null) }}
                    disabled={state.isPlaying || state.isLoading}
                    controlsInline
                />

                <div class="relative overflow-auto p-2">
                    <div class="grid gap-1" style={{ 'grid-template-columns': `repeat(${state.N}, minmax(0, 1fr))` }}>
                        <For each={state.W}>
                            {(row, r) => (
                                <For each={row}>
                                    {(val, c) => (
                                        <button
                                            class={`aspect-square flex items-center justify-center text-[10px] md:text-xs font-mono font-bold border rounded-sm transition-colors ${selectedCell()?.r === r() && selectedCell()?.c === c() ? 'bg-primary text-primary-content border-primary' : val !== 0 ? 'bg-base-200 border-base-300' : 'bg-base-100 border-base-200 text-base-content/40'}`}
                                            onClick={() => setSelectedCell({ r: r(), c: c() })}
                                            title={`W[${r()}][${c()}] = ${val.toFixed(2)}`}
                                        >
                                            {Math.abs(val) > 0.01 ? val.toFixed(1) : '0'}
                                        </button>
                                    )}
                                </For>
                            )}
                        </For>
                    </div>
                </div>

                <Show when={selectedCell()}>
                    {(cell) => (
                        <div class="p-3 bg-base-200 border-2 border-[#2d2d2d] shadow-[3px_3px_0px_0px_#2d2d2d]" style={{ 'border-radius': '15px 255px 15px 225px / 255px 15px 225px 15px' }}>
                            <ConfigSlider
                                label={`Edit W[${cell().r}][${cell().c}]`}
                                value={state.W[cell().r][cell().c]}
                                min={state.globalMin}
                                max={state.globalMax}
                                step={0.1}
                                onChange={(val) => setMatrixValue(cell().r, cell().c, val)}
                                labelInline
                            />
                        </div>
                    )}
                </Show>
            </div>

            {/* Initial State Vector */}
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <h3 class="font-bold text-lg mb-2" style={{ 'font-family': "'Kalam', cursive", color: '#2d2d2d' }}>Initial Vector x₀</h3>
                    <button
                        class="btn btn-sm btn-outline border-2 border-[#2d2d2d] shadow-[2px_2px_0px_0px_#2d2d2d] active:shadow-none active:translate-x-[2px] active:translate-y-[2px] transition-all"
                        style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px', color: '#2d2d2d' }}
                        onClick={randomizeInitialState}
                        disabled={state.isLoading}
                    >
                        Randomize x₀
                    </button>
                </div>
                <div class="flex flex-wrap gap-2">
                    <For each={state.x0}>
                        {(val, idx) => (
                            <button
                                class={`px-2 py-1 text-xs md:text-sm font-mono font-bold border-2 transition-colors shadow-[2px_2px_0px_0px_#2d2d2d] ${selectedVecIdx() === idx() ? 'bg-secondary text-secondary-content border-[#2d2d2d]' : 'bg-white border-[#2d2d2d] text-[#2d2d2d]'}`}
                                style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px' }}
                                onClick={() => setSelectedVecIdx(idx())}
                            >
                                x_{idx()}: {val.toFixed(1)}
                            </button>
                        )}
                    </For>
                </div>

                <Show when={selectedVecIdx() !== null}>
                    <div class="p-3 bg-base-200 border-2 border-[#2d2d2d] shadow-[3px_3px_0px_0px_#2d2d2d]" style={{ 'border-radius': '15px 255px 15px 225px / 255px 15px 225px 15px' }}>
                        <ConfigSlider
                            label={`Edit x₀[${selectedVecIdx()}]`}
                            value={state.x0[selectedVecIdx()!]}
                            min={state.globalMin}
                            max={state.globalMax}
                            step={0.1}
                            onChange={(val) => setInitialStateValue(selectedVecIdx()!, val)}
                            disabled={state.isPlaying}
                            labelInline
                        />
                    </div>
                </Show>
            </div>
        </div>
    )
}
