import { For, createEffect, createMemo } from 'solid-js'
import { simulationStore } from './store'

export function VisualizationPanel() {
    const { state } = simulationStore
    let containerRef: HTMLDivElement | undefined

    createEffect(() => {
        if (state.history.length > 0 && containerRef) {
            containerRef.scrollLeft = containerRef.scrollWidth
        }
    })

    const dataBounds = createMemo(() => {
        let min = Infinity
        let max = -Infinity
        for (const step of state.history) {
            for (const val of step) {
                if (val < min) min = val
                if (val > max) max = val
            }
        }
        if (min === Infinity || min === max) {
            min = state.globalMin
            max = state.globalMax
        }
        return { min, max }
    })

    const getColor = (val: number) => {
        const { min, max } = dataBounds()
        const range = max - min
        const norm = range === 0 ? 0.5 : (val - min) / range // 0 to 1

        if (norm < 0.5) {
            // norm from 0 to 0.5 maps to lightness 55% to 100%
            const lightness = 55 + (norm / 0.5) * 45
            return `hsl(210, 80%, ${lightness}%)`
        } else {
            // norm from 0.5 to 1.0 maps to lightness 100% to 55%
            const lightness = 100 - ((norm - 0.5) / 0.5) * 45
            return `hsl(0, 80%, ${lightness}%)`
        }
    }

    return (
        <div class="h-full flex flex-col p-4 md:p-6" style={{ 'font-family': "'Patrick Hand', cursive", color: '#2d2d2d' }}>
            <h2 class="text-2xl font-bold mb-4 shrink-0" style={{ 'font-family': "'Kalam', cursive" }}>System State Evolution</h2>

            <div class="flex-1 min-h-0 relative bg-[#fdfbf7] border-[3px] border-[#2d2d2d] shadow-inner rounded-xl overflow-hidden shadow-[inset_2px_2px_6px_rgba(0,0,0,0.1)]" style={{ 'border-radius': '15px 255px 15px 225px / 255px 15px 225px 15px' }}>
                <div
                    ref={containerRef}
                    class="absolute inset-0 overflow-auto p-4 flex flex-col scroll-smooth"
                >
                    <div class="inline-flex flex-col gap-1 min-w-max h-full">
                        <For each={Array(state.N).fill(0)}>
                            {(_, rowIdx) => (
                                <div class="flex gap-1 flex-1 min-h-0">
                                    <div class="w-8 shrink-0 flex items-center justify-end pr-2 text-xs md:text-sm font-mono font-bold">
                                        x_{rowIdx()}
                                    </div>
                                    <For each={state.history}>
                                        {(colData, colIdx) => {
                                            const val = colData[rowIdx()]
                                            return (
                                                <div
                                                    class="h-full aspect-square shrink-0 border border-black/10 transition-colors"
                                                    style={{
                                                        'background-color': getColor(val),
                                                        'border-radius': '3px 4px 2px 3px' // slight wobble feeling for tiny rects is tough, but slight varied radius helps
                                                    }}
                                                    title={`Step ${colIdx()}, Node ${rowIdx()}: ${val.toFixed(2)}`}
                                                />
                                            )
                                        }}
                                    </For>
                                </div>
                            )}
                        </For>
                    </div>
                </div>
            </div>

            <div class="shrink-0 mt-4 flex items-center justify-between font-bold text-sm md:text-base">
                <span>Time (Steps) →</span>
                <div class="flex gap-2 items-center">
                    <span class="font-mono">{dataBounds().min.toFixed(2)}</span>
                    <div class="w-32 h-4 bg-gradient-to-r from-[#59a8ff] via-white to-[#ff5959] border-[3px] border-[#2d2d2d] shadow-[2px_2px_0px_0px_#2d2d2d]" style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px' }} />
                    <span class="font-mono">{dataBounds().max.toFixed(2)}</span>
                </div>
            </div>
        </div>
    )
}
