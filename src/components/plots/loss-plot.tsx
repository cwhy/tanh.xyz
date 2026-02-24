import { For, Show, createMemo, createSignal, onCleanup, onMount, type JSX } from 'solid-js'

export interface LossPlotPoint {
    x: number
    train: number
    test: number
}

interface LossPlotProps {
    points: LossPlotPoint[]
    xMax: number
    xAxisLabel?: string
    trainLabel?: string
    testLabel?: string
    emptyLabel?: string
}

export function LossPlot(props: LossPlotProps): JSX.Element {
    let svgRef: SVGSVGElement | undefined
    const [size, setSize] = createSignal({ width: 900, height: 340 })

    const width = () => Math.max(320, size().width)
    const height = () => Math.max(220, size().height)
    const paddingLeft = () => Math.max(44, Math.min(72, width() * 0.07))
    const paddingRight = () => Math.max(20, Math.min(40, width() * 0.03))
    const paddingTop = () => Math.max(16, Math.min(28, height() * 0.06))
    const paddingBottom = () => Math.max(42, Math.min(56, height() * 0.14))
    const chartWidth = () => Math.max(1, width() - paddingLeft() - paddingRight())
    const chartHeight = () => Math.max(1, height() - paddingTop() - paddingBottom())

    const updateSize = () => {
        if (!svgRef) return
        const rect = svgRef.getBoundingClientRect()
        if (rect.width > 0 && rect.height > 0) {
            setSize({ width: rect.width, height: rect.height })
        }
    }

    onMount(() => {
        updateSize()
        if (typeof ResizeObserver === 'undefined' || !svgRef) return
        const observer = new ResizeObserver(() => updateSize())
        observer.observe(svgRef)
        onCleanup(() => observer.disconnect())
    })

    const xMax = () => Math.max(1, props.xMax, props.points[props.points.length - 1]?.x ?? 1)
    const yBounds = () => {
        if (props.points.length === 0) return { min: 0, max: 1 }
        const values = props.points.flatMap(point => [point.train, point.test])
        const min = Math.min(...values)
        const max = Math.max(...values)
        const pad = Math.max((max - min) * 0.15, 1e-4)
        return { min: min - pad, max: max + pad }
    }

    const xScale = (x: number) => paddingLeft() + (x / xMax()) * chartWidth()
    const yScale = (y: number) => {
        const bounds = yBounds()
        const ratio = (y - bounds.min) / Math.max(bounds.max - bounds.min, 1e-6)
        return paddingTop() + (1 - ratio) * chartHeight()
    }

    const trainPath = createMemo(() => {
        if (props.points.length === 0) return ''
        return props.points
            .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xScale(point.x)} ${yScale(point.train)}`)
            .join(' ')
    })
    const testPath = createMemo(() => {
        if (props.points.length === 0) return ''
        return props.points
            .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xScale(point.x)} ${yScale(point.test)}`)
            .join(' ')
    })

    const yTicks = createMemo(() => {
        const bounds = yBounds()
        const ticks: Array<{ value: number; y: number }> = []
        for (let i = 0; i < 5; i++) {
            const t = i / 4
            const value = bounds.max - t * (bounds.max - bounds.min)
            ticks.push({ value, y: yScale(value) })
        }
        return ticks
    })

    return (
        <svg ref={svgRef} viewBox={`0 0 ${width()} ${height()}`} class="w-full h-full">
            <rect x={paddingLeft()} y={paddingTop()} width={chartWidth()} height={chartHeight()} fill="rgba(255,255,255,0.6)" rx="10" />

            <For each={yTicks()}>
                {(tick) => (
                    <>
                        <line
                            x1={paddingLeft()}
                            y1={tick.y}
                            x2={paddingLeft() + chartWidth()}
                            y2={tick.y}
                            stroke="rgba(0,0,0,0.12)"
                            stroke-width="1"
                        />
                        <text
                            x={paddingLeft() - 8}
                            y={tick.y + 4}
                            text-anchor="end"
                            class="fill-base-content/60"
                            style={{ 'font-size': '11px' }}
                        >
                            {tick.value.toFixed(3)}
                        </text>
                    </>
                )}
            </For>

            <line x1={paddingLeft()} y1={paddingTop() + chartHeight()} x2={paddingLeft() + chartWidth()} y2={paddingTop() + chartHeight()} stroke="rgba(0,0,0,0.3)" />
            <line x1={paddingLeft()} y1={paddingTop()} x2={paddingLeft()} y2={paddingTop() + chartHeight()} stroke="rgba(0,0,0,0.3)" />

            <Show when={props.points.length > 0}>
                <>
                    <path d={trainPath()} fill="none" stroke="#1d4ed8" stroke-width="2.5" />
                    <path d={testPath()} fill="none" stroke="#d946ef" stroke-width="2.5" />
                </>
            </Show>

            <text x={width() / 2} y={height() - 10} text-anchor="middle" class="fill-base-content/60" style={{ 'font-size': '12px' }}>
                {props.xAxisLabel ?? 'Training step'}
            </text>

            <text x={paddingLeft() + 10} y={paddingTop() + 16} class="fill-blue-700" style={{ 'font-size': '12px', 'font-weight': 600 }}>
                {props.trainLabel ?? 'Train Loss'}
            </text>
            <text x={paddingLeft() + 100} y={paddingTop() + 16} class="fill-fuchsia-600" style={{ 'font-size': '12px', 'font-weight': 600 }}>
                {props.testLabel ?? 'Test Loss'}
            </text>

            <Show when={props.points.length === 0}>
                <text x={width() / 2} y={height() / 2} text-anchor="middle" class="fill-base-content/50" style={{ 'font-size': '14px' }}>
                    {props.emptyLabel ?? 'Start training to render the curve.'}
                </text>
            </Show>
        </svg>
    )
}
