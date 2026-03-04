import { createEffect, type JSX } from 'solid-js'
import type { EvennessSnapshot } from './store'

// ---------------------------------------------------------------------------
// Canvas-based line chart for evenness history
// ---------------------------------------------------------------------------

const SERIES = [
    { key: 'p75' as const, label: 'P75', color: '#22d3ee' },
    { key: 'median' as const, label: 'Median', color: '#a78bfa' },
    { key: 'p25' as const, label: 'P25', color: '#fb923c' },
    { key: 'min' as const, label: 'Min', color: '#f87171' },
] as const

const PADDING = { top: 28, right: 20, bottom: 40, left: 50 }

export function EvennessHistoryChart(props: {
    history: EvennessSnapshot[]
    width?: number
    height?: number
}): JSX.Element {
    let canvasRef!: HTMLCanvasElement

    const w = () => props.width ?? 640
    const h = () => props.height ?? 340

    createEffect(() => {
        const data = props.history
        const width = w()
        const height = h()
        drawChart(canvasRef, data, width, height)
    })

    return (
        <div class="flex flex-col items-center gap-3">
            <canvas
                ref={canvasRef}
                style={{ width: `${w()}px`, height: `${h()}px` }}
            />
            {/* Legend */}
            <div class="flex gap-5 text-xs" style={{ 'font-family': "'Inter', sans-serif" }}>
                {SERIES.map(s => (
                    <div class="flex items-center gap-1.5">
                        <div
                            class="w-3 h-0.5 rounded-full"
                            style={{ background: s.color }}
                        />
                        <span style={{ color: s.color, 'font-weight': 600 }}>{s.label}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function drawChart(
    canvas: HTMLCanvasElement,
    data: EvennessSnapshot[],
    width: number,
    height: number
): void {
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // Clear
    ctx.clearRect(0, 0, width, height)

    if (data.length === 0) {
        ctx.fillStyle = '#888'
        ctx.font = '14px Inter, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('No data yet', width / 2, height / 2)
        return
    }

    const plotW = width - PADDING.left - PADDING.right
    const plotH = height - PADDING.top - PADDING.bottom

    // Axis ranges
    const xMin = data[0].step
    const xMax = data[data.length - 1].step
    const xRange = Math.max(1, xMax - xMin)

    // Y always 0–1 for evenness
    const yMin = 0
    const yMax = 1

    const toX = (step: number) => PADDING.left + ((step - xMin) / xRange) * plotW
    const toY = (val: number) => PADDING.top + (1 - (val - yMin) / (yMax - yMin)) * plotH

    // Grid lines
    ctx.strokeStyle = 'rgba(128,128,128,0.15)'
    ctx.lineWidth = 1
    for (let y = 0; y <= 1; y += 0.25) {
        const py = toY(y)
        ctx.beginPath()
        ctx.moveTo(PADDING.left, py)
        ctx.lineTo(width - PADDING.right, py)
        ctx.stroke()
    }

    // Axes
    ctx.strokeStyle = 'rgba(128,128,128,0.4)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(PADDING.left, PADDING.top)
    ctx.lineTo(PADDING.left, height - PADDING.bottom)
    ctx.lineTo(width - PADDING.right, height - PADDING.bottom)
    ctx.stroke()

    // Y tick labels
    ctx.fillStyle = '#999'
    ctx.font = '11px Inter, monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let y = 0; y <= 1; y += 0.25) {
        ctx.fillText(y.toFixed(2), PADDING.left - 6, toY(y))
    }

    // X tick labels
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const desiredTicks = Math.min(8, data.length)
    const stepInterval = Math.max(1, Math.floor(data.length / desiredTicks))
    for (let i = 0; i < data.length; i += stepInterval) {
        const step = data[i].step
        ctx.fillText(String(step), toX(step), height - PADDING.bottom + 6)
    }
    // Always show last tick
    if (data.length > 1) {
        const lastStep = data[data.length - 1].step
        ctx.fillText(String(lastStep), toX(lastStep), height - PADDING.bottom + 6)
    }

    // Axis labels
    ctx.fillStyle = '#777'
    ctx.font = '12px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText('Step', width / 2, height - 8)

    ctx.save()
    ctx.translate(12, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText('Evenness', 0, 0)
    ctx.restore()

    // Title
    ctx.fillStyle = '#aaa'
    ctx.font = 'bold 13px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText('Evenness Distribution Over Time', width / 2, 6)

    // Draw P25–P75 filled band
    if (data.length > 1) {
        ctx.fillStyle = 'rgba(167, 139, 250, 0.10)'
        ctx.beginPath()
        ctx.moveTo(toX(data[0].step), toY(data[0].p75))
        for (let i = 1; i < data.length; i++) {
            ctx.lineTo(toX(data[i].step), toY(data[i].p75))
        }
        for (let i = data.length - 1; i >= 0; i--) {
            ctx.lineTo(toX(data[i].step), toY(data[i].p25))
        }
        ctx.closePath()
        ctx.fill()
    }

    // Draw line series
    for (const series of SERIES) {
        ctx.strokeStyle = series.color
        ctx.lineWidth = series.key === 'median' ? 2.5 : 1.5
        ctx.lineJoin = 'round'
        ctx.beginPath()
        for (let i = 0; i < data.length; i++) {
            const x = toX(data[i].step)
            const y = toY(data[i][series.key])
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
        }
        ctx.stroke()
    }
}
