import type { JSX } from 'solid-js'
import { createSignal, onCleanup, onMount } from 'solid-js'
import { isServer } from 'solid-js/web'

interface ControlSignalBoardProps {
    title: string
    subtitle?: string | JSX.Element | (() => JSX.Element)
    backHref?: string
    backLabel?: string
    top: JSX.Element
    bottom: JSX.Element
    initialTopHeightVh?: number
}

export function ControlSignalBoard(props: ControlSignalBoardProps): JSX.Element {
    const [topHeight, setTopHeight] = createSignal(props.initialTopHeightVh ?? 50)
    let isDragging = false

    const handlePointerDown = (e: PointerEvent) => {
        isDragging = true
        document.body.style.cursor = 'row-resize'
        e.preventDefault()
    }

    const handlePointerMove = (e: PointerEvent) => {
        if (!isDragging) return
        const vh = (e.clientY / window.innerHeight) * 100
        setTopHeight(Math.min(Math.max(vh, 15), 85))
    }

    const handlePointerUp = () => {
        if (isDragging) {
            isDragging = false
            document.body.style.cursor = ''
        }
    }

    onMount(() => {
        if (!isServer) {
            window.addEventListener('pointermove', handlePointerMove)
            window.addEventListener('pointerup', handlePointerUp)
        }
    })

    onCleanup(() => {
        if (!isServer) {
            window.removeEventListener('pointermove', handlePointerMove)
            window.removeEventListener('pointerup', handlePointerUp)
        }
    })

    const renderSubtitle = () => (typeof props.subtitle === 'function' ? props.subtitle() : props.subtitle)

    return (
        <div
            class="min-h-screen flex flex-col font-sans"
            style={{
                'background-color': '#fdfbf7',
                'background-image': 'radial-gradient(#e5e0d8 1px, transparent 1px)',
                'background-size': '24px 24px',
            }}
        >
            <header class="shrink-0 px-4 pt-6 md:px-6 md:pt-8 pb-2">
                <div class="mx-auto max-w-[1600px] flex gap-4 items-start">
                    {props.backHref && (
                        <a
                            href={props.backHref}
                            class="inline-flex items-center justify-center p-2 mt-1 hover:opacity-70 transition-opacity group bg-white border-[3px] border-[#2d2d2d] shadow-[3px_3px_0px_0px_#2d2d2d] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[1px_1px_0px_0px_#2d2d2d]"
                            style={{ color: '#2d2d2d', 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px' }}
                            title={props.backLabel ?? 'Back'}
                        >
                            <svg class="w-6 h-6 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 19l-7-7 7-7" />
                            </svg>
                        </a>
                    )}
                    <div>
                        <h1
                            class="text-3xl md:text-4xl font-bold"
                            style={{ 'font-family': "'Kalam', cursive", color: '#2d2d2d' }}
                        >
                            {props.title}
                        </h1>
                        {props.subtitle !== undefined && props.subtitle !== null && (
                            <p
                                class="text-base md:text-lg mt-1"
                                style={{ 'font-family': "'Patrick Hand', cursive", color: '#2d2d2d', opacity: '0.72' }}
                            >
                                {renderSubtitle()}
                            </p>
                        )}
                    </div>
                </div>
            </header>

            <div class="flex-1 flex flex-col w-full relative mx-auto max-w-[1600px] px-4 md:px-6 pb-6 min-h-0">
                {/* Top Panel (Visualization) */}
                <div
                    class="overflow-auto bg-white border-4 border-[#2d2d2d] shadow-[6px_6px_0px_0px_#2d2d2d] relative transition-none"
                    style={{
                        height: `calc(${topHeight()}vh - 8rem)`,
                        'min-height': '20vh',
                        'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px',
                    }}
                >
                    {props.top}
                </div>

                {/* Draggable Divider */}
                <div
                    class="h-6 mt-4 mb-3 cursor-row-resize flex items-center justify-center flex-shrink-0 z-10 mx-auto w-48 group select-none"
                    onPointerDown={handlePointerDown}
                >
                    <div
                        class="w-full h-4 bg-white border-2 border-[#2d2d2d] shadow-[2px_2px_0px_0px_#2d2d2d] group-hover:bg-[#ff4d4d] transition-colors flex items-center justify-center gap-2"
                        style={{ 'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px' }}
                    >
                        <div class="w-1.5 h-1.5 bg-[#2d2d2d] rounded-full group-hover:bg-white transition-colors" />
                        <div class="w-1.5 h-1.5 bg-[#2d2d2d] rounded-full group-hover:bg-white transition-colors" />
                        <div class="w-1.5 h-1.5 bg-[#2d2d2d] rounded-full group-hover:bg-white transition-colors" />
                    </div>
                </div>

                {/* Bottom Panel (Controls) */}
                <div
                    class="flex-1 min-h-0 overflow-auto bg-white border-4 border-[#2d2d2d] shadow-[4px_4px_0px_0px_#2d2d2d] relative"
                    style={{
                        'border-radius': '15px 255px 15px 225px / 255px 15px 225px 15px',
                    }}
                >
                    {props.bottom}
                </div>
            </div>
        </div>
    )
}
