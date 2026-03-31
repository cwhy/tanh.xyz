import { Show } from 'solid-js'
import { getCardVariantForTitle, renderOuterDecorations, renderInnerDecorations } from './CardDecorations'

interface DemoCardProps {
    title: string
    description: string
    href: string
    thumbnail?: string
    status?: 'ready' | 'coming-soon' | 'wip'
}

export function DemoCard(props: DemoCardProps) {
    const isReady = () => props.status !== 'coming-soon'
    const variant = () => getCardVariantForTitle(props.title)

    return (
        <div
            class="relative transition-transform duration-200"
            style={{
                transform: `rotate(${variant().rotation}deg)`,
                flex: '0 0 320px',
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'rotate(0deg) scale(1.02)'
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.transform = `rotate(${variant().rotation}deg)`
            }}
        >
            {renderOuterDecorations(variant())}

            <a
                href={isReady() ? props.href : '#'}
                class={`card bg-base-100 shadow-[4px_4px_0px_0px_#2d2d2d] border-[3px] ${isReady()
                    ? 'border-base-content'
                    : 'border-dashed border-base-content/50'
                    } hover:shadow-[2px_2px_0px_0px_#2d2d2d] hover:translate-x-[2px] hover:translate-y-[2px] transition-all duration-100 group overflow-hidden w-full relative block`}
                style={{
                    'border-radius': variant().borderRadius,
                    opacity: isReady() ? 1 : 0.6,
                    cursor: isReady() ? 'pointer' : 'not-allowed',
                    'aspect-ratio': '1.2 / 1',
                }}
                onClick={(e: MouseEvent) => {
                    if (!isReady()) e.preventDefault()
                }}
            >
                {renderInnerDecorations(variant())}

                {/* Absolute inner wrapper to prevent content from stretch-breaking the aspect ratio */}
                <div class="absolute inset-0 flex flex-col">
                    <Show when={props.thumbnail}>
                        <div class="relative w-full flex-shrink-0" style={{ height: '55%' }}>
                            <img
                                src={props.thumbnail}
                                alt={`${props.title} thumbnail`}
                                class="w-full h-full object-contain p-6 pb-2"
                            />
                        </div>
                    </Show>
                    <div class="card-body p-6 flex-1 overflow-hidden" style={{ "padding-top": props.thumbnail ? '0.5rem' : '1.5rem' }}>
                        <div class="flex items-start gap-4 h-full">
                            <div class="flex-1 h-full flex flex-col">
                                <h3 class="card-title text-base sm:text-lg lg:text-xl font-heading font-bold mb-2 flex items-center gap-2">
                                    {props.title}
                                    <Show when={props.status === 'coming-soon'}>
                                        <span class="badge badge-sm bg-base-300 border-base-content/20 text-[10px] sm:text-xs">
                                            Coming Soon
                                        </span>
                                    </Show>
                                    <Show when={props.status === 'wip'}>
                                        <span class="badge badge-sm bg-warning/20 border-warning text-[10px] sm:text-xs" style={{ color: '#d97706' }}>
                                            WIP
                                        </span>
                                    </Show>
                                </h3>
                                <p class="text-base-content/70 text-xs sm:text-sm leading-relaxed line-clamp-3 overflow-hidden">
                                    {props.description}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </a>
        </div>
    )
}
