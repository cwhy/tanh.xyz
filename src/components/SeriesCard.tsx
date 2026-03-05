import { createSignal, For, Show } from 'solid-js'
import { Portal } from 'solid-js/web'
import { getCardVariantForTitle, getStackVariantForTitle, renderOuterDecorations, renderInnerDecorations } from './CardDecorations'

interface SeriesDemo {
    title: string
    description: string
    href: string
    thumbnail?: string
}

interface SeriesCardProps {
    seriesName: string
    description: string
    thumbnail?: string
    demos: SeriesDemo[]
}

export function SeriesCard(props: SeriesCardProps) {
    const [open, setOpen] = createSignal(false)
    const [isHovered, setIsHovered] = createSignal(false)
    const variant = () => getCardVariantForTitle(props.seriesName)

    // Generate distinct styles for the background stack items
    const stack1 = () => getStackVariantForTitle(props.seriesName, 1)
    const stack2 = () => getStackVariantForTitle(props.seriesName, 2)

    const cardBaseClass =
        'card bg-base-100 shadow-[4px_4px_0px_0px_#2d2d2d] border-[3px] border-base-content transition-all duration-200'

    return (
        <>
            {/* Stacked card trigger */}
            <div
                class="relative transition-transform duration-200"
                style={{
                    'padding-top': '8px',
                    'padding-left': '8px',
                    transform: `rotate(${variant().rotation}deg)`,
                    flex: '0 0 320px',
                }}
                onMouseEnter={(e) => {
                    setIsHovered(true)
                    e.currentTarget.style.transform = 'rotate(0deg) scale(1.02)'
                }}
                onMouseLeave={(e) => {
                    setIsHovered(false)
                    e.currentTarget.style.transform = `rotate(${variant().rotation}deg)`
                }}
            >
                {renderOuterDecorations(variant())}

                {/* Back cards (stack effect) using their own specific variations based on the hash */}
                <div
                    class={`${cardBaseClass} absolute cursor-pointer`}
                    style={{
                        'border-radius': stack2().borderRadius,
                        transform: isHovered()
                            ? `translate(-26px, -20px) rotate(${stack2().rotation - 6}deg) scale(0.98)`
                            : `translate(0px, 0px) rotate(${stack2().rotation}deg) scale(0.98)`,
                        top: '-2px',
                        left: '-2px',
                        right: '10px',
                        bottom: '10px',
                        opacity: 0.5,
                    }}
                />
                <div
                    class={`${cardBaseClass} absolute cursor-pointer`}
                    style={{
                        'border-radius': stack1().borderRadius,
                        transform: isHovered()
                            ? `translate(26px, -10px) rotate(${stack1().rotation + 6}deg) scale(0.99)`
                            : `translate(0px, 0px) rotate(${stack1().rotation}deg) scale(0.99)`,
                        top: '2px',
                        left: '2px',
                        right: '6px',
                        bottom: '6px',
                        opacity: 0.7,
                    }}
                />

                {/* Front card */}
                <div
                    class={`${cardBaseClass} relative hover:shadow-[2px_2px_0px_0px_#2d2d2d] hover:translate-x-[2px] hover:translate-y-[2px] group overflow-hidden cursor-pointer bg-base-100 w-full block`}
                    style={{
                        'border-radius': variant().borderRadius,
                        'aspect-ratio': '1.2 / 1'
                    }}
                    onClick={() => setOpen(true)}
                >
                    {renderInnerDecorations(variant())}

                    {/* Absolute inner wrapper to prevent content from stretch-breaking the aspect ratio */}
                    <div class="absolute inset-0 flex flex-col">
                        <Show when={props.thumbnail}>
                            <div class="relative w-full flex-shrink-0" style={{ height: '55%' }}>
                                <img
                                    src={props.thumbnail}
                                    alt={`${props.seriesName} thumbnail`}
                                    class="w-full h-full object-contain p-6 pb-2"
                                />
                            </div>
                        </Show>
                        <div class="card-body p-6 flex-1 overflow-hidden" style={{ "padding-top": props.thumbnail ? '0.5rem' : '1.5rem' }}>
                            <div class="flex items-start gap-4 h-full">
                                <div class="flex-1 h-full flex flex-col">
                                    <h3 class="card-title text-base sm:text-lg lg:text-xl font-heading font-bold mb-2 flex items-center gap-2">
                                        {props.seriesName}
                                        <span class="badge badge-sm bg-base-300 border-base-content/20 text-[10px] sm:text-xs">
                                            {props.demos.length} demos
                                        </span>
                                    </h3>
                                    <p class="text-base-content/70 text-xs sm:text-sm leading-relaxed line-clamp-3 overflow-hidden">
                                        {props.description}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Modal */}
            <Show when={open()}>
                <Portal>
                    <div
                        class="fixed inset-0 z-50 flex items-center justify-center p-4"
                        onClick={(e) => {
                            if (e.target === e.currentTarget) setOpen(false)
                        }}
                    >
                        {/* Backdrop */}
                        <div class="absolute inset-0 bg-base-content/40 backdrop-blur-sm" />

                        {/* Modal content */}
                        <div
                            class="relative bg-base-200 border-[3px] border-base-content shadow-[6px_6px_0px_0px_#2d2d2d] w-full max-w-2xl max-h-[80vh] overflow-y-auto p-6"
                            style={{
                                'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px',
                                'background-image': 'radial-gradient(var(--color-base-300) 1px, transparent 1px)',
                                'background-size': '20px 20px',
                            }}
                        >
                            {/* Header */}
                            <div class="flex items-center justify-between mb-6">
                                <h2 class="text-2xl font-heading font-bold">
                                    {props.seriesName}
                                </h2>
                                <button
                                    class="btn btn-sm btn-ghost border-2 border-base-content/20"
                                    onClick={() => setOpen(false)}
                                >
                                    ✕
                                </button>
                            </div>
                            <p class="text-base-content/70 text-sm mb-6">
                                {props.description}
                            </p>

                            {/* Demo cards */}
                            <div class="flex flex-col gap-4">
                                <For each={props.demos}>
                                    {(demo) => {
                                        const demoVariant = () => getCardVariantForTitle(demo.title)
                                        return (
                                            <a
                                                href={demo.href}
                                                class={`${cardBaseClass} hover:shadow-[2px_2px_0px_0px_#2d2d2d] hover:translate-x-[2px] hover:translate-y-[2px] group flex flex-row overflow-hidden`}
                                                style={{ 'border-radius': demoVariant().borderRadius }}
                                            >
                                                <Show when={demo.thumbnail}>
                                                    <div class="w-32 flex-shrink-0 overflow-hidden flex items-center justify-center">
                                                        <img
                                                            src={demo.thumbnail}
                                                            alt={`${demo.title} thumbnail`}
                                                            class="w-full h-full object-contain p-4"
                                                        />
                                                    </div>
                                                </Show>
                                                <div class="card-body p-5 flex-1">
                                                    <h4 class="card-title text-base font-heading font-bold mb-1">
                                                        {demo.title}
                                                    </h4>
                                                    <p class="text-base-content/70 text-sm leading-relaxed">
                                                        {demo.description}
                                                    </p>
                                                </div>
                                            </a>
                                        )
                                    }}
                                </For>
                            </div>
                        </div>
                    </div>
                </Portal>
            </Show>
        </>
    )
}
