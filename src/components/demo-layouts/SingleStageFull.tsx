import type { Component, JSX } from 'solid-js'
import { Dynamic } from 'solid-js/web'

interface SingleStageFullLayoutProps {
    title: string
    subtitle?: string
    backHref?: string
    backLabel?: string
    config: Component
    stage: Component
}

export function SingleStageFullLayout(props: SingleStageFullLayoutProps): JSX.Element {
    return (
        <div
            class="min-h-screen"
            style={{
                'background-color': '#fdfbf7',
                'background-image': 'radial-gradient(#e5e0d8 1px, transparent 1px)',
                'background-size': '24px 24px',
            }}
        >
            <div class="min-h-screen w-full px-4 py-6 md:px-6 md:py-8 lg:h-screen lg:pr-0">
                <div class="flex h-full min-h-0 flex-col">
                    <div class="grid min-h-0 grow grid-cols-1 gap-6 lg:grid-cols-[minmax(280px,360px)_minmax(0,1fr)]">
                        <aside class="min-h-0 flex flex-col lg:h-full">
                            <header class="shrink-0 pb-4">
                                {props.backHref && (
                                    <a
                                        href={props.backHref}
                                        class="inline-flex items-center gap-3 mb-4 hover:opacity-70 transition-opacity group"
                                        style={{ color: '#2d2d2d', 'font-family': "'Patrick Hand', cursive" }}
                                    >
                                        <svg class="w-5 h-5 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                                        </svg>
                                        {props.backLabel ?? 'Back'}
                                    </a>
                                )}
                                <h1
                                    class="text-3xl md:text-4xl font-bold"
                                    style={{ 'font-family': "'Kalam', cursive", color: '#2d2d2d' }}
                                >
                                    {props.title}
                                </h1>
                                {props.subtitle && (
                                    <p
                                        class="text-base md:text-lg"
                                        style={{ 'font-family': "'Patrick Hand', cursive", color: '#2d2d2d', opacity: '0.72' }}
                                    >
                                        {props.subtitle}
                                    </p>
                                )}
                            </header>
                            <div class="min-h-0 grow">
                                <Dynamic component={props.config} />
                            </div>
                        </aside>
                        <section class="min-h-0 lg:h-full">
                            <Dynamic component={props.stage} />
                        </section>
                    </div>
                </div>
            </div>
        </div>
    )
}
