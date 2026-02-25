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
    const Config = props.config
    const Stage = props.stage

    return (
        <div
            class="min-h-screen"
            style={{
                'background-color': '#fdfbf7',
                'background-image': 'radial-gradient(#e5e0d8 1px, transparent 1px)',
                'background-size': '24px 24px',
            }}
        >
            <div class="h-screen w-full pl-4 pr-0 py-6 md:pl-6 md:pr-0 md:py-8">
                <div class="flex h-full min-h-0 flex-col">
                    <div class="grid min-h-0 grow grid-cols-[minmax(280px,360px)_minmax(0,1fr)] gap-6">
                        <aside class="h-full min-h-0 flex flex-col">
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
                                <Dynamic component={Config} />
                            </div>
                        </aside>
                        <section class="h-full min-h-0">
                            <Dynamic component={Stage} />
                        </section>
                    </div>
                </div>
            </div>
        </div>
    )
}
