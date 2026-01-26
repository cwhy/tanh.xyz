import { createFileRoute } from '@tanstack/solid-router'
import { TwoMoonDemo } from '../../components/TwoMoonDemo'

export const Route = createFileRoute('/demos/two-moon')({ component: TwoMoonPage })

function TwoMoonPage() {
    return (
        <div class="min-h-screen" style={{
            'background-color': '#fdfbf7',
            'background-image': 'radial-gradient(#e5e0d8 1px, transparent 1px)',
            'background-size': '24px 24px'
        }}>
            {/* Content */}
            <div class="relative">
                {/* Header */}
                <header class="pt-8 pb-4 px-6">
                    <div class="max-w-6xl mx-auto">
                        <a href="/" class="inline-flex items-center gap-3 mb-4 hover:opacity-70 transition-opacity group" style="color: #2d2d2d; font-family: 'Patrick Hand', cursive;">
                            <svg class="w-5 h-5 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                            </svg>
                            Back to Demos
                        </a>
                        <h1 class="text-4xl md:text-5xl font-bold mb-2" style="font-family: 'Kalam', cursive; color: #2d2d2d;">
                            Two-Moon Classification
                        </h1>
                        <p class="text-lg" style="font-family: 'Patrick Hand', cursive; color: #2d2d2d; opacity: 0.7;">
                            Train a neural network to classify two interleaving half-moon shapes
                        </p>
                    </div>
                </header>

                {/* Demo Section */}
                <main class="container mx-auto px-4 pb-20 max-w-6xl">
                    <TwoMoonDemo />
                </main>
            </div>
        </div>
    )
}
