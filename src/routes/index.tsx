import { createFileRoute } from '@tanstack/solid-router'
import { TwoMoonDemo } from '../components/TwoMoonDemo'

export const Route = createFileRoute('/')({ component: App })

function App() {
  return (
    <div class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Animated background */}
      <div class="fixed inset-0 overflow-hidden pointer-events-none">
        <div class="absolute top-1/4 -left-48 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse"></div>
        <div class="absolute bottom-1/4 -right-48 w-96 h-96 bg-secondary/10 rounded-full blur-3xl animate-pulse" style="animation-delay: 1s"></div>
      </div>

      {/* Content */}
      <div class="relative">
        {/* Hero Section */}
        <header class="pt-16 pb-8 px-6 text-center">
          <div class="max-w-4xl mx-auto">
            <h1 class="text-6xl md:text-7xl font-black mb-6 tracking-tight">
              <span class="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
                tanh.xyz
              </span>
            </h1>
            <p class="text-xl md:text-2xl text-base-content/60 mb-4 font-light">
              Neural Network Playground
            </p>
            <p class="text-base text-base-content/40 max-w-2xl mx-auto">
              Train a neural network in your browser using{' '}
              <a
                href="https://github.com/ekzhang/jax-js"
                target="_blank"
                class="link link-primary link-hover"
              >
                jax-js
              </a>
              {' '}with WebGPU acceleration
            </p>
          </div>
        </header>

        {/* Demo Section */}
        <main class="container mx-auto px-4 pb-20 max-w-6xl">
          <TwoMoonDemo />
        </main>

        {/* Footer */}
        <footer class="border-t border-base-content/5 py-8">
          <div class="container mx-auto px-4 text-center">
            <p class="text-sm text-base-content/30">
              Built with{' '}
              <a href="https://solidjs.com" target="_blank" class="link link-hover">SolidJS</a>
              {' · '}
              <a href="https://tanstack.com/start" target="_blank" class="link link-hover">TanStack Start</a>
              {' · '}
              <a href="https://daisyui.com" target="_blank" class="link link-hover">DaisyUI</a>
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}
