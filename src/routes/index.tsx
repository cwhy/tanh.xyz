import { createFileRoute } from '@tanstack/solid-router'
import { DemoCard } from '../components/DemoCard'

export const Route = createFileRoute('/')({ component: HomePage })

function HomePage() {
  return (
    <div class="min-h-screen" style={{
      'background-color': '#fdfbf7',
      'background-image': 'radial-gradient(#e5e0d8 1px, transparent 1px)',
      'background-size': '24px 24px'
    }}>
      {/* Content */}
      <div class="relative">
        {/* Hero Section */}
        <header class="pt-12 pb-8 px-6">
          <div class="max-w-5xl mx-auto">
            {/* Logo and Title - Same Line */}
            <div class="flex items-center gap-6 mb-6">
              <img
                src="/logo-sq.svg"
                alt="tanh.xyz"
                class="h-16 md:h-20 w-auto flex-shrink-0"
              />
              <div>
                <h1 class="text-3xl md:text-4xl font-bold" style="font-family: 'Kalam', cursive; color: #2d2d2d;">
                  Neural Network Playground
                </h1>
              </div>
            </div>

            <p class="text-lg md:text-xl mb-6 max-w-3xl" style="font-family: 'Patrick Hand', cursive; color: #2d2d2d;">
              Interactive demos running in your browser with{' '}
              <a
                href="https://github.com/ekzhang/jax-js"
                target="_blank"
                class="link link-hover"
                style="color: #ff4d4d;"
              >
                jax-js
              </a>
              {' '}and WebGPU
            </p>

            {/* Decorative arrow pointing to demos */}
            <div class="hidden md:block relative h-16">
              <svg
                class="absolute left-32"
                width="100"
                height="60"
                viewBox="0 0 100 60"
                style="transform: rotate(-5deg);"
              >
                <path
                  d="M 50 5 Q 45 25, 50 45 L 50 50 M 50 50 L 45 45 M 50 50 L 55 45"
                  stroke="#2d2d2d"
                  stroke-width="2"
                  fill="none"
                  stroke-dasharray="4 4"
                  stroke-linecap="round"
                />
              </svg>
            </div>
          </div>
        </header>

        {/* Demos Grid */}
        <main class="container mx-auto px-6 pb-20 max-w-5xl">
          <h2 class="text-3xl md:text-4xl font-bold mb-8 text-center" style="font-family: 'Kalam', cursive; color: #2d2d2d;">
            Explore Demos
          </h2>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <DemoCard
              title="Two-Moon Classification"
              description="Train a neural network to classify two interleaving half-moon shapes. Watch the decision boundary evolve in real-time as the network learns."
              href="/demos/two-moon"
              status="ready"
              icon={
                <svg class="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              }
            />

            <DemoCard
              title="MNIST Digit Recognition"
              description="Train a convolutional neural network to recognize handwritten digits. Draw your own digits and see the network classify them."
              href="/demos/mnist"
              status="coming-soon"
              icon={
                <svg class="w-6 h-6 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
              }
            />

            <DemoCard
              title="Autoencoder Visualization"
              description="Explore how autoencoders compress and reconstruct data. Visualize the latent space and see how the network learns efficient representations."
              href="/demos/autoencoder"
              status="coming-soon"
              icon={
                <svg class="w-6 h-6 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                </svg>
              }
            />

            <DemoCard
              title="Gradient Descent Playground"
              description="Visualize different optimization algorithms navigating loss landscapes. Compare SGD, Adam, RMSprop, and more on various terrain types."
              href="/demos/gradient-descent"
              status="coming-soon"
              icon={
                <svg class="w-6 h-6 text-info" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              }
            />
          </div>
        </main>

        {/* Footer */}
        <footer class="border-t-2 border-dashed py-8" style="border-color: #2d2d2d;">
          <div class="container mx-auto px-4 text-center">
            <p class="text-sm" style="font-family: 'Patrick Hand', cursive; color: #2d2d2d;">
              Built with{' '}
              <a href="https://solidjs.com" target="_blank" class="link link-hover" style="color: #2d5da1;">SolidJS</a>
              {' · '}
              <a href="https://tanstack.com/start" target="_blank" class="link link-hover" style="color: #2d5da1;">TanStack Start</a>
              {' · '}
              <a href="https://daisyui.com" target="_blank" class="link link-hover" style="color: #2d5da1;">DaisyUI</a>
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}
