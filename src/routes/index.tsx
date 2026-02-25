import { createFileRoute } from '@tanstack/solid-router'
import { DemoCard } from '../components/DemoCard'
import twoMoonThumb from '../demos/two-moons-clustering/thumbnail.svg'
import nkClusterThumb from '../demos/nk-clustering/thumbnail.svg'
import mnistThumb from '../demos/mnist-training/thumbnail.svg'
import additionThumb from '../demos/addition-grokking/thumbnail.svg'
import linearSystemsThumb from '../demos/linear-systems/thumbnail.svg'
export const Route = createFileRoute('/')({ component: HomePage })

function HomePage() {
  return (
    <div class="min-h-screen bg-base-200" style={{
      'background-image': 'radial-gradient(var(--color-base-300) 1px, transparent 1px)',
      'background-size': '24px 24px'
    }}>
      {/* Content */}
      <div class="relative font-sans text-base-content">
        {/* Hero Section */}
        <header class="pt-12 pb-8 px-6">
          <div class="max-w-5xl mx-auto">
            {/* Logo and Title - Same Line */}
            {/* Logo and Title - Text Logo */}
            <div class="flex items-center gap-6 mb-6">
              <img
                src="/logo-text.svg"
                alt="tanh.xyz"
                class="h-16 md:h-20 w-auto flex-shrink-0"
              />
            </div>

            <p class="text-lg md:text-xl mb-6 max-w-3xl">
              Interactive demos running in your browser with{' '}
              <a
                href="https://github.com/ekzhang/jax-js"
                target="_blank"
                class="link link-hover"
                style={{ color: "var(--color-base-content)" }}
              >
                jax-js
              </a>
            </p>

            {/* Decorative arrow pointing to demos */}
            <div class="hidden md:block relative h-16">
              <svg
                class="absolute left-32 text-base-content"
                width="100"
                height="60"
                viewBox="0 0 100 60"
                style={{ "transform": "rotate(-5deg)" }}
              >
                <path
                  d="M 50 5 Q 45 25, 50 45 L 50 50 M 50 50 L 45 45 M 50 50 L 55 45"
                  stroke="currentColor"
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
        <main class="mx-auto w-full px-6 pb-20 lg:w-[85%]">
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <DemoCard
              title="Two-Moon Classification"
              description="Train a neural network to classify two interleaving half-moon shapes. Watch the decision boundary evolve in real-time as the network learns."
              href="/demos/two-moon"
              status="ready"
              thumbnail={twoMoonThumb}
            />

            <DemoCard
              title="NK Clustering"
              description="Online clustering algorithm with bounded cluster sizes. Watch points stream in one-by-one and get assigned to clusters in real-time."
              href="/demos/nk-clustering"
              status="ready"
              thumbnail={nkClusterThumb}
            />

            <DemoCard
              title="MNIST with Gradient Norm"
              description="Online MNIST training with seeded reproducibility, live train/test curves, per-digit recall, and top gradient-norm samples."
              href="/demos/mnist-training"
              status="ready"
              thumbnail={mnistThumb}
            />

            <DemoCard
              title="10-Digit Addition Grokking"
              description="Train a compact model on 10-digit addition and watch the grokking-style train/test loss transition across curriculum phases."
              href="/demos/addition-grokking"
              status="ready"
              thumbnail={additionThumb}
            />
            <DemoCard
              title="Linear Systems"
              description="Interactive simulation of a linear dynamical system over time. Observe state evolution and matrix operations."
              href="/demos/linear-systems"
              status="ready"
              thumbnail={linearSystemsThumb}
            />

            <DemoCard
              title="Autoencoder Visualization"
              description="Explore how autoencoders compress and reconstruct data. Visualize the latent space and see how the network learns efficient representations."
              href="/demos/autoencoder"
              status="coming-soon"
            />

            <DemoCard
              title="Gradient Descent Playground"
              description="Visualize different optimization algorithms navigating loss landscapes. Compare SGD, Adam, RMSprop, and more on various terrain types."
              href="/demos/gradient-descent"
              status="coming-soon"
            />
          </div>
        </main>

        {/* Footer */}
        <footer class="border-t-2 border-dashed border-base-content py-8">
          <div class="container mx-auto px-4 text-center">
            <p class="text-sm">
              Built with{' '}
              <a href="https://solidjs.com" target="_blank" class="link link-hover link-secondary">SolidJS</a>
              {' · '}
              <a href="https://tanstack.com/start" target="_blank" class="link link-hover link-secondary">TanStack Start</a>
              {' · '}
              <a href="https://daisyui.com" target="_blank" class="link link-hover link-secondary">DaisyUI</a>
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}
