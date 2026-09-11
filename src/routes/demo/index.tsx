import { createFileRoute } from '@tanstack/solid-router'
import { DemoCard } from '../../components/DemoCard'
import { SeriesCard } from '../../components/SeriesCard'
import twoMoonThumb from '../../demos/two-moons-clustering/thumbnail.svg'
import nkClusterThumb from '../../demos/nk-clustering/thumbnail.svg'
import mnistThumb from '../../demos/mnist-training/thumbnail.svg'
import sheafAdmmMnistThumb from '../../demos/sheaf-admm-mnist/thumbnail.svg'
import additionThumb from '../../demos/addition-grokking/thumbnail.svg'
import linearSystemsThumb from '../../demos/linear-systems/thumbnail.svg'
import damThumb from '../../demos/dense-associative-memory/thumbnail.svg'
import needleThumb from '../../demos/needle/thumbnail.svg'

export const Route = createFileRoute('/demo/')({
  component: HomePage,
  head: () => ({ meta: [{ title: 'Neural demos — tanh.xyz' }] }),
})

function HomePage() {
  return (
    <div class="min-h-screen bg-base-200 flex flex-col" style={{
      'background-image': 'radial-gradient(var(--color-base-300) 1px, transparent 1px)',
      'background-size': '24px 24px'
    }}>
      {/* Content */}
      <div class="relative font-sans text-base-content flex flex-col flex-1">
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
        <main class="mx-auto w-full px-6 pb-20 max-w-[1600px] 2xl:max-w-[1800px]">
          <div class="flex flex-wrap justify-center gap-6 md:gap-8 xl:gap-10">
            <DemoCard
              title="Two-Moon Classification"
              description="Train a neural network to classify two interleaving half-moon shapes. Watch the decision boundary evolve in real-time as the network learns."
              href="/demo/two-moon"
              status="ready"
              thumbnail={twoMoonThumb}
            />

            <SeriesCard
              seriesName="Online Clustering"
              description="Online clustering algorithms with bounded cluster sizes. Explore different data domains and visualizations."
              thumbnail={nkClusterThumb}
              demos={[
                {
                  title: "2D Online K-Means",
                  description: "Streaming k-means — each incoming point joins its nearest centroid, updating the mean incrementally.",
                  href: "/demo/online-k-means",
                  thumbnail: nkClusterThumb,
                },
                {
                  title: "2D Point NK Clustering",
                  description: "Watch points stream in one-by-one and get assigned to clusters in real-time with a 2D scatter visualization.",
                  href: "/demo/nk-clustering",
                  thumbnail: nkClusterThumb,
                },
                {
                  title: "MNIST Digit NK Clustering",
                  description: "NK clustering on MNIST digits with JAX-JS accelerated distance computation. Clusters shown as image grids sorted by tightness.",
                  href: "/demo/mnist-nk-clustering",
                  thumbnail: nkClusterThumb,
                },
                {
                  title: "2D Rival-K Clustering",
                  description: "K-only online clustering — clusters self-regulate via evenness-based eviction with no hard size cap.",
                  href: "/demo/rival-k",
                  thumbnail: nkClusterThumb,
                },
              ]}
            />

            <DemoCard
              title="MNIST with Gradient Norm"
              description="Online MNIST training with seeded reproducibility, live train/test curves, per-digit recall, and top gradient-norm samples."
              href="/demo/mnist-training"
              status="ready"
              thumbnail={mnistThumb}
            />

            <DemoCard
              title="Sheaf-ADMM MNIST"
              description="Limited-view MNIST patch agents negotiate a global digit class through ADMM-style sheaf consensus."
              href="/demo/sheaf-admm-mnist"
              status="ready"
              thumbnail={sheafAdmmMnistThumb}
            />

            <DemoCard
              title="10-Digit Addition Grokking"
              description="Train a compact model on 10-digit addition and watch the grokking-style train/test loss transition across curriculum phases."
              href="/demo/addition-grokking"
              status="ready"
              thumbnail={additionThumb}
            />
            <DemoCard
              title="Needle SAN"
              description="Run a 26M-scale Simple Attention Network forward pass in jax-js with GQA, RoPE, gated residuals, and tied embeddings."
              href="/demo/needle"
              status="ready"
              thumbnail={needleThumb}
            />
            <DemoCard
              title="Linear Systems"
              description="Interactive simulation of a linear dynamical system over time. Observe state evolution and matrix operations."
              href="/demo/linear-systems"
              status="ready"
              thumbnail={linearSystemsThumb}
            />
            <DemoCard
              title="Dense Associative Memory"
              description="Explore biologically plausible memory networks with threshold nonlinearity. Watch how noisy images are recalled step by step through attractor dynamics."
              href="/demo/dense-associative-memory"
              status="wip"
              thumbnail={damThumb}
            />

            <DemoCard
              title="Autoencoder Visualization"
              description="Explore how autoencoders compress and reconstruct data. Visualize the latent space and see how the network learns efficient representations."
              href="/demo/autoencoder"
              status="coming-soon"
            />

            <DemoCard
              title="Gradient Descent Playground"
              description="Visualize different optimization algorithms navigating loss landscapes. Compare SGD, Adam, RMSprop, and more on various terrain types."
              href="/demo/gradient-descent"
              status="coming-soon"
            />
          </div>
        </main>

        {/* Footer */}
        <footer class="border-t-2 border-dashed border-base-content py-8 mt-auto">
          <div class="container mx-auto px-4 text-center">
            <p class="text-sm">
              Built with{' '}
              <a href="https://solidjs.com" target="_blank" class="link link-hover link-secondary">SolidJS</a>
              {' · '}
              <a href="https://tanstack.com/start" target="_blank" class="link link-hover link-secondary">TanStack Start</a>
              {' · '}
              <a href="https://daisyui.com" target="_blank" class="link link-hover link-secondary">DaisyUI</a>
              {' · '}
              <a href="/research-notes/" class="link link-hover link-secondary">Research Notes</a>
              {' · '}
              <a href="/paper-notes/" class="link link-hover link-secondary">Paper Notes</a>
              {' · '}
              <a href="/" class="link link-hover link-secondary">Site Index</a>
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}
