import { For, Show } from 'solid-js'
import { createFileRoute } from '@tanstack/solid-router'
import twoMoonThumb from '../demos/two-moons-clustering/thumbnail.svg'
import nkClusterThumb from '../demos/nk-clustering/thumbnail.svg'
import mnistThumb from '../demos/mnist-training/thumbnail.svg'
import sheafAdmmMnistThumb from '../demos/sheaf-admm-mnist/thumbnail.svg'
import additionThumb from '../demos/addition-grokking/thumbnail.svg'
import linearSystemsThumb from '../demos/linear-systems/thumbnail.svg'
import damThumb from '../demos/dense-associative-memory/thumbnail.svg'
import needleThumb from '../demos/needle/thumbnail.svg'

type DemoDirectoryItem = {
  title: string
  thumbnail?: string
  href?: string
  status?: 'coming soon'
}

const demoColumns: DemoDirectoryItem[][] = [
  [
    { title: 'Two-Moon Classification', thumbnail: twoMoonThumb, href: '/demo/two-moon' },
    { title: '2D Online K-Means', thumbnail: nkClusterThumb, href: '/demo/online-k-means' },
    { title: '2D Point NK Clustering', thumbnail: nkClusterThumb, href: '/demo/nk-clustering' },
    { title: 'MNIST Digit NK Clustering', thumbnail: nkClusterThumb, href: '/demo/mnist-nk-clustering' },
    { title: '2D Rival-K Clustering', thumbnail: nkClusterThumb, href: '/demo/rival-k' },
    { title: 'MNIST with Gradient Norm', thumbnail: mnistThumb, href: '/demo/mnist-training' },
    { title: 'Sheaf-ADMM MNIST', thumbnail: sheafAdmmMnistThumb, href: '/demo/sheaf-admm-mnist' },
  ],
  [
    { title: '10-Digit Addition Grokking', thumbnail: additionThumb, href: '/demo/addition-grokking' },
    { title: 'Needle SAN', thumbnail: needleThumb, href: '/demo/needle' },
    { title: 'Linear Systems', thumbnail: linearSystemsThumb, href: '/demo/linear-systems' },
    { title: 'Dense Associative Memory', thumbnail: damThumb, href: '/demo/dense-associative-memory' },
    { title: 'Autoencoder Visualization', status: 'coming soon' },
    { title: 'Gradient Descent Playground', status: 'coming soon' },
  ],
]

export const Route = createFileRoute('/')({
  component: SiteIndex,
  head: () => ({
    meta: [
      { title: 'tanh.xyz — Experiments and notes' },
      {
        name: 'description',
        content: 'Interactive neural demos, research notes, and critical readings.',
      },
    ],
  }),
})

function ArrowIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 28 18" class="h-[18px] w-7 flex-none" fill="none">
      <path d="M1 9h24M18 2l7 7-7 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  )
}

function FooterWordmark() {
  return (
    <span aria-hidden="true" class="footer-wordmark">
      <span class="footer-wordmark__tanh">tanh</span>
      <span class="footer-wordmark__dot">•</span>
      <span class="footer-wordmark__xyz">xyz</span>
    </span>
  )
}

function TextLink(props: { href: string; children: string }) {
  return (
    <a
      href={props.href}
      class="group inline-flex items-center gap-3 border-b-2 border-[#2d5da1] pb-1 font-sans text-xl text-[#2d5da1] transition-colors hover:border-[#d93636] hover:text-[#d93636] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2d5da1]"
    >
      <span>{props.children}</span>
      <span class="transition-transform group-hover:translate-x-1"><ArrowIcon /></span>
    </a>
  )
}

function DemoRow(props: { item: DemoDirectoryItem }) {
  const content = (
    <>
      <Show
        when={props.item.thumbnail}
        fallback={(
          <span
            aria-hidden="true"
            class="flex aspect-[4/3] w-20 flex-none items-center justify-center border border-dashed border-[#8b8781] bg-[#fdfbf7]/70 font-mono text-[.65rem] uppercase tracking-[.16em] text-[#8b8781] md:w-28"
          >
            soon
          </span>
        )}
      >
        <span class="aspect-[4/3] w-20 flex-none overflow-hidden border border-[#2d2d2d] bg-[#fdfbf7] md:w-28">
          <img src={props.item.thumbnail!} alt="" class="h-full w-full object-cover" />
        </span>
      </Show>
      <span class="min-w-0 flex-1 font-sans text-xl leading-tight md:text-[1.65rem]">
        {props.item.title}
        {props.item.status && (
          <span class="ml-2 text-base text-[#69645f] md:text-xl">— {props.item.status}</span>
        )}
      </span>
      {props.item.href && <ArrowIcon />}
    </>
  )

  return (
    <Show
      when={props.item.href}
      fallback={(
        <div class="flex min-h-24 items-center gap-3 border-b border-[#2d2d2d] px-4 py-5 text-[#8b8781] md:px-6">
          {content}
        </div>
      )}
    >
      <a
        href={props.item.href!}
        class="group flex min-h-24 items-center gap-3 border-b border-[#2d2d2d] px-4 py-5 text-[#2d5da1] transition-colors hover:bg-[#fff2a8]/55 hover:text-[#d93636] focus-visible:relative focus-visible:z-10 focus-visible:outline-2 focus-visible:outline-[#2d5da1] md:px-6"
      >
        {content}
      </a>
    </Show>
  )
}

function SiteIndex() {
  return (
    <div
      class="relative min-h-screen overflow-hidden bg-[#fdfbf7] text-[#2d2d2d]"
      style={{
        'background-image':
          'radial-gradient(circle at 1px 1px, rgba(45,45,45,.1) 1px, transparent 1.1px), linear-gradient(rgba(45,93,161,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(45,93,161,.025) 1px, transparent 1px)',
        'background-size': '24px 24px, 96px 96px, 96px 96px',
      }}
    >
      <div aria-hidden="true" class="pointer-events-none absolute inset-y-0 left-6 z-20 border-l-2 border-[#d93636]/35 md:left-[clamp(24px,4vw,60px)]" />

      <div class="relative z-10 border-b-2 border-[#2d2d2d]">
        <header class="notebook-shell notebook-shell--wide flex min-h-24 items-center justify-between gap-6 md:min-h-28">
          <a href="/" aria-label="tanh.xyz home" class="focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2d5da1]">
            <img src="/logo-text.svg" alt="tanh.xyz" class="h-11 w-auto md:h-14" />
          </a>

          <nav aria-label="Content sections" class="hidden items-center gap-8 font-sans text-xl md:flex lg:gap-12">
            <a href="#demos" class="border-b-2 border-[#2d5da1] text-[#2d2d2d] hover:text-[#2d5da1]">Demos</a>
            <a href="#research-notes" class="border-b-2 border-[#2d5da1] text-[#2d2d2d] hover:text-[#2d5da1]">Research notes</a>
            <a href="#paper-notes" class="border-b-2 border-[#2d5da1] text-[#2d2d2d] hover:text-[#2d5da1]">Paper notes</a>
          </nav>
        </header>
      </div>

      <main class="relative z-10">
        <section
          id="demos"
          class="notebook-bg-dots scroll-mt-6 border-b-2 border-[#2d2d2d] py-14 md:py-20"
        >
          <div class="notebook-shell notebook-shell--demos">
            <div class="mb-10 flex flex-col items-start justify-between gap-6 md:flex-row md:items-end">
              <h1 class="border-b-[5px] border-[#f0be24] pb-1 font-heading text-5xl font-bold leading-none">Demos</h1>
              <TextLink href="/demo">Open demo gallery</TextLink>
            </div>

            <div class="grid border-t-2 border-[#2d2d2d] lg:grid-cols-2">
              <For each={demoColumns}>
                {(column, columnIndex) => (
                  <div class={columnIndex() === 1 ? 'lg:border-l-2 lg:border-[#2d2d2d]' : ''}>
                    <For each={column}>{(item) => <DemoRow item={item} />}</For>
                  </div>
                )}
              </For>
            </div>
          </div>
        </section>

        <section id="research-notes" class="notebook-bg-grid scroll-mt-6 border-b-2 border-[#2d2d2d] py-16 md:py-24">
          <div class="notebook-shell grid min-h-[320px] gap-12 md:grid-cols-[minmax(260px,.7fr)_minmax(0,1.3fr)] md:items-center md:gap-20">
            <div>
              <h2 class="inline-block border-b-[5px] border-[#f0be24] pb-1 font-heading text-5xl font-bold">Research notes</h2>
              <p class="mt-6 max-w-sm font-sans text-xl leading-snug md:text-2xl">Technical essays, working notes, and printable papers.</p>
              <div class="mt-7"><TextLink href="/research-notes/">Open research notes</TextLink></div>
            </div>

            <p class="justify-self-start font-heading text-3xl leading-tight md:justify-self-center md:text-4xl">
              <span aria-hidden="true" class="mr-4 text-[#f0be24]">›</span>
              <a href="/research-notes/unified-rl.html" class="text-[#2d5da1] hover:underline">Unified RL: one connected learning loop</a>
              <span aria-hidden="true" class="ml-4 text-[#f0be24]">‹</span>
            </p>
          </div>
        </section>

        <section id="paper-notes" class="notebook-bg-grid scroll-mt-6 border-b-2 border-[#2d2d2d] py-16 md:py-24">
          <div class="notebook-shell grid min-h-[360px] gap-12 md:grid-cols-[minmax(280px,.7fr)_minmax(0,1.6fr)] md:items-center md:gap-16">
            <div>
              <h2 class="inline-block border-b-[5px] border-[#f0be24] pb-1 font-heading text-5xl font-bold">Paper notes</h2>
              <p class="mt-6 max-w-sm font-sans text-xl leading-snug md:text-2xl">Introductions, diagrams, and critical readings of machine-learning papers.</p>
              <div class="mt-7"><TextLink href="/paper-notes/">Read paper notes</TextLink></div>
            </div>

            <a
              href="/paper-notes/tabicl-architecture.html"
              class="group grid border-y-2 border-[#2d2d2d] py-8 text-[#2d2d2d] transition-colors hover:bg-[#fff2a8]/55 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2d5da1] sm:grid-cols-[110px_1fr_auto] sm:items-center sm:gap-6 sm:px-6 md:py-10"
            >
              <span class="font-heading text-6xl font-bold text-[#d93636] sm:text-7xl">01</span>
              <span class="mt-4 min-w-0 sm:mt-0">
                <span class="block font-heading text-2xl font-bold leading-tight md:text-4xl">TabICL: teaching a table to become a prompt</span>
                <span class="mt-3 block font-mono text-sm text-[#69645f] md:text-base">Qu et al. · ICML 2025 · HTML</span>
              </span>
              <span class="mt-6 text-[#2d5da1] transition-transform group-hover:translate-x-1 sm:mt-0"><ArrowIcon /></span>
            </a>
          </div>
        </section>
      </main>

      <footer id="site-footer" class="notebook-shell relative z-10 flex min-h-36 scroll-mt-6 flex-col items-start justify-center gap-6 py-8 md:flex-row md:items-center md:justify-between">
        <a href="/" aria-label="tanh.xyz home" class="inline-flex py-2 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2d5da1]">
          <FooterWordmark />
        </a>
        <nav aria-label="Footer" class="flex flex-wrap gap-x-4 gap-y-2 font-sans text-lg text-[#2d5da1] md:gap-x-6 md:text-xl">
          <a href="/demo" class="hover:text-[#d93636]">Neural demos</a>
          <span aria-hidden="true" class="text-[#2d2d2d]">·</span>
          <a href="/research-notes/" class="hover:text-[#d93636]">Research notes</a>
          <span aria-hidden="true" class="text-[#2d2d2d]">·</span>
          <a href="/paper-notes/" class="hover:text-[#d93636]">Paper notes</a>
        </nav>
      </footer>
    </div>
  )
}
