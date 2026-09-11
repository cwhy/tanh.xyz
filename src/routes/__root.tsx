import {
  HeadContent,
  Outlet,
  Scripts,
  createRootRouteWithContext,
} from '@tanstack/solid-router'

import { HydrationScript } from 'solid-js/web'
import { Suspense } from 'solid-js'

import styleCss from '../styles.css?url'

export const Route = createRootRouteWithContext()({
  head: () => ({
    links: [
      { rel: 'stylesheet', href: styleCss },
      { rel: 'icon', type: 'image/svg+xml', href: '/logo-sq.svg' },
    ],
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'tanh.xyz - Neural Network Playground' },
      { name: 'description', content: 'Interactive neural network demo running in your browser with WebGPU' },
    ],
  }),
  notFoundComponent: () => (
    <main class="min-h-screen flex items-center justify-center p-6">
      <div class="text-center space-y-3">
        <h1 class="text-3xl font-bold">Page Not Found</h1>
        <a href="/" class="link link-primary">Back to Site Index</a>
      </div>
    </main>
  ),
  errorComponent: (props: { error: unknown; reset: () => void }) => (
    <main class="min-h-screen flex items-center justify-center p-6">
      <div class="max-w-xl rounded-xl border border-error/30 bg-error/10 p-5 space-y-3">
        <h1 class="text-2xl font-bold text-error">Route Error</h1>
        <p class="text-sm opacity-80">
          {props.error instanceof Error ? props.error.message : String(props.error)}
        </p>
        <button class="btn btn-sm btn-error" onClick={() => props.reset()}>
          Retry
        </button>
      </div>
    </main>
  ),
  shellComponent: RootComponent,
})

function RootComponent() {
  return (
    <html data-theme="light">
      <head>
        <HydrationScript />
      </head>
      <body>
        <HeadContent />
        <Suspense>
          <Outlet />
        </Suspense>
        <Scripts />
      </body>
    </html>
  )
}
