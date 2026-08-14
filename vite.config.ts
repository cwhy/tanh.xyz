import { defineConfig } from 'vite'
import { devtools } from '@tanstack/devtools-vite'
import viteTsConfigPaths from 'vite-tsconfig-paths'
import tailwindcss from '@tailwindcss/vite'

import { tanstackStart } from '@tanstack/solid-start/plugin/vite'
import solidPlugin from 'vite-plugin-solid'
import { nitro } from 'nitro/vite'

import lucidePreprocess from 'vite-plugin-lucide-preprocess'

export default defineConfig({
  ssr: {
    noExternal: [
      '@tanstack/router-core',
      '@tanstack/start-client-core',
      '@tanstack/start-server-core',
    ],
  },
  plugins: [
    lucidePreprocess(),
    devtools(),
    nitro({
      preset: 'cloudflare-module',
      // Keep local SSR on Nitro's Node runner. The Cloudflare Miniflare runner
      // cannot currently evaluate Vite's workspace-relative module paths.
      devServer: { runner: 'node-worker' },
    }),
    // this is the plugin that enables path aliases
    viteTsConfigPaths({
      projects: ['./tsconfig.json'],
    }),
    tailwindcss(),
    tanstackStart(),
    solidPlugin({ ssr: true }),
  ],
})
