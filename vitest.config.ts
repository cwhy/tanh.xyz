import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  // Use the client scheduler even in Node so tests catch batched-write regressions.
  resolve: {
    alias: [
      { find: /^solid-js$/, replacement: fileURLToPath(new URL('./node_modules/solid-js/dist/solid.dev.js', import.meta.url)) },
      { find: /^@solidjs\/signals$/, replacement: fileURLToPath(new URL('./node_modules/@solidjs/signals/dist/dev.js', import.meta.url)) },
    ],
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
})
