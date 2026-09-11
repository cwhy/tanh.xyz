import { createFileRoute, redirect } from '@tanstack/solid-router'

export const Route = createFileRoute('/demos/')({
  beforeLoad: () => {
    throw redirect({ href: '/demo', statusCode: 308 })
  },
})
