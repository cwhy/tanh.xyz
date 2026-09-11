import { createFileRoute, redirect } from '@tanstack/solid-router'

export const Route = createFileRoute('/demos/$slug')({
  beforeLoad: ({ params }) => {
    throw redirect({ href: `/demo/${params.slug}`, statusCode: 308 })
  },
})
