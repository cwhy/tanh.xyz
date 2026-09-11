import { createFileRoute } from '@tanstack/solid-router'
import { LinearSystemsDemo } from '../../demos/linear-systems/LinearSystemsDemo'

export const Route = createFileRoute('/demo/linear-systems')({ component: LinearSystemsDemo })
