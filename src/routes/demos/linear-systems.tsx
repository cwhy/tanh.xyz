import { createFileRoute } from '@tanstack/solid-router'
import { LinearSystemsDemo } from '../../demos/linear-systems/LinearSystemsDemo'

export const Route = createFileRoute('/demos/linear-systems')({ component: LinearSystemsDemo })
