import { createFileRoute } from '@tanstack/solid-router'
import { TwoMoonPage } from '../../demos/two-moons-clustering/TwoMoonDemo'

export const Route = createFileRoute('/demo/two-moon')({ component: TwoMoonPage })
