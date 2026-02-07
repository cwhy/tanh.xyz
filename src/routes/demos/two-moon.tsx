import { createFileRoute } from '@tanstack/solid-router'
import { TwoMoonPage } from '../../demos/two-moons-clustering/TwoMoonDemo'

export const Route = createFileRoute('/demos/two-moon')({ component: TwoMoonPage })
