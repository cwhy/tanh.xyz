import { createFileRoute } from '@tanstack/solid-router'
import { NKClusteringPage } from '../../demos/nk-clustering/NKClusteringDemo'

export const Route = createFileRoute('/demo/nk-clustering')({ component: NKClusteringPage })
