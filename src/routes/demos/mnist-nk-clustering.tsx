import { createFileRoute } from '@tanstack/solid-router'
import { MnistNkClusteringPage } from '../../demos/mnist-nk-clustering/MnistNkClusteringDemo'

export const Route = createFileRoute('/demos/mnist-nk-clustering')({ component: MnistNkClusteringPage })
