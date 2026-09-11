import { createFileRoute } from '@tanstack/solid-router'
import { MnistNkClusteringPage } from '../../demos/mnist-nk-clustering/MnistNkClusteringDemo'

export const Route = createFileRoute('/demo/mnist-nk-clustering')({ component: MnistNkClusteringPage })
