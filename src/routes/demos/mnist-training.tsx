import { createFileRoute } from '@tanstack/solid-router'
import { MnistTrainingPage } from '../../demos/mnist-training/MnistTrainingDemo'

export const Route = createFileRoute('/demos/mnist-training')({ component: MnistTrainingPage })
