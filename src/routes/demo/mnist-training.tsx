import { createFileRoute } from '@tanstack/solid-router'
import { MnistTrainingPage } from '../../demos/mnist-training/MnistTrainingDemo'

export const Route = createFileRoute('/demo/mnist-training')({ component: MnistTrainingPage })
