import { createFileRoute } from '@tanstack/solid-router'
import { DenseAssociativeMemoryPage } from '../../demos/dense-associative-memory/DenseAssociativeMemoryDemo'

export const Route = createFileRoute('/demo/dense-associative-memory')({ component: DenseAssociativeMemoryPage })
