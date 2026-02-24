import { createFileRoute } from '@tanstack/solid-router'
import { AdditionGrokkingPage } from '../../demos/addition-grokking/AdditionGrokkingDemo'

export const Route = createFileRoute('/demos/addition-grokking')({ component: AdditionGrokkingPage })
