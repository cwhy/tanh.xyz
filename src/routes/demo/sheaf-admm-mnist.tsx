import { createFileRoute } from '@tanstack/solid-router'
import { SheafAdmmMnistPage } from '../../demos/sheaf-admm-mnist/SheafAdmmMnistDemo'

export const Route = createFileRoute('/demo/sheaf-admm-mnist')({ component: SheafAdmmMnistPage })
