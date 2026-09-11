import { createFileRoute } from '@tanstack/solid-router'
import { OnlineKMeansPage } from '../../demos/online-k-means/OnlineKMeansDemo'

export const Route = createFileRoute('/demo/online-k-means')({ component: OnlineKMeansPage })
