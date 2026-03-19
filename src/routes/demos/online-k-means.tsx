import { createFileRoute } from '@tanstack/solid-router'
import { OnlineKMeansPage } from '../../demos/online-k-means/OnlineKMeansDemo'

export const Route = createFileRoute('/demos/online-k-means')({ component: OnlineKMeansPage })
