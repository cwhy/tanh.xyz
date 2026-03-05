import { createFileRoute } from '@tanstack/solid-router'
import { RivalKPage } from '../../demos/rival-k/RivalKDemo'

export const Route = createFileRoute('/demos/rival-k')({ component: RivalKPage })
