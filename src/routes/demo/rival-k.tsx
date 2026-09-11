import { createFileRoute } from '@tanstack/solid-router'
import { RivalKPage } from '../../demos/rival-k/RivalKDemo'

export const Route = createFileRoute('/demo/rival-k')({ component: RivalKPage })
