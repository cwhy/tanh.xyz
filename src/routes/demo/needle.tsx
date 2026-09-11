import { createFileRoute } from '@tanstack/solid-router'
import { NeedlePage } from '../../demos/needle/NeedleDemo'

export const Route = createFileRoute('/demo/needle')({ component: NeedlePage })
