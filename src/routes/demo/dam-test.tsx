import { createFileRoute } from '@tanstack/solid-router'
import { DamTestPage } from '../../demos/dense-associative-memory/DamTest'

export const Route = createFileRoute('/demo/dam-test')({ component: DamTestPage })
