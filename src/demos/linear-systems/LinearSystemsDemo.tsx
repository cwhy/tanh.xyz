import { onMount, onCleanup } from 'solid-js'
import { ControlSignalBoard } from '../../components/demo-layouts/ControlSignalBoard'
import { ControlPanel } from './ControlPanel'
import { VisualizationPanel } from './VisualizationPanel'
import { simulationStore } from './store'

export function LinearSystemsDemo() {
    onMount(() => {
        simulationStore.initialize()
    })

    onCleanup(() => {
        if (simulationStore.state.isPlaying) {
            simulationStore.togglePlaying()
        }
    })
    return (
        <ControlSignalBoard
            title="Linear Systems"
            subtitle="Matrix multiplication representing linear state evolution over time."
            backHref="/"
            backLabel="Back"
            initialTopHeightVh={50}
            top={<VisualizationPanel />}
            bottom={<ControlPanel />}
        />
    )
}
