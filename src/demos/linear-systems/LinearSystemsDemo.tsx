import { onSettled } from 'solid-js'
import { ControlSignalBoard } from '../../components/demo-layouts/ControlSignalBoard'
import { ControlPanel } from './ControlPanel'
import { VisualizationPanel } from './VisualizationPanel'
import { createSimulationStore, SimulationContext } from './store'

export function LinearSystemsDemo() {
    const simulationStore = createSimulationStore()
    onSettled(() => {
        void simulationStore.initialize()
        return simulationStore.dispose
    })
    return (
        <SimulationContext value={simulationStore}>
            <ControlSignalBoard
                title="Linear Systems"
                subtitle="Matrix multiplication representing linear state evolution over time."
                backHref="/demo"
                backLabel="Back"
                initialTopHeightVh={50}
                top={<VisualizationPanel />}
                bottom={<ControlPanel />}
            />
        </SimulationContext>
    )
}
