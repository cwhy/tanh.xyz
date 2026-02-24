import { defaultDevice, init } from '@jax-js/jax'

let initialized = false

/**
 * Initialize jax-js with WebGPU or WASM fallback.
 */
export async function initJax(): Promise<string> {
    if (initialized) return defaultDevice() as string

    const devices = await init()
    if (devices.includes('webgpu')) {
        defaultDevice('webgpu')
        console.log('Using WebGPU backend')
    } else {
        console.log('WebGPU not available, using WASM backend')
    }

    initialized = true
    return defaultDevice() as string
}
