import { Show } from 'solid-js'
import type { JSX } from 'solid-js'

// --- Types ---

export type CardDecoration = 'none' | 'dog-ear-tr' | 'dog-ear-bl' | 'torn-top' | 'tape-tl' | 'tape-tr'

export interface CardVariant {
    borderRadius: string
    rotation: number
    decoration: CardDecoration
}

// --- Data ---

const BORDER_RADII = [
    '255px 15px 225px 15px / 15px 225px 15px 255px', // Classic wobbly
    '15px 225px 15px 255px / 255px 15px 225px 15px', // Flipped wobbly
    '55px 15px 55px 15px / 15px 55px 15px 55px', // Top/bottom symmetry
    '185px 25px 155px 15px / 25px 185px 15px 155px', // Asymmetric top-heavy
    '25px 185px 25px 155px / 155px 25px 185px 25px', // Asymmetric bottom-heavy
    '125px 45px 185px 25px / 45px 125px 25px 185px', // Mixed wobbly
    '225px 25px 15px 185px / 25px 255px 185px 15px', // Boxy wobbly
]

const DECORATIONS: CardDecoration[] = [
    'none', 'none', // Weight 'none' more heavily so not EVERY card has a decoration
    'dog-ear-tr',
    'dog-ear-bl',
    'torn-top',
    'tape-tl',
    'tape-tr'
]

// Base rotations are smaller (-0.6 to 0.6)
const ROTATIONS = [
    -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6
]

// --- Hashing & Generation ---

// Simple string hash function to generate a stable number from the title
function hashString(str: string): number {
    let hash = 0
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i)
        hash = (hash << 5) - hash + char
        hash = hash & hash // Convert to 32bit integer
    }
    return Math.abs(hash)
}

export function getCardVariantForTitle(title: string): CardVariant {
    const hash = hashString(title)

    // Extract pieces of the hash for different properties
    // Use prime modulos to minimize correlations between properties
    const rxIndex = hash % BORDER_RADII.length
    const decIndex = (hash >> 3) % DECORATIONS.length
    const rotIndex = (hash >> 5) % ROTATIONS.length

    return {
        borderRadius: BORDER_RADII[rxIndex],
        rotation: ROTATIONS[rotIndex],
        decoration: DECORATIONS[decIndex],
    }
}

// Additional variants for SeriesCard stack items
export function getStackVariantForTitle(title: string, layerIndex: number): CardVariant {
    const baseHash = hashString(title)
    // Create a new hash bounded to this specific layer
    const layerHash = baseHash ^ (layerIndex * 16777619)

    const rxIndex = layerHash % BORDER_RADII.length

    // We want the rotation of back-cards in a stack to contrast slightly with the front card
    // Base rotation gets reversed or shifted to create the "messy stack" look
    const rotIndex = (layerHash >> 5) % ROTATIONS.length
    let stackRotation = ROTATIONS[rotIndex]

    // Exaggerate rotation slightly for background stack items so they peek out more
    if (layerIndex > 0) {
        stackRotation = stackRotation > 0 ? stackRotation + 0.4 : stackRotation - 0.4
    }

    return {
        borderRadius: BORDER_RADII[rxIndex],
        rotation: stackRotation,
        decoration: 'none', // Background stack items shouldn't have tapes/dog-ears
    }
}

// --- Decoration Components ---

export function DogEar(props: { position: 'top-right' | 'bottom-left' }) {
    const isTopRight = () => props.position === 'top-right'
    return (
        <div
            class="absolute z-10"
            style={{
                top: isTopRight() ? '0' : 'auto',
                bottom: isTopRight() ? 'auto' : '0',
                right: isTopRight() ? '0' : 'auto',
                left: isTopRight() ? 'auto' : '0',
                width: '28px',
                height: '28px',
                background: isTopRight()
                    ? 'linear-gradient(225deg, var(--color-base-200) 50%, var(--color-base-300) 50%)'
                    : 'linear-gradient(45deg, var(--color-base-200) 50%, var(--color-base-300) 50%)',
            }}
        />
    )
}

export function TornEdge() {
    return (
        <div
            class="absolute top-0 left-0 right-0 z-10 pointer-events-none"
            style={{
                height: '10px',
                background: 'var(--color-base-200)',
                'clip-path': 'polygon(0% 0%, 100% 0%, 100% 40%, 97% 80%, 94% 30%, 91% 70%, 88% 20%, 85% 90%, 82% 50%, 79% 70%, 76% 10%, 73% 60%, 70% 30%, 67% 80%, 64% 40%, 61% 70%, 58% 20%, 55% 60%, 52% 40%, 49% 90%, 46% 30%, 43% 60%, 40% 50%, 37% 80%, 34% 20%, 31% 70%, 28% 40%, 25% 90%, 22% 30%, 19% 60%, 16% 50%, 13% 80%, 10% 20%, 7% 70%, 4% 40%, 1% 80%, 0% 30%)',
            }}
        />
    )
}

export function TapeDecoration(props: { position: 'top-left' | 'top-right' }) {
    const isLeft = () => props.position === 'top-left'
    return (
        <div
            class="absolute z-20 pointer-events-none"
            style={{
                top: '-8px',
                left: isLeft() ? '20px' : 'auto',
                right: isLeft() ? 'auto' : '20px',
                width: '56px',
                height: '20px',
                background: 'rgba(255, 223, 142, 0.75)',
                border: '1px solid rgba(210, 180, 100, 0.4)',
                transform: `rotate(${isLeft() ? -8 : 5}deg)`,
                'border-radius': '2px',
            }}
        />
    )
}

// --- Render helpers ---

export function renderOuterDecorations(variant: CardVariant): JSX.Element {
    return (
        <>
            <Show when={variant.decoration === 'tape-tl'}>
                <TapeDecoration position="top-left" />
            </Show>
            <Show when={variant.decoration === 'tape-tr'}>
                <TapeDecoration position="top-right" />
            </Show>
        </>
    )
}

export function renderInnerDecorations(variant: CardVariant): JSX.Element {
    return (
        <>
            <Show when={variant.decoration === 'dog-ear-tr'}>
                <DogEar position="top-right" />
            </Show>
            <Show when={variant.decoration === 'dog-ear-bl'}>
                <DogEar position="bottom-left" />
            </Show>
            <Show when={variant.decoration === 'torn-top'}>
                <TornEdge />
            </Show>
        </>
    )
}
