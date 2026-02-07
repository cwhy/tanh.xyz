import type { JSX } from 'solid-js'

interface ConfigSliderProps {
    label: string
    value: number
    min: number
    max: number
    step?: number
    onChange: (value: number) => void
    disabled?: boolean
    minLabel?: string
    maxLabel?: string
    helperText?: string
    valueFormatter?: (value: number) => string
    valueClass?: string
    rangeClass?: string
    labelInline?: boolean
}

export function ConfigSlider(props: ConfigSliderProps): JSX.Element {
    const valueText = () => (props.valueFormatter ? props.valueFormatter(props.value) : String(props.value))
    const isLabelInline = () => props.labelInline ?? true

    return (
        <div class="form-control">
            <label class={`label ${isLabelInline() ? 'pb-2 flex justify-between items-center' : 'pb-1'}`}>
                <span class="label-text font-medium">{props.label}</span>
                {isLabelInline() && (
                    <span class={props.valueClass ?? 'label-text-alt font-mono'}>{valueText()}</span>
                )}
            </label>
            {!isLabelInline() && (
                <div class={`text-right text-xs mb-1 ${props.valueClass ?? 'text-base-content/60 font-mono'}`}>
                    {valueText()}
                </div>
            )}
            <input
                type="range"
                min={props.min}
                max={props.max}
                step={props.step ?? 1}
                value={props.value}
                onInput={(e) => props.onChange(parseFloat(e.currentTarget.value))}
                class={props.rangeClass ?? 'range range-primary range-sm'}
                disabled={props.disabled}
            />
            <div class="flex justify-between text-xs text-base-content/40 mt-1">
                <span>{props.minLabel ?? props.min}</span>
                <span>{props.maxLabel ?? props.max}</span>
            </div>
            {props.helperText && (
                <p class="text-xs text-base-content/50 mt-1">{props.helperText}</p>
            )}
        </div>
    )
}
