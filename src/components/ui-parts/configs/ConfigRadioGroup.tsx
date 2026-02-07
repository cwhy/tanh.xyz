import { For, Show, type JSX } from 'solid-js'

interface ConfigRadioOption<T extends string> {
    label: string
    value: T
}

interface ConfigRadioGroupProps<T extends string> {
    label: string
    name: string
    value: T
    options: Array<ConfigRadioOption<T>>
    onChange: (value: T) => void
    disabled?: boolean
    labelRight?: JSX.Element | string
    labelInline?: boolean
    labelRightClass?: string
    controlsInline?: boolean
}

export function ConfigRadioGroup<T extends string>(props: ConfigRadioGroupProps<T>): JSX.Element {
    const isLabelInline = () => props.labelInline ?? Boolean(props.labelRight)
    const isControlsInline = () => props.controlsInline ?? false

    const options = (
        <For each={props.options}>
            {(option) => (
                <input
                    class="join-item btn btn-sm btn-outline px-4"
                    type="radio"
                    name={props.name}
                    aria-label={option.label}
                    checked={props.value === option.value}
                    onClick={() => props.onChange(option.value)}
                    disabled={props.disabled}
                />
            )}
        </For>
    )

    return (
        <div class="form-control">
            <Show
                when={isControlsInline()}
                fallback={
                    <>
                        <label class={`label pb-2 ${isLabelInline() ? 'flex justify-between items-center' : ''}`}>
                            <span class="label-text font-medium">{props.label}</span>
                            <Show when={props.labelRight !== undefined}>
                                <span class={props.labelRightClass ?? 'label-text-alt text-base-content/60'}>{props.labelRight}</span>
                            </Show>
                        </label>
                        <div class="join flex-wrap">{options}</div>
                    </>
                }
            >
                <div class="flex flex-wrap items-center justify-between gap-3 pb-2">
                    <label class={`label py-0 ${isLabelInline() ? 'flex justify-between items-center' : ''}`}>
                        <span class="label-text font-medium">{props.label}</span>
                        <Show when={props.labelRight !== undefined}>
                            <span class={props.labelRightClass ?? 'label-text-alt text-base-content/60'}>{props.labelRight}</span>
                        </Show>
                    </label>
                    <div class="join flex-wrap justify-end">{options}</div>
                </div>
            </Show>
        </div>
    )
}
