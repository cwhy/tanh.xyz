import type { JSX } from 'solid-js'


interface DemoCardProps {
    title: string
    description: string
    href: string
    icon: JSX.Element
    status?: 'ready' | 'coming-soon'
}

export function DemoCard(props: DemoCardProps) {
    const isReady = () => props.status !== 'coming-soon'

    return (
        <a
            href={isReady() ? props.href : '#'}
            class="card bg-base-100 shadow-[4px_4px_0px_0px_#2d2d2d] border-[3px] border-base-content hover:shadow-[2px_2px_0px_0px_#2d2d2d] hover:translate-x-[2px] hover:translate-y-[2px] transition-all duration-100 group"
            style={{
                'border-radius': '255px 15px 225px 15px / 15px 225px 15px 255px',
                opacity: isReady() ? 1 : 0.6,
                cursor: isReady() ? 'pointer' : 'not-allowed'
            }}
            onClick={(e: MouseEvent) => {
                if (!isReady()) e.preventDefault()
            }}
        >
            <div class="card-body p-6">
                <div class="flex items-start gap-4">
                    <div class="p-3 bg-primary/10 rounded-full border-2 border-primary/30 group-hover:bg-primary/20 transition-colors">
                        {props.icon}
                    </div>
                    <div class="flex-1">
                        <h3 class="card-title text-xl font-bold mb-2 flex items-center gap-2">
                            {props.title}
                            {!isReady() && (
                                <span class="badge badge-sm bg-base-300 border-base-content/20 text-xs">
                                    Coming Soon
                                </span>
                            )}
                        </h3>
                        <p class="text-base-content/70 text-sm leading-relaxed">
                            {props.description}
                        </p>
                    </div>
                </div>
            </div>
        </a>
    )
}
