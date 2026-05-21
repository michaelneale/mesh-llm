type ReserveWakeStatProps = {
  label: string
  value: string | number
  title?: string
}

export function ReserveWakeStat({ label, value, title }: ReserveWakeStatProps) {
  return (
    <div className="flex flex-col items-center px-0 text-center leading-[1.1]" title={title}>
      <div className="whitespace-nowrap text-[13px] font-medium leading-none tabular-nums text-foreground">{value}</div>
      <div className="mt-[2px] text-[9.5px] font-medium uppercase leading-none tracking-[0.05em] text-fg-faint">
        {label}
      </div>
    </div>
  )
}
