export function formatReserveVram(valueGB: number) {
  if (valueGB >= 1000) {
    const valueTB = valueGB / 1000
    return `${formatNumber(valueTB)} TB`
  }

  return `${formatNumber(valueGB)} GB`
}

export function formatReserveEta(seconds: number) {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  if (seconds < 86_400) return `${Math.ceil(seconds / 3600)} hr`
  return `${Math.ceil(seconds / 86_400)} d`
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)
}
