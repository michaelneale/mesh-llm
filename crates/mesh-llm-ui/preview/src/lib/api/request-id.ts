export function generateRequestId(): string {
  return crypto.randomUUID()
}
