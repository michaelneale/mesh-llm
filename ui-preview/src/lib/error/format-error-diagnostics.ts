export function formatErrorDiagnostics(error?: Error, componentStack?: string) {
  if (!error) return undefined

  const trace = error.stack || error.message
  if (!componentStack) return trace

  return `${trace}\n\nReact component stack:\n${componentStack.trim()}`
}
