export type AssistantContentSegment =
  | {
      kind: 'thinking'
      text: string
      open: boolean
    }
  | {
      kind: 'response'
      text: string
    }

type SplitAssistantThinkingOptions = {
  streaming?: boolean
}

const THINK_OPEN_TAG = '<think>'
const THINK_CLOSE_TAG = '</think>'

function indexOfTag(value: string, tag: string, fromIndex: number) {
  for (let index = fromIndex; index <= value.length - tag.length; index += 1) {
    if (value.slice(index, index + tag.length).toLowerCase() === tag) return index
  }

  return -1
}

export function splitAssistantThinking(
  body: string,
  { streaming = false }: SplitAssistantThinkingOptions = {}
): AssistantContentSegment[] {
  if (body.length === 0) return []

  const segments: AssistantContentSegment[] = []
  let cursor = 0
  let firstSegment = true

  if (streaming && indexOfTag(body, THINK_OPEN_TAG, 0) === -1 && indexOfTag(body, THINK_CLOSE_TAG, 0) === -1) {
    return [{ kind: 'thinking', text: body, open: true }]
  }

  while (cursor < body.length) {
    const openIndex = indexOfTag(body, THINK_OPEN_TAG, cursor)
    const closeIndex = indexOfTag(body, THINK_CLOSE_TAG, cursor)

    if (closeIndex !== -1 && (openIndex === -1 || closeIndex < openIndex)) {
      const thinkingText = body.slice(cursor, closeIndex)
      if (thinkingText.length > 0) {
        segments.push({ kind: 'thinking', text: thinkingText, open: false })
      }
      cursor = closeIndex + THINK_CLOSE_TAG.length
      firstSegment = false
      continue
    }

    if (openIndex === -1) {
      const responseText = body.slice(cursor)
      if (responseText.length > 0) {
        segments.push({ kind: 'response', text: responseText })
      }
      break
    }

    const responseText = body.slice(cursor, openIndex)
    if (responseText.length > 0 || (firstSegment && openIndex > cursor)) {
      segments.push({ kind: 'response', text: responseText })
    }

    const thinkingStart = openIndex + THINK_OPEN_TAG.length
    const thinkingEnd = indexOfTag(body, THINK_CLOSE_TAG, thinkingStart)
    if (thinkingEnd === -1) {
      const thinkingText = body.slice(thinkingStart)
      segments.push({ kind: 'thinking', text: thinkingText, open: true })
      break
    }

    const thinkingText = body.slice(thinkingStart, thinkingEnd)
    if (thinkingText.length > 0) {
      segments.push({ kind: 'thinking', text: thinkingText, open: false })
    }
    cursor = thinkingEnd + THINK_CLOSE_TAG.length
    firstSegment = false
  }

  return segments
}
