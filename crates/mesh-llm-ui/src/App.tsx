import { RouterProvider } from '@tanstack/react-router'
import { AppProviders } from '@/app/providers/AppProviders'
import { router } from '@/app/router/router'

export function App() {
  return (
    <AppProviders initialDataMode="live" persistDataMode={false}>
      <RouterProvider router={router} />
    </AppProviders>
  )
}

/* eslint-disable react-refresh/only-export-components */
// Re-exports for backward compatibility with tests
export {
  attachmentForMessage,
  ChatPage,
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText
} from '@/lib/chatExports'
/* eslint-enable react-refresh/only-export-components */
