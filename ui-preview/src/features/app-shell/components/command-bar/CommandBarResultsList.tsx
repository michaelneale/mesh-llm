import { Loader2 } from 'lucide-react'
import type { ComponentType, HTMLAttributes } from 'react'
import { CommandBarOptionRow } from './CommandBarOptionRow'
import { Alert, AlertDescription, AlertTitle } from './command-bar-primitives'
import type {
  CommandBarBehavior,
  CommandBarMode,
  CommandBarNormalizedResult,
  CommandBarResultContainerProps,
} from './command-bar-types'

interface CommandBarResultsListProps<T> {
  activeIndex: number
  activeModeId: string | null
  asyncErrorMessage?: string
  behavior: CommandBarBehavior
  emptyMessage: string
  listProps: HTMLAttributes<HTMLDivElement>
  listboxId: string
  modeById: ReadonlyMap<string, CommandBarMode<T>>
  onRegisterOptionElement: (compositeKey: string, node: HTMLDivElement | null) => void
  onSelectResult: (result: CommandBarNormalizedResult<T>) => void
  onSetActiveIndex: (index: number) => void
  query: string
  results: readonly CommandBarNormalizedResult<T>[]
  ResultContainer: ComponentType<CommandBarResultContainerProps<T>>
  selectionError: string | null
  showErrorState: boolean
  showLoadingState: boolean
}

export function CommandBarResultsList<T>({
  activeIndex,
  activeModeId,
  asyncErrorMessage,
  behavior,
  emptyMessage,
  listProps,
  listboxId,
  modeById,
  onRegisterOptionElement,
  onSelectResult,
  onSetActiveIndex,
  query,
  results,
  ResultContainer,
  selectionError,
  showErrorState,
  showLoadingState,
}: CommandBarResultsListProps<T>) {
  return (
    <div data-testid="command-bar-results" className="bg-panel px-2.5 py-2.5">
      {selectionError ? (
        <Alert variant="destructive" className="mb-2">
          <AlertTitle>Action failed</AlertTitle>
          <AlertDescription>{selectionError}</AlertDescription>
        </Alert>
      ) : null}

      <div {...listProps}>
        {results.length > 0 ? (
          <ResultContainer listProps={listProps} query={query} modeId={activeModeId} activeIndex={activeIndex} results={results}>
            {results.map((result, index) => {
              const mode = modeById.get(result.modeId)
              const ResultItem = mode?.ResultItem

              return (
                <CommandBarOptionRow
                  key={result.compositeKey}
                  behavior={behavior}
                  isActive={index === activeIndex}
                  listboxId={listboxId}
                  optionClassName={mode?.optionClassName}
                  optionRef={(node) => {
                    onRegisterOptionElement(result.compositeKey, node)
                  }}
                  result={result}
                  onClick={() => {
                    onSelectResult(result)
                  }}
                  onPointerMove={() => {
                    if (index === activeIndex) return
                    onSetActiveIndex(index)
                  }}
                  renderItem={(currentResult, isActive) => {
                    if (ResultItem) {
                      return (
                        <ResultItem
                          item={currentResult.item}
                          selected={isActive}
                          query={query}
                          modeLabel={currentResult.modeLabel}
                        />
                      )
                    }

                    return <div className="truncate text-[length:var(--density-type-control)] font-medium text-foreground">{currentResult.searchText}</div>
                  }}
                />
              )
            })}
          </ResultContainer>
        ) : showLoadingState ? (
          <div className="flex min-h-28 items-center gap-3 px-3 py-6 text-[length:var(--density-type-caption)] text-muted-foreground">
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            <div>
              <div className="font-medium text-foreground">Loading results</div>
              <div className="text-[length:var(--density-type-label)] text-muted-foreground">Checking the latest matches.</div>
            </div>
          </div>
        ) : showErrorState ? (
          <div className="px-3 py-6 text-[length:var(--density-type-caption)] text-muted-foreground">
            <div className="font-medium text-foreground">Could not load results</div>
            <div className="text-[length:var(--density-type-label)] text-muted-foreground">{asyncErrorMessage}</div>
          </div>
        ) : (
          <div className="px-3 py-6 text-[length:var(--density-type-caption)] text-muted-foreground">{emptyMessage}</div>
        )}
      </div>
    </div>
  )
}
