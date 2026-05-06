import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

type CommandBarContextValue = {
  isOpen: boolean;
  query: string;
  setQuery: Dispatch<SetStateAction<string>>;
  activeModeId: string | null;
  setActiveModeId: Dispatch<SetStateAction<string | null>>;
  activeIndex: number;
  setActiveIndex: Dispatch<SetStateAction<number>>;
  selectionError: string | null;
  setSelectionError: Dispatch<SetStateAction<string | null>>;
  returnFocusElement: HTMLElement | null;
  openCommandBar: (modeId?: string) => void;
  closeCommandBar: () => void;
  toggleCommandBar: (modeId?: string) => void;
};

export const CommandBarContext = createContext<CommandBarContextValue | null>(null);

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') {
    return true;
  }

  return target.closest('[contenteditable="true"]') !== null;
}

function isMacPlatform() {
  if (typeof navigator === 'undefined') return false;
  return /Mac|iPhone|iPad|iPod/.test(navigator.userAgent);
}

export function CommandBarProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [activeModeId, setActiveModeId] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [returnFocusElement, setReturnFocusElement] = useState<HTMLElement | null>(null);
  const isOpenRef = useRef(isOpen);
  const returnFocusElementRef = useRef(returnFocusElement);

  useEffect(() => {
    isOpenRef.current = isOpen;
  }, [isOpen]);

  useEffect(() => {
    returnFocusElementRef.current = returnFocusElement;
  }, [returnFocusElement]);

  const openCommandBar = useCallback((modeId?: string) => {
    if (!isOpenRef.current) {
      const activeElement = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
      returnFocusElementRef.current = activeElement;
      setReturnFocusElement(activeElement);
    }

    if (modeId !== undefined) setActiveModeId(modeId);
    setActiveIndex(0);
    setSelectionError(null);
    setIsOpen(true);
  }, []);

  const closeCommandBar = useCallback(() => {
    if (!isOpenRef.current) return;

    const focusTarget = returnFocusElementRef.current;

    setIsOpen(false);
    setQuery('');
    setActiveModeId(null);
    setActiveIndex(0);
    setSelectionError(null);
    setReturnFocusElement(null);
    returnFocusElementRef.current = null;

    if (focusTarget?.isConnected) focusTarget.focus();
  }, []);

  const toggleCommandBar = useCallback((modeId?: string) => {
    if (isOpenRef.current) {
      closeCommandBar();
      return;
    }

    openCommandBar(modeId);
  }, [closeCommandBar, openCommandBar]);

  useEffect(() => {
     const onKeyDown = (event: KeyboardEvent) => {
       if (isOpenRef.current && event.key === 'Escape') {
         event.preventDefault();
         closeCommandBar();
         return;
       }

       // Hotkey (cmd-k) is disabled
       // if (event.key.toLowerCase() !== 'k' || event.altKey || event.shiftKey) return;
       // const shouldOpen = isMacPlatform() ? event.metaKey : event.ctrlKey;
       // if (!shouldOpen) return;
       // if (!isOpenRef.current && isEditableTarget(event.target)) return;
       // event.preventDefault();
       // openCommandBar();
     };

     window.addEventListener('keydown', onKeyDown);
     return () => window.removeEventListener('keydown', onKeyDown);
   }, [closeCommandBar]);

  const value = useMemo<CommandBarContextValue>(
    () => ({
      isOpen,
      query,
      setQuery,
      activeModeId,
      setActiveModeId,
      activeIndex,
      setActiveIndex,
      selectionError,
      setSelectionError,
      returnFocusElement,
      openCommandBar,
      closeCommandBar,
      toggleCommandBar,
    }),
    [
      activeIndex,
      activeModeId,
      closeCommandBar,
      isOpen,
      openCommandBar,
      query,
      returnFocusElement,
      selectionError,
      toggleCommandBar,
    ],
  );

  return <CommandBarContext.Provider value={value}>{children}</CommandBarContext.Provider>;
}
