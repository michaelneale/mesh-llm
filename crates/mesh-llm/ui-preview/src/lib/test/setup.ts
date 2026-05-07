import '@testing-library/jest-dom/vitest'

import { afterEach } from 'vitest'

class TestResizeObserver implements ResizeObserver {
  constructor(_callback: ResizeObserverCallback) {}

  observe(_target: Element, _options?: ResizeObserverOptions): void {}

  unobserve(_target: Element): void {}

  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = TestResizeObserver
}

function createTestStorage(): Storage {
  const values = new Map<string, string>()

  return {
    get length() {
      return values.size
    },
    clear() {
      values.clear()
    },
    getItem(key: string) {
      return values.get(key) ?? null
    },
    key(index: number) {
      return Array.from(values.keys())[index] ?? null
    },
    removeItem(key: string) {
      values.delete(key)
    },
    setItem(key: string, value: string) {
      values.set(key, String(value))
    }
  }
}

function isUsableStorage(storage: Storage | undefined): storage is Storage {
  return (
    typeof storage?.getItem === 'function' &&
    typeof storage.setItem === 'function' &&
    typeof storage.removeItem === 'function' &&
    typeof storage.clear === 'function'
  )
}

function storageFor(name: 'localStorage' | 'sessionStorage'): Storage | undefined {
  try {
    return window[name]
  } catch {
    return undefined
  }
}

function installStorageIfNeeded(name: 'localStorage' | 'sessionStorage') {
  if (isUsableStorage(storageFor(name))) return

  Object.defineProperty(window, name, {
    configurable: true,
    value: createTestStorage()
  })
}

installStorageIfNeeded('localStorage')
installStorageIfNeeded('sessionStorage')

afterEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()
})

Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
  configurable: true,
  value: () => undefined
})

Object.defineProperty(window, 'scrollTo', {
  configurable: true,
  value: () => undefined
})
