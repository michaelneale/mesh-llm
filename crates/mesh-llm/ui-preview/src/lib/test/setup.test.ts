import { describe, expect, it } from 'vitest'

describe('test environment setup', () => {
  it('provides usable web storage APIs', () => {
    window.localStorage.setItem('mesh-test-key', 'mesh-value')
    window.sessionStorage.setItem('mesh-session-key', 'session-value')

    expect(window.localStorage.getItem('mesh-test-key')).toBe('mesh-value')
    expect(window.sessionStorage.getItem('mesh-session-key')).toBe('session-value')

    window.localStorage.removeItem('mesh-test-key')
    expect(window.localStorage.getItem('mesh-test-key')).toBeNull()
  })
})
