import { describe, expect, it } from 'vitest'
import { hrefWithBasePath, normalizeRouterBasePath, stripBasePath } from '@/lib/env'

describe('router base path helpers', () => {
  it('normalizes empty and relative base paths', () => {
    expect(normalizeRouterBasePath(undefined)).toBe('/')
    expect(normalizeRouterBasePath('./')).toBe('/')
    expect(normalizeRouterBasePath('mesh/llm/ui-preview/')).toBe('/mesh/llm/ui-preview')
  })

  it('builds public hrefs for root and nested routes', () => {
    expect(hrefWithBasePath('/', '/mesh/llm/ui-preview')).toBe('/mesh/llm/ui-preview/')
    expect(hrefWithBasePath('/chat', '/mesh/llm/ui-preview')).toBe('/mesh/llm/ui-preview/chat')
    expect(hrefWithBasePath('configuration/defaults', '/mesh/llm/ui-preview/')).toBe('/mesh/llm/ui-preview/configuration/defaults')
  })

  it('leaves hrefs root-relative when mounted at the origin root', () => {
    expect(hrefWithBasePath('/', '/')).toBe('/')
    expect(hrefWithBasePath('/configuration/defaults', '/')).toBe('/configuration/defaults')
  })

  it('strips a public base path before matching route-local pathnames', () => {
    expect(stripBasePath('/mesh/llm/ui-preview/chat', '/mesh/llm/ui-preview')).toBe('/chat')
    expect(stripBasePath('/mesh/llm/ui-preview', '/mesh/llm/ui-preview')).toBe('/')
    expect(stripBasePath('/chat', '/mesh/llm/ui-preview')).toBe('/chat')
  })
})
