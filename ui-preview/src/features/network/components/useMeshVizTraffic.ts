import { useCallback, useRef } from 'react'
import { animate, type JSAnimation } from 'animejs'
import type { MeshLink } from '@/features/network/lib/mesh-links'
import { packetTransform, type Point, type Viewport } from '@/features/network/lib/mesh-viewport'
import type { MeshNode } from '@/features/app-tabs/types'

type TrafficPacket = {
  element: HTMLSpanElement
  position: Point
  animation?: JSAnimation
  fadeOutTimer?: number
}

type MeshVizTrafficOptions = {
  canvasRef: React.RefObject<HTMLDivElement | null>
  canvasSizeRef: React.RefObject<{ width: number; height: number }>
  links: MeshLink[]
  liveLayerBaseViewportRef: React.RefObject<Viewport>
  liveLayerTransformActiveRef: React.RefObject<boolean>
  packetLayerRef: React.RefObject<HTMLDivElement | null>
  reduceMotion: boolean
  renderNodes: MeshNode[]
  nodeColorForTraffic: (node: MeshNode) => string
  selfId: string
  updateCanvasSize: () => void
  viewportRef: React.RefObject<Viewport>
}

const PACKET_ANIMATION_DURATION = 2500
const PACKET_EASE = 'inOutExpo'
const PACKET_FADE_DURATION = 180
const PACKET_VISIBLE_OPACITY = '0.92'
const PACKET_HIDDEN_OPACITY = '0'
const PACKET_CLASS_NAME = 'mesh-packet absolute left-0 top-0 block size-2 rounded-full'

export function packetColor(sourceColor: string, targetColor: string, progress: number) {
  const boundedProgress = Math.min(1, Math.max(0, Number.isFinite(progress) ? progress : 0))

  if (boundedProgress === 0 || sourceColor === targetColor) return sourceColor
  if (boundedProgress === 1) return targetColor

  return `color-mix(in oklab, ${targetColor} ${(boundedProgress * 100).toFixed(2)}%, ${sourceColor})`
}

function applyPacketColor(element: HTMLSpanElement, sourceColor: string, targetColor: string, progress: number) {
  const color = packetColor(sourceColor, targetColor, progress)

  element.style.background = color
  element.style.color = color
}

export function useMeshVizTraffic({
  canvasRef,
  canvasSizeRef,
  links,
  liveLayerBaseViewportRef,
  liveLayerTransformActiveRef,
  packetLayerRef,
  reduceMotion,
  renderNodes,
  nodeColorForTraffic,
  selfId,
  updateCanvasSize,
  viewportRef
}: MeshVizTrafficOptions) {
  const trafficPacketsRef = useRef<Map<number, TrafficPacket>>(new Map())
  const trafficPacketIdRef = useRef(0)

  const placePacket = useCallback(
    (element: HTMLSpanElement, point: Point) => {
      // While a live pan/zoom transform is active, packet spans need to stay in the same
      // committed viewport coordinate space as the React-rendered nodes/links. Their parent
      // packet layer applies the live transform, so using viewportRef here would double-apply
      // the in-progress pan/zoom and make packets drift until the viewport commit lands.
      const placementViewport = liveLayerTransformActiveRef.current
        ? liveLayerBaseViewportRef.current
        : viewportRef.current

      element.style.transform = packetTransform(
        point,
        canvasSizeRef.current.width,
        canvasSizeRef.current.height,
        placementViewport
      )
    },
    [canvasSizeRef, liveLayerBaseViewportRef, liveLayerTransformActiveRef, viewportRef]
  )

  const hidePacket = useCallback((element: HTMLSpanElement) => {
    element.style.opacity = PACKET_HIDDEN_OPACITY
  }, [])

  const showPacket = useCallback((element: HTMLSpanElement) => {
    element.getBoundingClientRect()
    element.style.opacity = PACKET_VISIBLE_OPACITY
  }, [])

  const removeTrafficPacket = useCallback((packetId: number) => {
    const packet = trafficPacketsRef.current.get(packetId)

    if (!packet) {
      return
    }

    trafficPacketsRef.current.delete(packetId)
    if (packet.fadeOutTimer !== undefined) {
      window.clearTimeout(packet.fadeOutTimer)
    }
    packet.element.remove()
  }, [])

  const clearTrafficPackets = useCallback(() => {
    trafficPacketsRef.current.forEach((packet) => {
      packet.animation?.revert()
      if (packet.fadeOutTimer !== undefined) {
        window.clearTimeout(packet.fadeOutTimer)
      }
      packet.element.remove()
    })
    trafficPacketsRef.current.clear()
  }, [])

  const createTrafficPacketElement = useCallback(
    (initialColor: string) => {
      const packetLayerElement = packetLayerRef.current

      if (!packetLayerElement) {
        return undefined
      }

      const packetElement = document.createElement('span')
      packetElement.className = PACKET_CLASS_NAME
      packetElement.style.background = initialColor
      packetElement.style.color = initialColor
      packetElement.style.opacity = PACKET_HIDDEN_OPACITY
      packetElement.style.transition = `opacity ${PACKET_FADE_DURATION}ms ease-out`
      packetLayerElement.append(packetElement)

      return packetElement
    },
    [packetLayerRef]
  )

  const placeTrafficPackets = useCallback(() => {
    trafficPacketsRef.current.forEach((packet) => {
      placePacket(packet.element, packet.position)
    })
  }, [placePacket])

  const playTraffic = useCallback(
    (sourceNodeId: string, targetNodeId: string) => {
      if (sourceNodeId === targetNodeId) {
        return false
      }

      const source = renderNodes.find((node) => node.id === sourceNodeId)
      const target = renderNodes.find((node) => node.id === targetNodeId)

      if (!source || !target || !canvasRef.current) {
        return false
      }

      const linkIndex = links.findIndex(
        (link) =>
          (link.source.id === source.id && link.target.id === target.id) ||
          (link.source.id === target.id && link.target.id === source.id)
      )

      if (linkIndex === -1) {
        return false
      }

      const sourceColor = nodeColorForTraffic(source)
      const targetColor = nodeColorForTraffic(target)
      const packetElement = createTrafficPacketElement(sourceColor)

      if (!packetElement) {
        return false
      }

      updateCanvasSize()

      const packetId = trafficPacketIdRef.current
      trafficPacketIdRef.current += 1

      const packetPosition = { x: source.x, y: source.y, colorProgress: 0 }
      trafficPacketsRef.current.set(packetId, { element: packetElement, position: packetPosition })
      applyPacketColor(packetElement, sourceColor, targetColor, 0)
      placePacket(packetElement, packetPosition)
      showPacket(packetElement)

      if (reduceMotion) {
        const targetPosition = { x: target.x, y: target.y }
        const packet = trafficPacketsRef.current.get(packetId)

        if (packet) {
          packet.position = targetPosition
        }

        placePacket(packetElement, targetPosition)
        applyPacketColor(packetElement, sourceColor, targetColor, 1)
        hidePacket(packetElement)
        removeTrafficPacket(packetId)
        return true
      }

      const animation = animate(packetPosition, {
        x: { from: source.x, to: target.x },
        y: { from: source.y, to: target.y },
        colorProgress: { from: 0, to: 1 },
        duration: PACKET_ANIMATION_DURATION,
        ease: PACKET_EASE,
        loop: false,
        onUpdate: () => {
          const packet = trafficPacketsRef.current.get(packetId)

          if (!packet) {
            return
          }

          packet.position = packetPosition
          applyPacketColor(packet.element, sourceColor, targetColor, packetPosition.colorProgress)
          placePacket(packet.element, packet.position)
        },
        onComplete: () => {
          const packet = trafficPacketsRef.current.get(packetId)

          if (!packet) {
            return
          }

          packet.position = { x: target.x, y: target.y }
          applyPacketColor(packet.element, sourceColor, targetColor, 1)
          hidePacket(packet.element)
          packet.fadeOutTimer = window.setTimeout(() => {
            removeTrafficPacket(packetId)
          }, PACKET_FADE_DURATION)
        }
      })

      const packet = trafficPacketsRef.current.get(packetId)

      if (packet) {
        packet.animation = animation
      }

      return true
    },
    [
      canvasRef,
      createTrafficPacketElement,
      hidePacket,
      links,
      nodeColorForTraffic,
      placePacket,
      reduceMotion,
      removeTrafficPacket,
      renderNodes,
      showPacket,
      updateCanvasSize
    ]
  )

  const playRandomTraffic = useCallback(() => {
    if (links.length === 0) {
      return false
    }

    const link = links[Math.floor(Math.random() * links.length)]
    const reverse = Math.random() >= 0.5

    return playTraffic(reverse ? link.target.id : link.source.id, reverse ? link.source.id : link.target.id)
  }, [links, playTraffic])

  const playSelfTraffic = useCallback(() => {
    const selfLinks = links.filter((link) => link.source.id === selfId || link.target.id === selfId)
    const link = selfLinks[Math.floor(Math.random() * selfLinks.length)]

    if (!link) {
      return false
    }

    return playTraffic(selfId, link.source.id === selfId ? link.target.id : link.source.id)
  }, [links, playTraffic, selfId])

  return {
    clearTrafficPackets,
    placeTrafficPackets,
    playRandomTraffic,
    playSelfTraffic,
    playTraffic
  }
}
