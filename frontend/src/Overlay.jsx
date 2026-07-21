import { useState, useRef, useCallback, forwardRef, useImperativeHandle } from 'react'
import {
  NODE_STYLES,
  DEFAULT_NODE_STYLE,
  edgeHasPath,
  edgePoints,
  layoutTagLabels,
} from './overlayGeometry'

const FOOTER_HEIGHT = 90

const Overlay = forwardRef(function Overlay({ imageUrl, result, showEdges, selectedNodeId, onNodeClick }, ref) {
  const [natural, setNatural] = useState(null) // { w, h }
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 })
  const dragRef = useRef(null)
  const imgRef = useRef(null)

  const onImgLoad = (e) => {
    setNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight })
  }

  const onWheel = useCallback((e) => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
    setView((v) => ({ ...v, scale: Math.min(8, Math.max(0.2, v.scale * factor)) }))
  }, [])

  const onMouseDown = (e) => {
    dragRef.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
  }
  const onMouseMove = (e) => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.x
    const dy = e.clientY - dragRef.current.y
    setView((v) => ({ ...v, tx: dragRef.current.tx + dx, ty: dragRef.current.ty + dy }))
  }
  const onMouseUp = () => { dragRef.current = null }

  const nodesById = new Map((result?.nodes ?? []).map((n) => [n.id, n]))
  const labelLayout = result ? layoutTagLabels(result.nodes ?? []) : new Map()

  // NEW (docs/phase5_design.md follow-up): full-sheet-resolution PNG export, independent
  // of the current pan/zoom viewport. Draws the same layers the user sees on screen,
  // via the exact same shared geometry helpers, plus a burned-in legend + metrics footer
  // so the honest caveat travels with the image wherever it's shared.
  useImperativeHandle(ref, () => ({
    exportOverlayPng(sheetId) {
      if (!natural || !result || !imgRef.current) return
      const canvas = document.createElement('canvas')
      canvas.width = natural.w
      canvas.height = natural.h + FOOTER_HEIGHT
      const ctx = canvas.getContext('2d')

      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(imgRef.current, 0, 0, natural.w, natural.h)

      if (showEdges) {
        for (const edge of result.edges ?? []) {
          const pts = edgePoints(edge, nodesById)
          if (!pts) continue
          const hasPath = edgeHasPath(edge)
          ctx.beginPath()
          ctx.moveTo(pts[0][0], pts[0][1])
          for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y)
          ctx.strokeStyle = '#888888'
          ctx.globalAlpha = hasPath ? 0.55 : 0.35
          ctx.lineWidth = 1.4
          ctx.setLineDash(hasPath ? [] : [2, 4])
          ctx.stroke()
        }
        ctx.globalAlpha = 1
        ctx.setLineDash([])
      }

      for (const node of result.nodes ?? []) {
        const style = NODE_STYLES[node.node_type] ?? DEFAULT_NODE_STYLE
        const [x1, y1, x2, y2] = node.bbox
        ctx.strokeStyle = style.stroke
        ctx.fillStyle = style.fill
        ctx.lineWidth = 2.2
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      }
      ctx.font = '13px sans-serif'
      ctx.textBaseline = 'alphabetic'
      for (const [, label] of labelLayout) {
        ctx.lineWidth = 3
        ctx.strokeStyle = '#ffffff'
        ctx.fillStyle = label.confident ? '#1a1a1a' : '#9a9a9a'
        if (!label.confident) ctx.font = 'italic 13px sans-serif'
        ctx.strokeText(label.text, label.x, label.y)
        ctx.fillText(label.text, label.x, label.y)
        ctx.font = '13px sans-serif'
      }

      drawFooter(ctx, natural.w, natural.h, result.metrics, showEdges)

      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `pidetect_${sheetId}_overlay.png`
        a.click()
        URL.revokeObjectURL(url)
      }, 'image/png')
    },
  }))

  return (
    <div className="overlay-wrap">
      <div className="overlay-toolbar">
        <button onClick={() => setView({ scale: 1, tx: 0, ty: 0 })}>Reset view</button>
        <span className="hint">scroll to zoom · drag to pan</span>
      </div>
      <div
        className="overlay-viewport"
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <div
          className="overlay-canvas"
          style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})` }}
        >
          <img ref={imgRef} src={imageUrl} alt="uploaded P&ID sheet" onLoad={onImgLoad} draggable={false} crossOrigin="anonymous" />
          {natural && result && (
            <svg
              className="overlay-svg"
              viewBox={`0 0 ${natural.w} ${natural.h}`}
              width={natural.w}
              height={natural.h}
            >
              {/* SECONDARY layer: connectivity edges. Edges WITH a traced path get the
                  regular muted-solid style; edges WITHOUT one (short-gap ink-inference
                  AND contracted-through connector/crossing edges alike -- see FIX 2
                  report, 81.7% of edges on a measured sheet) are dotted and fainter --
                  a straight chord asserting a route that was never actually traced. */}
              {showEdges && (result.edges ?? []).map((edge, i) => {
                const pts = edgePoints(edge, nodesById)
                if (!pts) return null
                const hasPath = edgeHasPath(edge)
                return (
                  <polyline
                    key={`e${i}`}
                    points={pts.map(([x, y]) => `${x},${y}`).join(' ')}
                    fill="none"
                    stroke="#888"
                    strokeOpacity={hasPath ? 0.55 : 0.35}
                    strokeWidth={1.4}
                    strokeDasharray={hasPath ? undefined : '2,4'}
                  />
                )
              })}

              {/* PRIMARY layer: detections, confidently styled. Clickable -- opens the
                  node inspector panel (App.jsx) via onNodeClick. pointerEvents is set
                  per-rect rather than on the whole <svg> (which stays 'none' via CSS) so
                  pan/drag on empty canvas is untouched. */}
              {(result.nodes ?? []).map((node) => {
                const style = NODE_STYLES[node.node_type] ?? DEFAULT_NODE_STYLE
                const [x1, y1, x2, y2] = node.bbox
                return (
                  <rect
                    key={node.id}
                    className="node-box"
                    x={x1} y={y1} width={x2 - x1} height={y2 - y1}
                    stroke={style.stroke} fill={style.fill} strokeWidth={2.2}
                    style={{ pointerEvents: 'auto' }}
                    onClick={(e) => { e.stopPropagation(); onNodeClick?.(node.id) }}
                  />
                )
              })}

              {/* Selection highlight -- drawn last so it always sits on top. */}
              {selectedNodeId && nodesById.has(selectedNodeId) && (() => {
                const [x1, y1, x2, y2] = nodesById.get(selectedNodeId).bbox
                const pad = 4
                return (
                  <rect
                    x={x1 - pad} y={y1 - pad}
                    width={(x2 - x1) + pad * 2} height={(y2 - y1) + pad * 2}
                    fill="none" stroke="#ffcc00" strokeWidth={3.5} strokeDasharray="6,3"
                  />
                )
              })()}

              {/* Tag labels, laid out with collision avoidance (FIX 1) -- rendered as
                  a separate pass so a label can never be visually clipped by a
                  neighbour's box drawn after it. */}
              {[...labelLayout.entries()].map(([id, label]) => (
                <text
                  key={id}
                  x={label.x} y={label.y}
                  fontSize={13}
                  fontStyle={label.confident ? 'normal' : 'italic'}
                  fill={label.confident ? '#1a1a1a' : '#9a9a9a'}
                  style={{ paintOrder: 'stroke', stroke: '#fff', strokeWidth: 3 }}
                >
                  {label.text}
                </text>
              ))}
            </svg>
          )}
        </div>
      </div>
    </div>
  )
})

function drawFooter(ctx, w, imgH, metrics, showEdges) {
  const y0 = imgH
  ctx.fillStyle = '#fafafa'
  ctx.fillRect(0, y0, w, FOOTER_HEIGHT)
  ctx.strokeStyle = '#ddd'
  ctx.beginPath()
  ctx.moveTo(0, y0)
  ctx.lineTo(w, y0)
  ctx.stroke()

  ctx.textBaseline = 'middle'
  ctx.font = '13px sans-serif'
  const legend = [
    ['valve', '#3c78dc'], ['instrument', '#2fa050'], ['unknown_fitting', '#dc8214'],
    ['flow_arrow', '#b432b4'], ['off_page', '#dc1e1e'],
  ]
  let x = 16
  const swY = y0 + 22
  for (const [label, color] of legend) {
    ctx.fillStyle = color
    ctx.fillRect(x, swY - 7, 14, 14)
    ctx.fillStyle = '#1a1a1a'
    ctx.fillText(label, x + 20, swY)
    x += 20 + ctx.measureText(label).width + 24
  }
  if (showEdges) {
    ctx.strokeStyle = '#888'
    ctx.setLineDash([2, 4])
    ctx.beginPath()
    ctx.moveTo(x, swY)
    ctx.lineTo(x + 26, swY)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = '#1a1a1a'
    ctx.fillText('inferred connection — route not traced', x + 34, swY)
  }

  const capY = y0 + 55
  ctx.font = 'bold 15px sans-serif'
  ctx.fillStyle = '#1a1a1a'
  const m = metrics || {}
  const caption =
    `${m.detection_node_match_pct ?? '?'}% node match · ` +
    `${m.tag_exact_match_ok_rows_pct ?? '?'}% tag exact-match · ` +
    `connectivity F1 ${m.connectivity_f1 ?? '?'} — experimental, below the ${m.connectivity_gate_pct ?? '0.70'} gate`
  ctx.fillText(caption, 16, capY)
  ctx.font = '11px sans-serif'
  ctx.fillStyle = '#666'
  ctx.fillText('docs/phase4_final.md · docs/phase3_results.md — frozen 3-sheet OPEN100 eval, not a live score for this upload', 16, capY + 22)
}

export default Overlay
