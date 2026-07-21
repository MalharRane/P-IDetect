// Shared node/edge/label styling + layout, used by both the on-screen SVG overlay
// (Overlay.jsx) and the full-resolution PNG export (exportOverlayPng in Overlay.jsx),
// so the two never visually drift apart -- one set of rules, two renderers.

export const NODE_STYLES = {
  valve:           { stroke: '#3c78dc', fill: 'rgba(60,120,220,0.12)' },
  instrument:      { stroke: '#2fa050', fill: 'rgba(47,160,80,0.12)' },
  unknown_fitting: { stroke: '#dc8214', fill: 'rgba(220,130,20,0.12)' },
  flow_arrow:      { stroke: '#b432b4', fill: 'rgba(180,50,180,0.12)' },
  off_page:        { stroke: '#dc1e1e', fill: 'rgba(220,30,30,0.10)' },
}
export const DEFAULT_NODE_STYLE = { stroke: '#888', fill: 'rgba(136,136,136,0.1)' }

// Reads validated by a regex in pidetect.text.ocr (docs/phase3_design.md). ok_placeholder
// is a CONFIRMED-correct read of a literal "XXX" callout (Phase 3 eval: 100% exact-match
// on these rows) -- it belongs with "ok", not with genuine parse failures.
export const CONFIDENT_TAG_STATUSES = new Set(['ok', 'ok_placeholder'])

// Shared caveat copy for the node inspector panel (App.jsx) and legend note (App.jsx) --
// one set of strings, so the two never drift apart. Phase 2 (fine-grained classifier) was
// never built: only the valve/instrument SUPERCATEGORY is measured (97.6%/99.4% CtrMt@50%
// on OPEN100, docs/phase1_8_final.md), never the cls_name subtype underneath it.
export const SUBTYPE_UNVALIDATED_NOTE =
  'Class subtype — unvalidated on real sheets; only valve/instrument supercategory detection ' +
  'is measured (97.6% / 99.4% center-match).'

export const UNKNOWN_FITTING_NOTE =
  'unknown_fitting = outside the scored valve/instrument supercategories (reducers, ' +
  'strainers, blinds, nozzles) — included as a graph node because it affects pipe ' +
  'reachability, not because it is unrecognized.'

export const PLACEHOLDER_CLASS_RE = /^Symbol_\d+$/
export const PLACEHOLDER_CLASS_NOTE =
  'Symbol_N placeholder — the glyph identity was not confidently resolvable from the source dataset.'

export function nodeCenter(node) {
  const [x1, y1, x2, y2] = node.bbox
  return [(x1 + x2) / 2, (y1 + y2) / 2]
}

// FIX 2 (docs/phase5_design.md follow-up): the single source of truth for "do we know
// the real pipe route, or are we drawing an inferred straight chord". 81.7% of edges on
// a measured OPEN100 sheet fall in the second bucket (short-gap ink-inferred connections
// AND contracted-through connector/crossing edges alike -- see report) -- both must read
// as "inferred", never with the same visual confidence as a traced route.
export function edgeHasPath(edge) {
  return Array.isArray(edge.path) && edge.path.length >= 2
}

export function edgePoints(edge, nodesById) {
  if (edgeHasPath(edge)) return edge.path
  const u = nodesById.get(edge.u)
  const v = nodesById.get(edge.v)
  if (!u || !v) return null
  return [nodeCenter(u), nodeCenter(v)]
}

const LABEL_HEIGHT = 15
const CHAR_WIDTH = 7.2 // rough average glyph width at fontSize 13, sans-serif

// FIX 1: greedy vertical-stacking collision avoidance for instrument tag labels.
// Adjacent bubbles (dense TE/PT clusters) each want to place their label just above
// their own box at the same height -- with overlapping x-ranges that runs the text
// together illegibly. Process left-to-right; when a candidate position's estimated
// text bbox would overlap an already-placed label, push it up one label-height and
// retry. A handful of retries is enough for the dense clusters this was measured
// against (docs/phase5_design.md); this deliberately does NOT hide labels at any zoom
// level -- offsetting alone resolved every reported collision.
export function layoutTagLabels(nodes) {
  const candidates = nodes
    .filter((n) => n.tag)
    .map((n) => {
      const confident = CONFIDENT_TAG_STATUSES.has(n.tag.parse_status)
      const text = confident
        ? `${n.tag.function ?? ''} ${n.tag.loop_number ?? ''}`.trim()
        : `? ${n.tag.raw_text || '(unread)'}`
      return {
        id: n.id,
        x: n.bbox[0],
        baseY: n.bbox[1] - 6,
        text,
        confident,
        width: text.length * CHAR_WIDTH,
      }
    })
    .sort((a, b) => a.x - b.x)

  const placed = []
  const layout = new Map()
  for (const c of candidates) {
    let y = c.baseY
    let tries = 0
    while (tries < 8) {
      const collides = placed.some(
        (p) => c.x < p.x + p.width && c.x + c.width > p.x && Math.abs(y - p.y) < LABEL_HEIGHT
      )
      if (!collides) break
      y -= LABEL_HEIGHT
      tries += 1
    }
    placed.push({ x: c.x, width: c.width, y })
    layout.set(c.id, { x: c.x, y, text: c.text, confident: c.confident })
  }
  return layout
}
