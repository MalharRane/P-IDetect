import {
  CONFIDENT_TAG_STATUSES,
  PLACEHOLDER_CLASS_RE,
  PLACEHOLDER_CLASS_NOTE,
  SUBTYPE_UNVALIDATED_NOTE,
  UNKNOWN_FITTING_NOTE,
} from './overlayGeometry'

// cls_name is a finer-grained subtype than node_type for these three categories --
// see CLAUDE.md Phase 2 (fine-grained classifier, never built) and the README notes
// this panel echoes: only the valve/instrument SUPERCATEGORY is measured, and
// unknown_fitting is a scope label, not a confidence signal.
function subtypeCaveats(node) {
  const notes = []
  if (node.node_type === 'valve' || node.node_type === 'instrument') {
    notes.push(SUBTYPE_UNVALIDATED_NOTE)
  } else if (node.node_type === 'unknown_fitting') {
    notes.push(UNKNOWN_FITTING_NOTE)
  }
  if (PLACEHOLDER_CLASS_RE.test(node.cls_name ?? '')) {
    notes.push(PLACEHOLDER_CLASS_NOTE)
  }
  return notes
}

function fmtConf(c) {
  return typeof c === 'number' ? `${(c * 100).toFixed(1)}%` : '—'
}

export default function NodePanel({ node, onClose }) {
  if (!node) {
    return (
      <aside className="node-panel node-panel-empty">
        <p>Click a symbol box on the sheet to inspect it here — class, detection confidence,
          bbox, and (for instruments) the parsed tag.</p>
      </aside>
    )
  }

  const [x1, y1, x2, y2] = node.bbox
  const caveats = subtypeCaveats(node)
  const tag = node.tag
  const tagConfident = tag && CONFIDENT_TAG_STATUSES.has(tag.parse_status)

  return (
    <aside className="node-panel">
      <div className="node-panel-header">
        <h3>{node.id}</h3>
        <button className="node-panel-close" onClick={onClose} aria-label="Close">×</button>
      </div>

      <dl className="node-panel-fields">
        <dt>node_type</dt>
        <dd>{node.node_type}</dd>
        <dt>cls_name</dt>
        <dd>{node.cls_name ?? '—'}</dd>
        <dt>confidence</dt>
        <dd>{fmtConf(node.conf)}</dd>
        <dt>bbox</dt>
        <dd>[{x1.toFixed(0)}, {y1.toFixed(0)}, {x2.toFixed(0)}, {y2.toFixed(0)}]</dd>
      </dl>

      {caveats.map((note) => (
        <p key={note} className="node-panel-caveat">{note}</p>
      ))}

      {tag && (
        <>
          <h4 className="node-panel-subhead">Instrument tag</h4>
          <dl className="node-panel-fields">
            <dt>function</dt>
            <dd>{tag.function ?? '—'}</dd>
            <dt>loop_number</dt>
            <dd>{tag.loop_number ?? '—'}</dd>
            <dt>raw_text</dt>
            <dd>{tag.raw_text ?? '—'}</dd>
            <dt>confidence</dt>
            <dd>{fmtConf(tag.confidence)}</dd>
            <dt>parse_status</dt>
            <dd className={tagConfident ? undefined : 'node-panel-warn'}>{tag.parse_status}</dd>
          </dl>
          {!tagConfident && (
            <p className="node-panel-caveat">
              OCR read did not parse cleanly into function/loop_number — raw_text is the only
              trustworthy field here.
            </p>
          )}
        </>
      )}
    </aside>
  )
}
