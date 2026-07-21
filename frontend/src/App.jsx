import { useEffect, useRef, useState } from 'react'
import Overlay from './Overlay'
import NodePanel from './NodePanel'
import { createJob, getJobStatus, getJobResult, jobImageUrl, jobResultDownloadUrl } from './api'
import { UNKNOWN_FITTING_NOTE } from './overlayGeometry'
import './App.css'

const STAGE_LABELS = {
  queued: 'Queued…',
  detecting: 'Detecting symbols (SAHI, slice=320)…',
  reading_tags: 'Reading instrument tags…',
  tracing_lines: 'Tracing pipe lines…',
  building_graph: 'Building connectivity graph…',
  done: 'Done',
}

function stageText(stage, progress) {
  if (stage === 'reading_tags' && progress) {
    return `Reading instrument tags… (${progress.current} of ${progress.total})`
  }
  return STAGE_LABELS[stage] ?? stage
}

const NODE_LEGEND = [
  ['valve', '#3c78dc'],
  ['instrument', '#2fa050'],
  ['unknown_fitting', '#dc8214'],
  ['flow_arrow', '#b432b4'],
  ['off_page', '#dc1e1e'],
]

export default function App() {
  const overlayRef = useRef(null)
  const [file, setFile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)   // 'queued' | 'running' | 'done' | 'failed'
  const [stage, setStage] = useState(null)
  const [progress, setProgress] = useState(null)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [showEdges, setShowEdges] = useState(true)
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const pollRef = useRef(null)

  useEffect(() => () => clearInterval(pollRef.current), [])

  const startUpload = async () => {
    if (!file) return
    setError(null)
    setResult(null)
    setSelectedNodeId(null)
    setStatus('queued')
    const { job_id } = await createJob(file)
    setJobId(job_id)
    pollRef.current = setInterval(async () => {
      try {
        const s = await getJobStatus(job_id)
        setStatus(s.status)
        setStage(s.stage)
        setProgress(s.progress)
        if (s.status === 'done') {
          clearInterval(pollRef.current)
          const r = await getJobResult(job_id)
          setResult(r)
        } else if (s.status === 'failed') {
          clearInterval(pollRef.current)
          setError(s.error)
        }
      } catch (e) {
        clearInterval(pollRef.current)
        setError(String(e))
      }
    }, 2000)
  }

  const reset = () => {
    setFile(null); setJobId(null); setStatus(null); setStage(null)
    setProgress(null); setError(null); setResult(null); setSelectedNodeId(null)
  }

  const showResult = status === 'done' && result
  const selectedNode = showResult
    ? (result.nodes ?? []).find((n) => n.id === selectedNodeId) ?? null
    : null

  return (
    <div className="app">
      <header className="app-header">
        <h1>PIDetect</h1>
        <p>Upload a P&ID sheet → detected symbols, instrument tags, connectivity graph.</p>
      </header>

      {!showResult && (
        <section className="upload-panel">
          <input
            type="file"
            accept="image/*"
            disabled={status === 'queued' || status === 'running'}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button
            disabled={!file || status === 'queued' || status === 'running'}
            onClick={startUpload}
          >
            Analyze sheet
          </button>

          {status && status !== 'failed' && (
            <div className="progress-block">
              <div className="progress-stage">{stageText(stage, progress)}</div>
              {stage === 'reading_tags' && progress && progress.total > 0 && (
                <div className="progress-bar">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${(100 * progress.current) / progress.total}%` }}
                  />
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-block">
              <strong>Job failed.</strong>
              <pre>{error}</pre>
            </div>
          )}
        </section>
      )}

      {showResult && (
        <section className="result-panel">
          <div className="result-toolbar">
            <button onClick={reset}>Analyze another sheet</button>
            <a
              className="download-btn"
              href={jobResultDownloadUrl(jobId)}
              download={`pidetect_${jobId}.json`}
            >
              Download JSON
            </a>
            <button
              className="download-btn"
              onClick={() => overlayRef.current?.exportOverlayPng(jobId)}
            >
              Download overlay (PNG)
            </button>
          </div>

          <MetricsBanner metrics={result.metrics} />

          <div className="legend">
            <span className="legend-title">Symbols (primary — 98.8% node match):</span>
            {NODE_LEGEND.map(([label, color]) => (
              <span
                key={label}
                className="legend-item"
                title={label === 'unknown_fitting' ? UNKNOWN_FITTING_NOTE : undefined}
              >
                <span className="swatch" style={{ background: color }} />
                {label}
              </span>
            ))}
            <label className="edge-toggle">
              <input
                type="checkbox"
                checked={showEdges}
                onChange={(e) => setShowEdges(e.target.checked)}
              />
              Show connectivity edges — <em>experimental, F1 {result.metrics?.connectivity_f1}, precision {result.metrics?.connectivity_precision}</em>
            </label>
          </div>
          {showEdges && (
            <div className="legend edge-legend">
              <span className="legend-item"><span className="line-swatch line-solid" /> traced route</span>
              <span className="legend-item"><span className="line-swatch line-dotted" /> inferred connection — route not traced</span>
            </div>
          )}
          <p className="legend-note">
            <strong>unknown_fitting</strong> — {UNKNOWN_FITTING_NOTE} Click any box for details.
          </p>

          <div className="workspace">
            <Overlay
              ref={overlayRef}
              imageUrl={jobImageUrl(jobId)}
              result={result}
              showEdges={showEdges}
              selectedNodeId={selectedNodeId}
              onNodeClick={setSelectedNodeId}
            />
            <NodePanel node={selectedNode} onClose={() => setSelectedNodeId(null)} />
          </div>
        </section>
      )}
    </div>
  )
}

function MetricsBanner({ metrics }) {
  if (!metrics) return null
  return (
    <div className="metrics-banner">
      <div><strong>{metrics.detection_node_match_pct}%</strong> symbol detection node-match</div>
      <div><strong>{metrics.tag_exact_match_ok_rows_pct}%</strong> tag exact-match (ok rows), CER {metrics.tag_cer}</div>
      <div className="metrics-experimental">
        <strong>F1 {metrics.connectivity_f1}</strong> connectivity (precision {metrics.connectivity_precision}, recall {metrics.connectivity_recall}) — below the {metrics.connectivity_gate_pct} gate, experimental
      </div>
      <div className="metrics-note">{metrics.note}</div>
    </div>
  )
}
