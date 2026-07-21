// Thin fetch wrappers around the Phase 5 FastAPI backend (docs/phase5_design.md).
// Relative paths -- vite.config.js proxies /jobs to the local backend, so no CORS
// setup is needed for local dev.

export async function createJob(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/jobs', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`upload failed: ${res.status}`)
  return res.json() // { job_id }
}

export async function getJobStatus(jobId) {
  const res = await fetch(`/jobs/${jobId}`)
  if (!res.ok) throw new Error(`status check failed: ${res.status}`)
  return res.json() // { job_id, status, stage, progress, error? }
}

export async function getJobResult(jobId) {
  const res = await fetch(`/jobs/${jobId}/result`)
  if (!res.ok) throw new Error(`result fetch failed: ${res.status}`)
  return res.json()
}

export function jobImageUrl(jobId) {
  return `/jobs/${jobId}/image`
}

export function jobResultDownloadUrl(jobId) {
  return `/jobs/${jobId}/result`
}
