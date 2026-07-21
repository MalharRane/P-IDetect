import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Proxy API calls to the local FastAPI backend (docs/phase5_design.md) so the
// frontend can use plain relative fetch('/jobs') paths with no CORS config needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/jobs': 'http://127.0.0.1:8000',
    },
  },
})
