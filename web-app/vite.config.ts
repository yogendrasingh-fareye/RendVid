import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Built into the Python package; FastAPI serves index.html at / and the bundle at /assets.
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../src/rendvid/web/app', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8000', '/outputs': 'http://127.0.0.1:8000', '/edit': 'http://127.0.0.1:8000', '/static': 'http://127.0.0.1:8000' } },
})
