import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-server proxy target for the FastAPI backend (match the backend's PORT).
const apiTarget = process.env.API_PROXY_TARGET || 'http://localhost:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': apiTarget,
      '/outputs': apiTarget,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.js'],
    globals: true,
  },
})
