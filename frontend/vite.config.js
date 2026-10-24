import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-server proxy target for the FastAPI backend (match the backend's PORT).
const apiTarget = process.env.API_PROXY_TARGET || 'http://localhost:8001'

// Long-lived vendor chunks. Charts and markdown are also lazy-loaded
// (StockChartPlot.jsx / Markdown.jsx), so they stay off the critical path.
const VENDOR_CHUNKS = [
  ['charts', /[\\/]node_modules[\\/](recharts|d3-[^\\/]+|victory-vendor|internmap|decimal\.js-light|es-toolkit|@reduxjs|redux|immer|reselect|react-redux|use-sync-external-store)[\\/]/],
  ['markdown', /[\\/]node_modules[\\/](react-markdown|remark-[^\\/]+|rehype-[^\\/]+|unified|micromark[^\\/]*|mdast-[^\\/]+|hast-[^\\/]+|unist-[^\\/]+|vfile[^\\/]*|property-information|space-separated-tokens|comma-separated-tokens|html-url-attributes|decode-named-character-reference|character-entities[^\\/]*|trim-lines|devlop|bail|trough|is-plain-obj|zwitch|longest-streak|markdown-table|ccount|escape-string-regexp|estree-util-[^\\/]+|style-to-[^\\/]+|inline-style-parser|@ungap)[\\/]/],
  ['react', /[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/],
]

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          for (const [name, pattern] of VENDOR_CHUNKS) if (pattern.test(id)) return name
        },
      },
    },
  },
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
