// Loader for the lazily-split markdown renderer (see Markdown.jsx).
export const loadMarkdown = () => import('./Markdown.jsx')

// Start fetching the markdown chunk early (e.g. on app mount) so the first
// answer doesn't wait for it.
export function preloadMarkdown() {
  return loadMarkdown().catch(() => {})
}
