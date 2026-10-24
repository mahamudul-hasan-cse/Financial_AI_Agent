// remark plugin: wrap signed percentages ("+1.2%", "-0.8 %", "−3%") in
// <span class="pct pct-up|pct-down"> so gains/losses render green/red.
// A sign right after a word/digit ("5-10%") is a range, not a change.
export const SIGNED_PCT_RE = /(?<![\w.%])([+\-−])\s?(\d[\d,]*(?:\.\d+)?)\s?%/g

function splitText(value) {
  const parts = []
  let last = 0
  for (const match of value.matchAll(SIGNED_PCT_RE)) {
    if (match.index > last) parts.push({ type: 'text', value: value.slice(last, match.index) })
    const up = match[1] === '+'
    parts.push({
      // Any phrasing node works; data.hName/hProperties turn it into a span.
      type: 'emphasis',
      data: { hName: 'span', hProperties: { className: ['pct', up ? 'pct-up' : 'pct-down'] } },
      children: [{ type: 'text', value: match[0] }],
    })
    last = match.index + match[0].length
  }
  if (!parts.length) return null
  if (last < value.length) parts.push({ type: 'text', value: value.slice(last) })
  return parts
}

function visit(node) {
  if (!node.children) return
  const next = []
  for (const child of node.children) {
    if (child.type === 'text') {
      const parts = splitText(child.value)
      if (parts) { next.push(...parts); continue }
    } else if (child.type !== 'code' && child.type !== 'inlineCode') {
      visit(child)
    }
    next.push(child)
  }
  node.children = next
}

export default function remarkSignedPercent() {
  return tree => visit(tree)
}
