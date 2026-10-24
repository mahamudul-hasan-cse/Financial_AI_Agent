// Markdown renderer, split into its own chunk (react-markdown + remark-gfm
// pull in the whole unified/micromark stack). Loaded lazily by ResearchCard.
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkSignedPercent from './signedPercent.js'

const REMARK_PLUGINS = [remarkGfm, remarkSignedPercent]

export default function Markdown({ children }) {
  return <ReactMarkdown remarkPlugins={REMARK_PLUGINS}>{children}</ReactMarkdown>
}
