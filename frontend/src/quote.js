// Usable quote = real price from the backend; never fills gaps with guesses.
export function hasQuote(quote) {
  return !!quote && typeof quote.price === 'number' && Number.isFinite(quote.price) && !!quote.symbol
}

// Unique ticker symbols detected by the NLP pipeline for a message.
export function tickerCount(nlpMeta) {
  const entities = Array.isArray(nlpMeta?.entities) ? nlpMeta.entities : []
  return new Set(entities.map((e) => e?.ticker).filter(Boolean)).size
}

// Show the single-stock card whenever we have a real quote, unless the answer
// is a genuine multi-ticker comparison.
export function shouldShowQuote(structuredResponse, nlpMeta) {
  const quote = structuredResponse?.quote
  if (!hasQuote(quote)) return false
  if (structuredResponse?.intent !== 'comparison') return true
  return tickerCount(nlpMeta) <= 1
}
