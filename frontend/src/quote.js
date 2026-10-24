// Usable quote = real price from the backend; never fills gaps with guesses.
export function hasQuote(quote) {
  return !!quote && typeof quote.price === 'number' && Number.isFinite(quote.price) && !!quote.symbol
}
