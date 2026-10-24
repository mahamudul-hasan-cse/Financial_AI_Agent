import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ResearchCard from '../ResearchCard'

describe('ResearchCard', () => {
  it('renders the assistant answer as markdown', async () => {
    render(<ResearchCard content={'Google is trading near **$173**.\n\n- Cloud momentum'} />)

    expect(screen.getByLabelText('Assistant response')).toBeInTheDocument()
    expect((await screen.findByText('$173')).tagName).toBe('STRONG')
    expect(screen.getByText('Cloud momentum').tagName).toBe('LI')
  })

  it('shows the plain text immediately while markdown loads', async () => {
    vi.resetModules() // fresh lazy() so the markdown chunk is not yet resolved
    const { default: FreshResearchCard } = await import('../ResearchCard')
    const { container } = render(<FreshResearchCard content={'Plain **answer**'} />)
    expect(container.querySelector('.markdown-fallback')?.textContent).toBe('Plain **answer**')
  })

  it('renders nothing when there is no content and no charts', () => {
    const { container } = render(<ResearchCard content="" data={{ artifacts: [] }} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows chart artifacts and strips standalone .png lines from the text', () => {
    render(
      <ResearchCard
        content={'Here is the chart.\naapl_chart.png'}
        data={{ artifacts: ['msft_chart.png', 'report.xlsx'] }}
      />,
    )

    expect(screen.queryByText(/aapl_chart\.png/)).not.toBeInTheDocument()
    const sources = screen.getAllByAltText('Generated chart').map(img => img.getAttribute('src'))
    expect(sources).toEqual(['/outputs/charts/msft_chart.png', '/outputs/charts/aapl_chart.png'])
  })
})
