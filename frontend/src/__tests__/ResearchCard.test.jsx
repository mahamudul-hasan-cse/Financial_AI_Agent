import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ResearchCard from '../ResearchCard'

describe('ResearchCard', () => {
  it('renders the assistant answer as markdown', () => {
    render(<ResearchCard content={'Google is trading near **$173**.\n\n- Cloud momentum'} />)

    expect(screen.getByLabelText('Assistant response')).toBeInTheDocument()
    expect(screen.getByText('$173').tagName).toBe('STRONG')
    expect(screen.getByText('Cloud momentum').tagName).toBe('LI')
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
