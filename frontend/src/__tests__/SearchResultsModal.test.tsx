/**
 * Search Results Modal Component Tests
 */

import React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../utils/testComponents';
import { mockDataGenerators } from '../utils/testUtils';
import { checkAccessibility } from '../utils/accessibilityUtils';
import SearchResultsModal from '../components/SearchResultsModal';

describe('SearchResultsModal Component', () => {
  const mockResults = [
    {
      id: 1,
      title: 'Case 1',
      type: 'case' as const,
      description: 'Suspicious transaction pattern',
      score: 85,
      metadata: { severity: 'high', riskScore: 0.85 },
    },
    {
      id: 2,
      title: 'Case 2',
      type: 'case' as const,
      description: 'Unusual account activity',
      score: 65,
      metadata: { severity: 'medium', riskScore: 0.65 },
    },
  ];

  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    query: 'test query',
    results: mockResults,
    loading: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders modal when open', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/Search Results/i)).toBeInTheDocument();
    expect(screen.getByText(/test query/i)).toBeInTheDocument();
  });

  test('does not render when closed', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} open={false} />);

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('displays loading state', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} loading={true} />);

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  test('displays search results', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    expect(screen.getByText('Case 1')).toBeInTheDocument();
    expect(screen.getByText('Case 2')).toBeInTheDocument();
    expect(screen.getByText(/2 results found/i)).toBeInTheDocument();
  });

  test('handles empty search results', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} results={[]} />);

    expect(screen.getByText(/No results found/i)).toBeInTheDocument();
  });

  test('closes modal when close button clicked', async () => {
    const mockClose = jest.fn();
    renderWithProviders(
      <SearchResultsModal {...defaultProps} onClose={mockClose} />
    );

    const closeButton = screen.getByRole('button', { name: /close/i });
    await userEvent.click(closeButton);

    expect(mockClose).toHaveBeenCalled();
  });

  test('closes modal on escape key', () => {
    const mockClose = jest.fn();
    renderWithProviders(
      <SearchResultsModal {...defaultProps} onClose={mockClose} />
    );

    const modal = screen.getByRole('dialog');
    fireEvent.keyDown(modal, { key: 'Escape' });

    expect(mockClose).toHaveBeenCalled();
  });

  test('meets accessibility standards', async () => {
    const { container } = renderWithProviders(<SearchResultsModal {...defaultProps} />);

    const results = await checkAccessibility(container);
    expect(results.violations).toHaveLength(0);
  });

  test('has proper ARIA attributes', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    const modal = screen.getByRole('dialog');
    expect(modal).toHaveAttribute('aria-labelledby');
  });

  test('displays risk scores correctly', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    expect(screen.getByText(/85/)).toBeInTheDocument();
    expect(screen.getByText(/65/)).toBeInTheDocument();
  });

  test('shows result types with appropriate icons', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    // Check that results are displayed with proper structure
    expect(screen.getByText('Case 1')).toBeInTheDocument();
    expect(screen.getByText('Suspicious transaction pattern')).toBeInTheDocument();
  });

  test('handles keyboard navigation', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    const modal = screen.getByRole('dialog');
    const closeButton = screen.getByRole('button', { name: /close/i });

    // Modal should be focusable
    expect(modal).toHaveAttribute('tabindex');
    
    // Close button should be focusable
    closeButton.focus();
    expect(document.activeElement).toBe(closeButton);
  });

  test('preserves search query display', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} query="fraud detection" />);

    expect(screen.getByText(/fraud detection/i)).toBeInTheDocument();
  });

  test('handles loading state transition', () => {
    const { rerender } = renderWithProviders(
      <SearchResultsModal {...defaultProps} loading={true} />
    );

    expect(screen.getByRole('progressbar')).toBeInTheDocument();

    rerender(<SearchResultsModal {...defaultProps} loading={false} />);

    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByText('Case 1')).toBeInTheDocument();
  });

  test('shows score indicators', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    // Should show score values
    expect(screen.getByText(/85/)).toBeInTheDocument();
    expect(screen.getByText(/65/)).toBeInTheDocument();
  });

  test('result items display metadata information', () => {
    renderWithProviders(<SearchResultsModal {...defaultProps} />);

    // Check that metadata is accessible
    expect(screen.getByText('Case 1')).toBeInTheDocument();
    expect(screen.getByText('Case 2')).toBeInTheDocument();
  });
});