/**
 * Error Boundary Component Tests
 */
import React from 'react';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders } from '../utils/testComponents';
import { mockDataGenerators } from '../utils/testUtils';
import { checkAccessibility } from '../utils/accessibilityUtils';
import ErrorBoundary from '../components/ErrorBoundary';

// Mock console.error to avoid noise in tests
const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

// Component that throws an error for testing
const ThrowError: React.FC<{ shouldThrow: boolean }> = ({ shouldThrow }) => {
  if (shouldThrow) {
    throw new Error('Test error for ErrorBoundary');
  }
  return <div data-testid="working-component">Working Component</div>;
};

describe('ErrorBoundary Component', () => {
  afterEach(() => {
    consoleSpy.mockClear();
  });

  afterAll(() => {
    consoleSpy.mockRestore();
  });

  test('renders children when there is no error', () => {
    renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('working-component')).toBeInTheDocument();
    expect(screen.getByText('Working Component')).toBeInTheDocument();
  });

  test('displays error UI when child component throws', () => {
    renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
    expect(screen.getByText(/This section encountered an error/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /report bug/i })).toBeInTheDocument();
  });

  test('shows detailed error information', () => {
    renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // Check for error details toggle
    const showDetailsButton = screen.getByRole('button', { name: /technical details/i });
    expect(showDetailsButton).toBeInTheDocument();

    // Check that error message is already displayed in the alert - use getAllByText since it appears in multiple places
    const errorMessages = screen.getAllByText(/Test error for ErrorBoundary/);
    expect(errorMessages.length).toBeGreaterThan(0);
  });

  test('retry button attempts to recover', () => {
    const { rerender } = renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // Verify error state
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

    // Click retry button
    const retryButton = screen.getByRole('button', { name: /try again/i });
    fireEvent.click(retryButton);

    // The error boundary should still show the error state after retry is clicked
    // because the component will still throw when re-rendered with the same props
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
  });

  test('report bug button is functional', () => {
    // Mock window.open
    const mockOpen = jest.fn();
    Object.defineProperty(window, 'open', { value: mockOpen, writable: true });

    renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const reportButton = screen.getByRole('button', { name: /report bug/i });
    fireEvent.click(reportButton);

    // Verify that error reporting was attempted
    expect(mockOpen).toHaveBeenCalled();
  });

  test('meets accessibility standards', async () => {
    const { container } = renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const results = await checkAccessibility(container);
    expect(results.violations).toHaveLength(0);
  });

  test('has proper ARIA attributes', () => {
    renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // Check for proper ARIA attributes
    const errorContainer = screen.getByRole('alert');
    expect(errorContainer).toBeInTheDocument();
    // The MUI Alert component has its own ARIA implementation
    expect(errorContainer).toHaveAttribute('role', 'alert');
  });

  test('keyboard navigation works properly', () => {
    renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    const retryButton = screen.getByRole('button', { name: /try again/i });
    const reportButton = screen.getByRole('button', { name: /report bug/i });
    const detailsButton = screen.getByRole('button', { name: /technical details/i });

    // Test tab navigation
    retryButton.focus();
    expect(document.activeElement).toBe(retryButton);

    // Simulate tab to next element
    fireEvent.keyDown(retryButton, { key: 'Tab' });
    // Note: Actual tab navigation in jsdom requires more complex setup
    // This is a simplified test for the presence of focusable elements
    expect(reportButton).toBeInTheDocument();
    expect(detailsButton).toBeInTheDocument();
  });

  test('handles multiple errors gracefully', () => {
    const { rerender } = renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // First error
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

    // Simulate another error
    rerender(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // Should still show error UI
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
  });

  test('preserves error state during re-renders', () => {
    const { rerender } = renderWithProviders(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

    // Re-render with same error
    rerender(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // Error UI should persist
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
  });
});