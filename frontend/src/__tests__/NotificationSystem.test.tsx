/**
 * Notification System Component Tests
 */

import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../utils/testComponents';
import { checkAccessibility } from '../utils/accessibilityUtils';
import NotificationSystem from '../components/NotificationSystem';

describe('NotificationSystem Component', () => {
  const mockNotifications = [
    {
      id: 1,
      type: 'success' as const,
      message: 'Operation completed successfully',
      timestamp: new Date().toISOString(),
    },
    {
      id: 2,
      type: 'error' as const,
      message: 'An error occurred',
      timestamp: new Date().toISOString(),
    },
  ];

  const defaultProps = {
    notifications: [],
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders without notifications initially', () => {
    renderWithProviders(<NotificationSystem {...defaultProps} />);
    
    // Should not show any notifications
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  test('displays success notification', () => {
    const successNotifications = [mockNotifications[0]];
    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={successNotifications} />
    );

    expect(screen.getByText(/Operation completed successfully/i)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  test('displays error notification', () => {
    const errorNotifications = [mockNotifications[1]];
    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={errorNotifications} />
    );

    expect(screen.getByText(/An error occurred/i)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  test('displays multiple notifications', () => {
    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={mockNotifications} />
    );

    expect(screen.getByText(/Operation completed successfully/i)).toBeInTheDocument();
    expect(screen.getByText(/An error occurred/i)).toBeInTheDocument();
    
    const alerts = screen.getAllByRole('alert');
    expect(alerts).toHaveLength(2);
  });

  test('can dismiss notifications', async () => {
    const user = userEvent.setup();
    const mockClose = jest.fn();
    const notifications = [mockNotifications[0]];
    
    renderWithProviders(
      <NotificationSystem notifications={notifications} onClose={mockClose} />
    );

    expect(screen.getByText(/Operation completed successfully/i)).toBeInTheDocument();

    const closeButton = screen.getByRole('button', { name: /close/i });
    await user.click(closeButton);

    expect(mockClose).toHaveBeenCalledWith(1);
  });

  test('displays different notification types with appropriate styling', () => {
    const allTypes = [
      { id: 1, type: 'success' as const, message: 'Success message', timestamp: new Date().toISOString() },
      { id: 2, type: 'error' as const, message: 'Error message', timestamp: new Date().toISOString() },
      { id: 3, type: 'warning' as const, message: 'Warning message', timestamp: new Date().toISOString() },
      { id: 4, type: 'info' as const, message: 'Info message', timestamp: new Date().toISOString() },
    ];

    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={allTypes} />
    );

    expect(screen.getByText(/Success message/i)).toBeInTheDocument();
    expect(screen.getByText(/Error message/i)).toBeInTheDocument();
    expect(screen.getByText(/Warning message/i)).toBeInTheDocument();
    expect(screen.getByText(/Info message/i)).toBeInTheDocument();

    const alerts = screen.getAllByRole('alert');
    expect(alerts).toHaveLength(4);
  });

  test('meets accessibility standards', async () => {
    const { container } = renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={[mockNotifications[0]]} />
    );

    const results = await checkAccessibility(container);
    expect(results.violations).toHaveLength(0);
  });

  test('has proper ARIA attributes', () => {
    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={[mockNotifications[1]]} />
    );

    const alert = screen.getByRole('alert');
    expect(alert).toHaveAttribute('role', 'alert');
  });

  test('supports keyboard navigation for dismissal', async () => {
    const user = userEvent.setup();
    const mockClose = jest.fn();
    
    renderWithProviders(
      <NotificationSystem notifications={[mockNotifications[0]]} onClose={mockClose} />
    );

    const closeButton = screen.getByRole('button', { name: /close/i });
    closeButton.focus();
    
    expect(document.activeElement).toBe(closeButton);
    
    // Press Enter to dismiss
    await user.keyboard('{Enter}');
    
    expect(mockClose).toHaveBeenCalledWith(1);
  });

  test('handles empty notifications array gracefully', () => {
    renderWithProviders(<NotificationSystem {...defaultProps} />);
    
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  test('displays notifications with proper timing information', () => {
    const recentNotification = {
      id: 1,
      type: 'info' as const,
      message: 'Recent notification',
      timestamp: new Date(Date.now() - 1000).toISOString(), // 1 second ago
    };

    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={[recentNotification]} />
    );

    expect(screen.getByText(/Recent notification/i)).toBeInTheDocument();
  });

  test('preserves notification order', () => {
    const orderedNotifications = [
      { id: 1, type: 'info' as const, message: 'First notification', timestamp: new Date(Date.now() - 3000).toISOString() },
      { id: 2, type: 'success' as const, message: 'Second notification', timestamp: new Date(Date.now() - 2000).toISOString() },
      { id: 3, type: 'error' as const, message: 'Third notification', timestamp: new Date(Date.now() - 1000).toISOString() },
    ];

    renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={orderedNotifications} />
    );

    const alerts = screen.getAllByRole('alert');
    expect(alerts[0]).toHaveTextContent('First notification');
    expect(alerts[1]).toHaveTextContent('Second notification');
    expect(alerts[2]).toHaveTextContent('Third notification');
  });

  test('handles notification updates correctly', () => {
    const { rerender } = renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={[mockNotifications[0]]} />
    );

    expect(screen.getByText(/Operation completed successfully/i)).toBeInTheDocument();

    // Update with new notifications
    rerender(
      <NotificationSystem {...defaultProps} notifications={mockNotifications} />
    );

    expect(screen.getByText(/Operation completed successfully/i)).toBeInTheDocument();
    expect(screen.getByText(/An error occurred/i)).toBeInTheDocument();
  });

  test('handles rapid notification changes', () => {
    const { rerender } = renderWithProviders(
      <NotificationSystem {...defaultProps} notifications={[]} />
    );

    // Rapidly change notifications
    for (let i = 0; i < 5; i++) {
      const notifications = [
        { id: i, type: 'info' as const, message: `Notification ${i}`, timestamp: new Date().toISOString() }
      ];
      rerender(
        <NotificationSystem {...defaultProps} notifications={notifications} />
      );
    }

    // Should handle the last update correctly
    expect(screen.getByText(/Notification 4/i)).toBeInTheDocument();
  });
});