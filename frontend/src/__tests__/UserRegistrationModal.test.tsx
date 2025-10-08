/**
 * User Registration Modal Component Tests
 */

import React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../utils/testComponents';
import { mockApiCall, fillForm, submitForm } from '../utils/testUtils';
import { checkAccessibility } from '../utils/accessibilityUtils';
import UserRegistrationModal from '../components/UserRegistrationModal';

// Mock API calls
const mockRegister = mockApiCall('/api/register', { success: true, userId: '123' });

jest.mock('../services/api', () => ({
  registerUser: mockRegister,
}));

describe('UserRegistrationModal Component', () => {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    onSubmit: jest.fn().mockResolvedValue(true),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders modal when open', () => {
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/User Registration/i)).toBeInTheDocument();
    expect(screen.getByText(/Step 1/i)).toBeInTheDocument();
  });

  test('does not render when closed', () => {
    renderWithProviders(<UserRegistrationModal {...defaultProps} open={false} />);

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('displays form validation errors', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Try to submit without filling required fields
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    await waitFor(() => {
      expect(screen.getByText(/First name is required/i)).toBeInTheDocument();
      expect(screen.getByText(/Last name is required/i)).toBeInTheDocument();
      expect(screen.getByText(/Email is required/i)).toBeInTheDocument();
    });
  });

  test('validates email format', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    const emailField = screen.getByLabelText(/email/i);
    await user.type(emailField, 'invalid-email');

    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    await waitFor(() => {
      expect(screen.getByText(/Please enter a valid email/i)).toBeInTheDocument();
    });
  });

  test('progresses through multiple steps', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Fill step 1
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');

    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Should move to step 2
    await waitFor(() => {
      expect(screen.getByText(/Step 2/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    });
  });

  test('validates password strength', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Navigate to step 2
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => {
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    });

    // Enter weak password
    const passwordField = screen.getByLabelText(/^password$/i);
    await user.type(passwordField, '123');

    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    await waitFor(() => {
      expect(screen.getByText(/Password must be at least 8 characters/i)).toBeInTheDocument();
    });
  });

  test('validates password confirmation', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Navigate to step 2
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => {
      expect(screen.getByLabelText(/^password$/i)).toBeInTheDocument();
    });

    // Enter mismatched passwords
    await user.type(screen.getByLabelText(/^password$/i), 'StrongPassword123!');
    await user.type(screen.getByLabelText(/confirm password/i), 'DifferentPassword123!');

    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    await waitFor(() => {
      expect(screen.getByText(/Passwords do not match/i)).toBeInTheDocument();
    });
  });

  test('allows going back to previous step', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Fill step 1 and proceed
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => {
      expect(screen.getByText(/Step 2/i)).toBeInTheDocument();
    });

    // Go back
    const backButton = screen.getByRole('button', { name: /back/i });
    await user.click(backButton);

    await waitFor(() => {
      expect(screen.getByText(/Step 1/i)).toBeInTheDocument();
      expect(screen.getByDisplayValue('John')).toBeInTheDocument();
    });
  });

  test('submits registration successfully', async () => {
    const user = userEvent.setup();
    const mockSubmit = jest.fn().mockResolvedValue(true);
    renderWithProviders(
      <UserRegistrationModal {...defaultProps} onSubmit={mockSubmit} />
    );

    // Complete step 1
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Complete step 2
    await waitFor(() => {
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    });
    
    await user.type(screen.getByLabelText(/username/i), 'johndoe');
    await user.type(screen.getByLabelText(/^password$/i), 'StrongPassword123!');
    await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Complete step 3 (department selection)
    await waitFor(() => {
      expect(screen.getByText(/Step 3/i)).toBeInTheDocument();
    });

    const departmentSelect = screen.getByLabelText(/department/i);
    await user.selectOptions(departmentSelect, 'fraud-detection');
    
    const submitButton = screen.getByRole('button', { name: /register/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockSubmit).toHaveBeenCalledWith({
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@test.com',
        username: 'johndoe',
        password: 'StrongPassword123!',
        department: 'fraud-detection',
      });
    });
  });

  test('handles registration error', async () => {
    const user = userEvent.setup();
    const mockSubmit = jest.fn().mockRejectedValue(new Error('Email already exists'));

    renderWithProviders(<UserRegistrationModal {...defaultProps} onSubmit={mockSubmit} />);

    // Fill and submit form
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => {
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    });
    
    await user.type(screen.getByLabelText(/username/i), 'johndoe');
    await user.type(screen.getByLabelText(/^password$/i), 'StrongPassword123!');
    await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => {
      expect(screen.getByLabelText(/department/i)).toBeInTheDocument();
    });

    await user.selectOptions(screen.getByLabelText(/department/i), 'fraud-detection');
    await user.click(screen.getByRole('button', { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByText(/Email already exists/i)).toBeInTheDocument();
    });
  });

  test('closes modal when close button clicked', async () => {
    const user = userEvent.setup();
    const mockClose = jest.fn();
    renderWithProviders(
      <UserRegistrationModal {...defaultProps} onClose={mockClose} />
    );

    const closeButton = screen.getByRole('button', { name: /close/i });
    await user.click(closeButton);

    expect(mockClose).toHaveBeenCalled();
  });

  test('closes modal on escape key', () => {
    const mockClose = jest.fn();
    renderWithProviders(
      <UserRegistrationModal {...defaultProps} onClose={mockClose} />
    );

    const modal = screen.getByRole('dialog');
    fireEvent.keyDown(modal, { key: 'Escape' });

    expect(mockClose).toHaveBeenCalled();
  });

  test('meets accessibility standards', async () => {
    const { container } = renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    const results = await checkAccessibility(container);
    expect(results.violations).toHaveLength(0);
  });

  test('has proper ARIA attributes', () => {
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    const modal = screen.getByRole('dialog');
    expect(modal).toHaveAttribute('aria-labelledby');
    expect(modal).toHaveAttribute('aria-describedby');
  });

  test('supports keyboard navigation', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    const firstNameField = screen.getByLabelText(/first name/i);
    const lastNameField = screen.getByLabelText(/last name/i);
    const emailField = screen.getByLabelText(/email/i);

    // Tab through form fields
    firstNameField.focus();
    expect(document.activeElement).toBe(firstNameField);

    await user.tab();
    expect(document.activeElement).toBe(lastNameField);

    await user.tab();
    expect(document.activeElement).toBe(emailField);
  });

  test('shows progress indicator', () => {
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Should show step progress
    expect(screen.getByText(/Step 1 of 3/i)).toBeInTheDocument();
  });

  test('preserves form data when navigating back', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Fill step 1
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Go to step 2, then back
    await waitFor(() => {
      expect(screen.getByText(/Step 2/i)).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /back/i }));

    // Data should be preserved
    await waitFor(() => {
      expect(screen.getByDisplayValue('John')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Doe')).toBeInTheDocument();
      expect(screen.getByDisplayValue('john.doe@test.com')).toBeInTheDocument();
    });
  });

  test('disables next button when form is invalid', async () => {
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    const nextButton = screen.getByRole('button', { name: /next/i });
    
    // Should be disabled initially
    expect(nextButton).toBeDisabled();

    // Should remain disabled with partial data
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/first name/i), 'John');
    
    expect(nextButton).toBeDisabled();
  });

  test('enables next button when form is valid', async () => {
    const user = userEvent.setup();
    renderWithProviders(<UserRegistrationModal {...defaultProps} />);

    // Fill all required fields
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john.doe@test.com');

    const nextButton = screen.getByRole('button', { name: /next/i });
    
    await waitFor(() => {
      expect(nextButton).toBeEnabled();
    });
  });
});