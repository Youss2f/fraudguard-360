/**
 * Test Utilities for FraudGuard 360
 * Comprehensive testing helpers and custom matchers
 */

import { createTheme } from '@mui/material/styles';

// Default test theme (simplified version)
const testTheme = createTheme({
  palette: {
    primary: {
      main: '#0066cc',
    },
    secondary: {
      main: '#ff9800',
    },
  },
});

// Custom render function options
interface CustomRenderOptions {
  initialEntries?: string[];
  withErrorBoundary?: boolean;
}

// Mock data generators
export const mockDataGenerators = {
  user: (overrides = {}) => ({
    id: 'user-1',
    username: 'testuser',
    email: 'test@fraudguard360.com',
    firstName: 'Test',
    lastName: 'User',
    role: 'analyst',
    department: 'fraud-detection',
    isActive: true,
    lastLogin: new Date().toISOString(),
    ...overrides,
  }),

  fraudCase: (overrides = {}) => ({
    id: 'case-1',
    title: 'Suspicious Transaction Pattern',
    description: 'Multiple high-value transactions detected',
    severity: 'high',
    status: 'active',
    riskScore: 0.85,
    amount: 10000,
    currency: 'USD',
    assignedTo: 'user-1',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    tags: ['transaction', 'pattern', 'high-risk'],
    ...overrides,
  }),

  fraudAlert: (overrides = {}) => ({
    id: 'alert-1',
    type: 'pattern-detection',
    severity: 'medium',
    message: 'Unusual calling pattern detected',
    timestamp: new Date().toISOString(),
    isRead: false,
    metadata: {
      pattern: 'rapid-succession-calls',
      confidence: 0.75,
    },
    ...overrides,
  }),

  analyticsData: (overrides = {}) => ({
    summary: {
      totalCases: 150,
      activeCases: 25,
      resolvedCases: 125,
      avgRiskScore: 0.65,
      trendsUp: 15,
      trendsDown: 5,
    },
    timeSeries: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      cases: Math.floor(Math.random() * 20) + 5,
      amount: Math.floor(Math.random() * 100000) + 10000,
      riskScore: Math.random() * 0.5 + 0.3,
    })),
    ...overrides,
  }),
};

// API mock helpers
export const createMockResponse = (data: any, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? 'OK' : 'Error',
  headers: new Headers(headers),
  json: () => Promise.resolve(data),
  text: () => Promise.resolve(JSON.stringify(data)),
});

export const mockApiCall = (endpoint: string, response: any, delay = 0) => {
  return jest.fn().mockImplementation(() =>
    new Promise((resolve) =>
      setTimeout(() => resolve(createMockResponse(response)), delay)
    )
  );
};

// Performance testing helpers
export const measurePerformance = async (fn: () => Promise<void> | void) => {
  const start = performance.now();
  await fn();
  const end = performance.now();
  return end - start;
};

// Component testing helpers
export const triggerResize = (width: number, height: number) => {
  Object.defineProperty(window, 'innerWidth', { value: width, configurable: true });
  Object.defineProperty(window, 'innerHeight', { value: height, configurable: true });
  window.dispatchEvent(new Event('resize'));
};

// Form testing helpers
export const fillForm = async (formData: Record<string, string>) => {
  const { screen } = await import('@testing-library/react');
  const userEvent = await import('@testing-library/user-event');
  const user = userEvent.default.setup();
  
  for (const [fieldName, value] of Object.entries(formData)) {
    const field = screen.getByLabelText(new RegExp(fieldName, 'i'));
    await user.clear(field);
    await user.type(field, value);
  }
};

export const submitForm = async (formTestId: string) => {
  const { screen } = await import('@testing-library/react');
  const userEvent = await import('@testing-library/user-event');
  const user = userEvent.default.setup();
  
  const submitButton = screen.getByRole('button', { name: /submit|save|create/i });
  await user.click(submitButton);
};

// Accessibility testing helpers
export const simulateKeyboardNavigation = (element: HTMLElement, key: string) => {
  element.focus();
  element.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  element.dispatchEvent(new KeyboardEvent('keyup', { key, bubbles: true }));
};

// Network testing helpers
export const mockNetworkError = () => {
  return jest.fn().mockRejectedValue(new Error('Network Error'));
};

export const mockNetworkDelay = (delay: number) => {
  return jest.fn().mockImplementation(
    () => new Promise(resolve => setTimeout(resolve, delay))
  );
};

// We'll export the theme and options for use in tsx files
export { testTheme };
export type { CustomRenderOptions };

export default {
  mockDataGenerators,
  createMockResponse,
  mockApiCall,
  measurePerformance,
  simulateKeyboardNavigation,
  triggerResize,
  fillForm,
  submitForm,
  mockNetworkError,
  mockNetworkDelay,
};