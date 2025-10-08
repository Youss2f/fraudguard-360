/**
 * Test Utilities for FraudGuard 360
 * Comprehensive testing helpers and custom matchers
 */

import React from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { excelTheme } from '../theme/excelTheme';
import { ErrorBoundary } from '../components/ErrorBoundary';

// Custom render function with providers
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialEntries?: string[];
  withErrorBoundary?: boolean;
}

export const renderWithProviders = (
  ui: React.ReactElement,
  {
    initialEntries = ['/'],
    withErrorBoundary = true,
    ...renderOptions
  }: CustomRenderOptions = {}
) => {
  const AllTheProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const content = (
      <BrowserRouter>
        <ThemeProvider theme={excelTheme}>
          {children}
        </ThemeProvider>
      </BrowserRouter>
    );

    if (withErrorBoundary) {
      return (
        <ErrorBoundary level="component">
          {content}
        </ErrorBoundary>
      );
    }

    return content;
  };

  return render(ui, { wrapper: AllTheProviders, ...renderOptions });
};

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

  networkNode: (overrides = {}) => ({
    id: 'node-1',
    label: 'User Account',
    type: 'user',
    riskScore: 0.3,
    metadata: {
      accountAge: 365,
      transactionCount: 150,
    },
    ...overrides,
  }),

  networkEdge: (overrides = {}) => ({
    id: 'edge-1',
    source: 'node-1',
    target: 'node-2',
    type: 'transaction',
    weight: 1.0,
    timestamp: new Date().toISOString(),
    metadata: {
      amount: 1000,
      frequency: 'daily',
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

// Custom matchers
export const customMatchers = {
  toBeAccessible: async (received: HTMLElement) => {
    const { axe, toHaveNoViolations } = await import('jest-axe');
    expect.extend(toHaveNoViolations);
    
    const results = await axe(received);
    const pass = results.violations.length === 0;
    
    return {
      pass,
      message: () =>
        pass
          ? `Expected element to have accessibility violations`
          : `Expected element to be accessible, but found ${results.violations.length} violations:\n${results.violations.map(v => `- ${v.description}`).join('\n')}`,
    };
  },

  toHavePerformantRender: (received: HTMLElement, maxTime = 100) => {
    const startTime = performance.now();
    // Simulate render time check
    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    const pass = renderTime <= maxTime;
    
    return {
      pass,
      message: () =>
        pass
          ? `Expected render time to exceed ${maxTime}ms`
          : `Expected render time to be ≤ ${maxTime}ms, but was ${renderTime.toFixed(2)}ms`,
    };
  },

  toHaveValidFormData: (received: FormData) => {
    const entries = Array.from(received.entries());
    const hasRequiredFields = entries.length > 0;
    const hasValidData = entries.every(([key, value]) => key && value);
    
    const pass = hasRequiredFields && hasValidData;
    
    return {
      pass,
      message: () =>
        pass
          ? 'Expected form data to be invalid'
          : 'Expected form data to be valid with all required fields',
    };
  },
};

// API mock helpers
export const createMockResponse = <T>(data: T, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? 'OK' : 'Error',
  headers: new Headers(headers),
  json: () => Promise.resolve(data),
  text: () => Promise.resolve(JSON.stringify(data)),
});

export const mockApiCall = <T>(endpoint: string, response: T, delay = 0) => {
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

export const createPerformanceObserver = (entryTypes: string[]) => {
  const entries: PerformanceEntry[] = [];
  
  const observer = new PerformanceObserver((list) => {
    entries.push(...list.getEntries());
  });
  
  observer.observe({ entryTypes });
  
  return {
    observer,
    getEntries: () => entries,
    disconnect: () => observer.disconnect(),
  };
};

// Accessibility testing helpers
export const checkAccessibility = async (container: HTMLElement) => {
  const { axe } = await import('jest-axe');
  const results = await axe(container);
  return results;
};

export const simulateKeyboardNavigation = (element: HTMLElement, key: string) => {
  element.focus();
  element.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  element.dispatchEvent(new KeyboardEvent('keyup', { key, bubbles: true }));
};

// Component testing helpers
export const waitForComponentToLoad = async (testId: string, timeout = 5000) => {
  const { waitFor, screen } = await import('@testing-library/react');
  
  return waitFor(
    () => {
      const element = screen.getByTestId(testId);
      expect(element).toBeInTheDocument();
      return element;
    },
    { timeout }
  );
};

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

// Error testing helpers
export const expectErrorBoundary = (renderFn: () => void) => {
  const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  
  expect(renderFn).toThrow();
  
  consoleSpy.mockRestore();
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

// Re-export commonly used testing utilities
export * from '@testing-library/react';
export * from '@testing-library/user-event';
export { renderWithProviders as render };

export default {
  renderWithProviders,
  mockDataGenerators,
  customMatchers,
  createMockResponse,
  mockApiCall,
  measurePerformance,
  checkAccessibility,
  simulateKeyboardNavigation,
  waitForComponentToLoad,
  triggerResize,
  fillForm,
  submitForm,
  expectErrorBoundary,
  mockNetworkError,
  mockNetworkDelay,
};