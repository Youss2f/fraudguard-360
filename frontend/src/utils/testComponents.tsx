/**
 * React Test Utilities for FraudGuard 360
 * Components and providers for testing
 */

import React from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { testTheme, CustomRenderOptions } from './testUtils';

export const renderWithProviders = (
  ui: React.ReactElement,
  options: CustomRenderOptions = {}
) => {
  const { initialEntries = ['/'], withErrorBoundary = true, ...renderOptions } = options;
  
  const AllTheProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    return (
      <BrowserRouter>
        <ThemeProvider theme={testTheme}>
          {children}
        </ThemeProvider>
      </BrowserRouter>
    );
  };

  return render(ui, { wrapper: AllTheProviders, ...renderOptions });
};

// Re-export commonly used testing utilities
export * from '@testing-library/react';
export { renderWithProviders as render };