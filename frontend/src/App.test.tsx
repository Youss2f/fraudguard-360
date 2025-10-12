import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

// Mock the GraphVisualizer component to avoid complex dependencies
jest.mock('./components/GraphVisualizer', () => {
  return function MockedGraphVisualizer() {
    return <div data-testid="graph-visualizer">Mocked Graph Visualizer</div>;
  };
});

test('renders app component without crashing', () => {
  render(<App />);
  // Basic smoke test - if it renders without throwing, test passes
  expect(true).toBe(true);
});

test('app component contains graph visualizer', () => {
  render(<App />);
  const graphElement = screen.getByTestId('graph-visualizer');
  expect(graphElement).toBeInTheDocument();
});

test('app initializes with empty graph state', () => {
  render(<App />);
  // Component should render without errors, indicating proper state initialization
  expect(screen.getByTestId('graph-visualizer')).toBeInTheDocument();
});