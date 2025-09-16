import React from 'react';
import { render } from '@testing-library/react';
import GraphVisualizer from './GraphVisualizer';

// Mock Cytoscape to avoid complex graph rendering in tests
jest.mock('cytoscape', () => {
  return jest.fn(() => ({
    mount: jest.fn(),
    layout: jest.fn(() => ({ run: jest.fn() })),
    add: jest.fn(),
    remove: jest.fn(),
    destroy: jest.fn(),
  }));
});

jest.mock('react-cytoscapejs', () => {
  return function MockedCytoscape() {
    return <div data-testid="cytoscape-graph">Mocked Cytoscape Graph</div>;
  };
});

const mockGraph = {
  nodes: [],
  edges: []
};

const mockOnNodeClick = jest.fn();

test('renders graph visualizer component with empty graph', () => {
  render(<GraphVisualizer graph={mockGraph} onNodeClick={mockOnNodeClick} />);
  // Basic smoke test for component rendering
  expect(true).toBe(true);
});

test('graph visualizer shows fallback message for empty graph', () => {
  const { getByText } = render(<GraphVisualizer graph={mockGraph} onNodeClick={mockOnNodeClick} />);
  expect(getByText(/no graph data/i)).toBeInTheDocument();
});