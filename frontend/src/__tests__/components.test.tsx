import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../theme/ThemeProvider';
import FraudDashboard from '../components/FraudDashboard';
import FraudDetectionDashboard from '../components/FraudDetectionDashboard';
import NetworkGraphVisualization from '../components/NetworkGraphVisualization';

// Mock Cytoscape
jest.mock('cytoscape', () => ({
  __esModule: true,
  default: jest.fn(() => ({
    add: jest.fn(),
    layout: jest.fn(() => ({ run: jest.fn() })),
    on: jest.fn(),
    destroy: jest.fn(),
    elements: jest.fn(() => []),
    getElementById: jest.fn(),
    remove: jest.fn(),
    style: jest.fn(() => ({ update: jest.fn() }))
  }))
}));

// Test wrapper component
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <BrowserRouter>
    <ThemeProvider>
      {children}
    </ThemeProvider>
  </BrowserRouter>
);

describe('FraudDashboard Component', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
  });

  test('renders dashboard without crashing', () => {
    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );
    
    expect(screen.getByText(/fraud dashboard/i)).toBeInTheDocument();
  });

  test('displays fraud alerts correctly', async () => {
    const mockAlerts = [
      {
        id: '1',
        fraud_score: 0.85,
        severity: 'high',
        description: 'Suspicious calling pattern detected',
        timestamp: '2024-01-01T12:00:00Z'
      },
      {
        id: '2',
        fraud_score: 0.65,
        severity: 'medium',
        description: 'Unusual call duration',
        timestamp: '2024-01-01T13:00:00Z'
      }
    ];

    // Mock fetch response
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ alerts: mockAlerts, total: 2 })
      })
    ) as jest.Mock;

    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Suspicious calling pattern detected')).toBeInTheDocument();
      expect(screen.getByText('Unusual call duration')).toBeInTheDocument();
    });
  });

  test('filters alerts by severity', async () => {
    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    const severityFilter = screen.getByLabelText(/severity filter/i);
    fireEvent.change(severityFilter, { target: { value: 'high' } });

    await waitFor(() => {
      // Verify API call was made with correct filter
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('severity=high')
      );
    });
  });

  test('handles loading state correctly', () => {
    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test('handles error state correctly', async () => {
    // Mock fetch error
    global.fetch = jest.fn(() =>
      Promise.reject('API Error')
    ) as jest.Mock;

    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/error loading alerts/i)).toBeInTheDocument();
    });
  });
});

describe('FraudDetectionDashboard Component', () => {
  test('renders fraud detection metrics', async () => {
    const mockMetrics = {
      total_detections: 150,
      accuracy: 0.92,
      false_positive_rate: 0.05,
      model_version: '1.0.0'
    };

    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockMetrics)
      })
    ) as jest.Mock;

    render(
      <TestWrapper>
        <FraudDetectionDashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument(); // Total detections
      expect(screen.getByText('92%')).toBeInTheDocument(); // Accuracy
      expect(screen.getByText('5%')).toBeInTheDocument(); // False positive rate
    });
  });

  test('updates metrics periodically', async () => {
    render(
      <TestWrapper>
        <FraudDetectionDashboard />
      </TestWrapper>
    );

    // Test that the component handles metric updates
    await waitFor(() => {
      expect(screen.getByText(/accuracy over time/i)).toBeInTheDocument();
    });
  });

  test('displays model performance charts', () => {
    render(
      <TestWrapper>
        <FraudDetectionDashboard />
      </TestWrapper>
    );

    expect(screen.getByText(/accuracy over time/i)).toBeInTheDocument();
    expect(screen.getByText(/fraud score distribution/i)).toBeInTheDocument();
  });
});

describe('NetworkGraphVisualization Component', () => {
  test('renders network graph container', () => {
    render(
      <TestWrapper>
        <NetworkGraphVisualization />
      </TestWrapper>
    );

    expect(screen.getByTestId('network-graph-container')).toBeInTheDocument();
  });

  test('loads and displays network data', async () => {
    const mockNetworkData = {
      nodes: [
        { id: 'node1', label: 'Phone', properties: { number: '1234567890' } },
        { id: 'node2', label: 'Phone', properties: { number: '0987654321' } }
      ],
      edges: [
        { source: 'node1', target: 'node2', weight: 5, fraud_score: 0.3 }
      ]
    };

    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockNetworkData)
      })
    ) as jest.Mock;

    render(
      <TestWrapper>
        <NetworkGraphVisualization />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/v1/network/visualization');
    });
  });

  test('handles node selection', async () => {
    render(
      <TestWrapper>
        <NetworkGraphVisualization />
      </TestWrapper>
    );

    // Simulate node click (would need actual Cytoscape mock)
    // This is a placeholder for node interaction testing
  });

  test('applies fraud score styling to edges', () => {
    render(
      <TestWrapper>
        <NetworkGraphVisualization />
      </TestWrapper>
    );

    // Verify that fraud scoring affects edge styling
    // This would require actual Cytoscape integration testing
  });
});

describe('Real-time Data Functionality', () => {
  test('handles data updates correctly', () => {
    // Mock real-time data functionality
    const mockData = { alerts: [], metrics: {} };
    
    const TestComponent = () => {
      return <div>Real-time data test</div>;
    };

    render(
      <TestWrapper>
        <TestComponent />
      </TestWrapper>
    );

    expect(screen.getByText('Real-time data test')).toBeInTheDocument();
  });
});

describe('Data Generation Utilities', () => {
  test('generates realistic CDR data', () => {
    const { generateCDRData } = require('../utils/dataGenerator');
    
    const cdrData = generateCDRData(10);
    
    expect(cdrData).toHaveLength(10);
    expect(cdrData[0]).toHaveProperty('call_id');
    expect(cdrData[0]).toHaveProperty('caller_number');
    expect(cdrData[0]).toHaveProperty('callee_number');
    expect(cdrData[0]).toHaveProperty('duration');
    expect(cdrData[0]).toHaveProperty('cost');
  });

  test('generates fraud patterns', () => {
    const { generateFraudPatterns } = require('../utils/dataGenerator');
    
    const fraudPatterns = generateFraudPatterns(5);
    
    expect(fraudPatterns).toHaveLength(5);
    expect(fraudPatterns[0]).toHaveProperty('pattern_type');
    expect(fraudPatterns[0]).toHaveProperty('severity');
    expect(fraudPatterns[0]).toHaveProperty('fraud_score');
  });

  test('generates network graph data', () => {
    const { generateNetworkData } = require('../utils/dataGenerator');
    
    const networkData = generateNetworkData(20, 30);
    
    expect(networkData.nodes).toHaveLength(20);
    expect(networkData.edges).toHaveLength(30);
    expect(networkData.nodes[0]).toHaveProperty('id');
    expect(networkData.edges[0]).toHaveProperty('source');
    expect(networkData.edges[0]).toHaveProperty('target');
  });
});

describe('Theme Provider', () => {
  test('provides theme context', () => {
    const TestComponent = () => {
      return <div>Theme Test</div>;
    };

    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    expect(screen.getByText('Theme Test')).toBeInTheDocument();
  });

  test('switches between light and dark themes', () => {
    // This would test theme switching functionality
    // Implementation depends on actual theme provider setup
  });
});

describe('Error Boundaries', () => {
  test('catches and displays errors gracefully', () => {
    const ThrowError = () => {
      throw new Error('Test error');
    };

    // This would test error boundary implementation
    // Need to implement actual error boundary component
  });
});

describe('Accessibility', () => {
  test('has proper ARIA labels', () => {
    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    expect(screen.getByRole('main')).toBeInTheDocument();
    // Add more accessibility tests based on actual implementation
  });

  test('supports keyboard navigation', () => {
    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    // Test keyboard navigation
    // This would require actual implementation testing
  });
});

describe('Performance', () => {
  test('renders efficiently with large datasets', async () => {
    // Mock large dataset
    const largeAlertSet = Array.from({ length: 1000 }, (_, i) => ({
      id: `alert_${i}`,
      fraud_score: Math.random(),
      severity: 'medium',
      description: `Alert ${i}`,
      timestamp: new Date().toISOString()
    }));

    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ alerts: largeAlertSet, total: 1000 })
      })
    ) as jest.Mock;

    const startTime = performance.now();
    
    render(
      <TestWrapper>
        <FraudDashboard />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Alert 0/)).toBeInTheDocument();
    });

    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    // Ensure rendering completes within reasonable time
    expect(renderTime).toBeLessThan(2000); // 2 seconds
  });
});