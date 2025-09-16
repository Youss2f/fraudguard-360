import { renderHook } from '@testing-library/react';
import { useWebSocket } from './useWebSocket';

// Mock WebSocket
global.WebSocket = jest.fn(() => ({
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  close: jest.fn(),
  send: jest.fn(),
  readyState: 1,
})) as any;

test('useWebSocket hook initializes without crashing', () => {
  const { result } = renderHook(() => useWebSocket('ws://localhost:8080'));
  
  // Basic test to ensure hook doesn't crash on initialization
  expect(result.current).toBeDefined();
});

test('useWebSocket hook handles empty URL', () => {
  const { result } = renderHook(() => useWebSocket(''));
  
  // Should handle empty URL gracefully
  expect(result.current).toBeDefined();
});