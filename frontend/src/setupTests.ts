/**
 * Enhanced Test Setup Configuration
 * Comprehensive testing utilities and mocks for FraudGuard 360
 */

import '@testing-library/jest-dom';

// Mock window.matchMedia for responsive design tests
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock ResizeObserver for components that use it
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver for performance tests
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock Service Worker for performance tests
Object.defineProperty(navigator, 'serviceWorker', {
  value: {
    register: jest.fn(() => Promise.resolve({
      installing: null,
      waiting: null,
      active: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    })),
    ready: Promise.resolve({
      installing: null,
      waiting: null,
      active: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    }),
  },
  configurable: true,
});

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock as any;

// Mock sessionStorage
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.sessionStorage = sessionStorageMock as any;

// Mock Web Vitals for performance testing
jest.mock('web-vitals', () => ({
  getCLS: jest.fn(),
  getFID: jest.fn(),
  getFCP: jest.fn(),
  getLCP: jest.fn(),
  getTTFB: jest.fn(),
  onCLS: jest.fn(),
  onFID: jest.fn(),
  onFCP: jest.fn(),
  onLCP: jest.fn(),
  onTTFB: jest.fn(),
}));

// Mock Cytoscape for network visualization tests
jest.mock('cytoscape', () => ({
  __esModule: true,
  default: jest.fn(() => ({
    add: jest.fn(),
    remove: jest.fn(),
    getElementById: jest.fn(),
    layout: jest.fn(() => ({ run: jest.fn() })),
    on: jest.fn(),
    off: jest.fn(),
    destroy: jest.fn(),
    elements: jest.fn(() => []),
    style: jest.fn(() => ({ update: jest.fn() })),
    fit: jest.fn(),
    center: jest.fn(),
    zoom: jest.fn(),
    pan: jest.fn(),
  })),
}));

// Mock React Router for navigation tests
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
  useLocation: () => ({
    pathname: '/',
    search: '',
    hash: '',
    state: null,
  }),
}));

// Mock axios for API tests
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
    put: jest.fn(() => Promise.resolve({ data: {} })),
    delete: jest.fn(() => Promise.resolve({ data: {} })),
    patch: jest.fn(() => Promise.resolve({ data: {} })),
    interceptors: {
      request: { use: jest.fn(), eject: jest.fn() },
      response: { use: jest.fn(), eject: jest.fn() },
    },
  })),
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} })),
  put: jest.fn(() => Promise.resolve({ data: {} })),
  delete: jest.fn(() => Promise.resolve({ data: {} })),
  patch: jest.fn(() => Promise.resolve({ data: {} })),
}));

// Mock Socket.IO for real-time features
jest.mock('socket.io-client', () => ({
  io: jest.fn(() => ({
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
    connect: jest.fn(),
    disconnect: jest.fn(),
    connected: true,
  })),
}));

// Mock Chart.js for analytics dashboard tests
jest.mock('recharts', () => ({
  LineChart: jest.fn(({ children }) => children),
  Line: jest.fn(() => null),
  XAxis: jest.fn(() => null),
  YAxis: jest.fn(() => null),
  CartesianGrid: jest.fn(() => null),
  Tooltip: jest.fn(() => null),
  Legend: jest.fn(() => null),
  ResponsiveContainer: jest.fn(({ children }) => children),
  BarChart: jest.fn(({ children }) => children),
  Bar: jest.fn(() => null),
  PieChart: jest.fn(({ children }) => children),
  Pie: jest.fn(() => null),
  Cell: jest.fn(() => null),
}));

// Mock Web APIs for performance tests
Object.defineProperty(window, 'performance', {
  value: {
    mark: jest.fn(),
    measure: jest.fn(),
    getEntriesByType: jest.fn(() => []),
    getEntriesByName: jest.fn(() => []),
    clearMarks: jest.fn(),
    clearMeasures: jest.fn(),
    now: jest.fn(() => Date.now()),
  },
  configurable: true,
});

// Global test utilities
declare global {
  var testUtils: {
    triggerResize: (width: number, height: number) => void;
    mockNetworkError: () => jest.Mock;
    mockNetworkDelay: (delay: number) => jest.Mock;
    simulateKeyboardNavigation: (element: HTMLElement, key: string) => void;
  };
}

(global as any).testUtils = {
  // Mock user data
  mockUser: {
    id: 'test-user-1',
    username: 'testuser',
    email: 'test@example.com',
    role: 'analyst',
    firstName: 'Test',
    lastName: 'User',
    department: 'fraud-detection',
  },
  
  // Mock fraud data
  mockFraudCase: {
    id: 'fraud-case-1',
    title: 'Suspicious Transaction Pattern',
    severity: 'high',
    status: 'active',
    riskScore: 0.85,
    amount: 10000,
    assignedTo: 'test-user-1',
    createdAt: '2024-01-01T12:00:00Z',
  },
  
  // Mock network data
  mockNetworkData: {
    nodes: [
      { id: 'node1', label: 'User A', type: 'user' },
      { id: 'node2', label: 'User B', type: 'user' },
    ],
    edges: [
      { id: 'edge1', source: 'node1', target: 'node2', type: 'transaction' },
    ],
  },
  
  // Mock analytics data
  mockAnalyticsData: {
    fraudStats: {
      totalCases: 150,
      activeCases: 25,
      resolvedCases: 125,
      avgRiskScore: 0.65,
    },
    trendData: [
      { date: '2024-01-01', cases: 10, amount: 50000 },
      { date: '2024-01-02', cases: 15, amount: 75000 },
    ],
  },
};

// Console override for cleaner test output
const originalError = console.error;
beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is no longer supported')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});

// Global cleanup
afterEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});

export {};