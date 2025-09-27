import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API client setup
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Types
export interface FraudAlert {
  alert_id: string;
  user_id: string;
  fraud_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  confidence_score: number;
  risk_score: number;
  description: string;
  evidence: any;
}

export interface CDR {
  cdr_id: string;
  caller_id: string;
  callee_id: string;
  timestamp: string;
  duration: number;
  call_type: string;
  location: string;
}

export interface NetworkVisualizationData {
  user_id: string;
  nodes: Array<{
    id: string;
    type: string;
    risk_score: number;
    label: string;
  }>;
  edges: Array<{
    source: string;
    target: string;
    type: string;
    weight: number;
  }>;
  risk_metrics: {
    network_risk: number;
    centrality: number;
  };
  fraud_indicators: Array<{
    type: string;
    severity: string;
    description: string;
  }>;
}

export interface UserRiskAssessment {
  user_id: string;
  risk_score: number;
  risk_level: string;
  fraud_probability: number;
  risk_factors: Array<{
    factor: string;
    value: number;
    impact: string;
  }>;
  recommendations: string[];
}

export interface DashboardData {
  summary: {
    total_alerts_today: number;
    critical_alerts: number;
    high_alerts: number;
    active_investigations: number;
    users_at_risk: number;
    fraud_cases_resolved: number;
  };
  alerts_by_hour: Array<{
    hour: string;
    count: number;
  }>;
  fraud_types: Array<{
    type: string;
    count: number;
    severity_breakdown?: {
      high: number;
      medium: number;
      low: number;
    };
  }>;
  top_risk_users: Array<{
    user_id: string;
    risk_score: number;
    alert_count: number;
  }>;
  system_performance?: {
    processing_time_ms: number;
    throughput_per_second: number;
    ml_model_accuracy: number;
    false_positive_rate: number;
  };
}

// API Services
export class FraudGuardApiService {
  // Health check
  static async healthCheck() {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Get fraud alerts
  static async getAlerts(params: {
    user_id?: string;
    severity?: string;
    fraud_type?: string;
    limit?: number;
    offset?: number;
  } = {}) {
    try {
      const response = await apiClient.get('/api/v1/alerts', { params });
      return response.data as FraudAlert[];
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      // Return mock data for development
      return this.getMockAlerts();
    }
  }

  // Analyze CDRs
  static async analyzeCDRs(cdrs: CDR[], analysisType: string = 'real_time') {
    try {
      const response = await apiClient.post('/api/v1/cdr/analyze', {
        cdrs,
        analysis_type: analysisType,
        options: {}
      });
      return response.data;
    } catch (error) {
      console.error('Failed to analyze CDRs:', error);
      throw error;
    }
  }

  // Get network visualization
  static async getNetworkVisualization(params: {
    user_id: string;
    depth?: number;
    relationship_types?: string[];
    time_window_hours?: number;
  }) {
    try {
      const response = await apiClient.post('/api/v1/network/visualize', params);
      return response.data as NetworkVisualizationData;
    } catch (error) {
      console.error('Failed to get network visualization:', error);
      // Return mock data for development
      return this.getMockNetworkData(params.user_id);
    }
  }

  // Get user risk assessment
  static async getUserRisk(userId: string) {
    try {
      const response = await apiClient.get(`/api/v1/users/${userId}/risk`);
      return response.data as UserRiskAssessment;
    } catch (error) {
      console.error('Failed to get user risk:', error);
      // Return mock data for development
      return this.getMockUserRisk(userId);
    }
  }

  // Get dashboard data
  static async getDashboardData() {
    try {
      const response = await apiClient.get('/api/v1/analytics/dashboard');
      return response.data as DashboardData;
    } catch (error) {
      console.error('Failed to get dashboard data:', error);
      // Return mock data for development
      return this.getMockDashboardData();
    }
  }

  // Mock data methods for development
  private static getMockAlerts(): FraudAlert[] {
    return [
      {
        alert_id: 'alert_001',
        user_id: 'user_001',
        fraud_type: 'sim_box',
        severity: 'critical',
        timestamp: new Date().toISOString(),
        confidence_score: 0.94,
        risk_score: 0.91,
        description: 'High-volume international calls with unusually short durations detected',
        evidence: {
          international_calls: 245,
          avg_duration: 32,
          call_frequency: 45.2,
          suspicious_destinations: ['IN', 'PK', 'NG']
        }
      },
      {
        alert_id: 'alert_002',
        user_id: 'user_002',
        fraud_type: 'velocity_anomaly',
        severity: 'high',
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        confidence_score: 0.87,
        risk_score: 0.83,
        description: 'Sudden 400% increase in call frequency over normal baseline',
        evidence: {
          normal_frequency: 15,
          current_frequency: 78,
          increase_percent: 420
        }
      },
      {
        alert_id: 'alert_003',
        user_id: 'user_003',
        fraud_type: 'premium_rate_fraud',
        severity: 'high',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        confidence_score: 0.89,
        risk_score: 0.85,
        description: 'Multiple calls to premium rate numbers with suspicious patterns',
        evidence: {
          premium_calls: 23,
          total_cost: 1850,
          avg_cost_per_call: 80.43
        }
      }
    ];
  }

  private static getMockNetworkData(userId: string): NetworkVisualizationData {
    return {
      user_id: userId,
      nodes: [
        { id: userId, type: 'user', risk_score: 0.75, label: `User ${userId}` },
        { id: 'device_001', type: 'device', risk_score: 0.30, label: 'Device 001' },
        { id: 'user_002', type: 'user', risk_score: 0.45, label: 'User 002' },
        { id: 'user_003', type: 'user', risk_score: 0.85, label: 'User 003' }
      ],
      edges: [
        { source: userId, target: 'device_001', type: 'uses', weight: 0.8 },
        { source: userId, target: 'user_002', type: 'calls', weight: 0.6 },
        { source: userId, target: 'user_003', type: 'calls', weight: 0.9 }
      ],
      risk_metrics: {
        network_risk: 0.65,
        centrality: 0.40
      },
      fraud_indicators: [
        {
          type: 'high_centrality',
          severity: 'medium',
          description: 'User has high network centrality indicating potential hub activity'
        }
      ]
    };
  }

  private static getMockUserRisk(userId: string): UserRiskAssessment {
    return {
      user_id: userId,
      risk_score: 0.75,
      risk_level: 'high',
      fraud_probability: 0.68,
      risk_factors: [
        { factor: 'call_frequency', value: 25.5, impact: 'high' },
        { factor: 'international_calls', value: 15, impact: 'medium' },
        { factor: 'velocity_anomaly', value: 8.2, impact: 'high' }
      ],
      recommendations: [
        'Monitor international call patterns closely',
        'Set up velocity checks for this user',
        'Review recent call destinations for premium numbers'
      ]
    };
  }

  private static getMockDashboardData(): DashboardData {
    return {
      summary: {
        total_alerts_today: 347,
        critical_alerts: 23,
        high_alerts: 89,
        active_investigations: 45,
        users_at_risk: 156,
        fraud_cases_resolved: 78
      },
      alerts_by_hour: [
        { hour: '00:00', count: 8 },
        { hour: '04:00', count: 12 },
        { hour: '08:00', count: 34 },
        { hour: '12:00', count: 56 },
        { hour: '16:00', count: 43 },
        { hour: '20:00', count: 29 }
      ],
      fraud_types: [
        { type: 'sim_box', count: 89, severity_breakdown: { high: 34, medium: 35, low: 20 } },
        { type: 'velocity_anomaly', count: 67, severity_breakdown: { high: 23, medium: 28, low: 16 } },
        { type: 'premium_rate_fraud', count: 54, severity_breakdown: { high: 18, medium: 22, low: 14 } },
        { type: 'location_anomaly', count: 43, severity_breakdown: { high: 12, medium: 19, low: 12 } }
      ],
      top_risk_users: [
        { user_id: 'user_001', risk_score: 0.95, alert_count: 8 },
        { user_id: 'user_003', risk_score: 0.88, alert_count: 5 },
        { user_id: 'user_007', risk_score: 0.82, alert_count: 6 }
      ],
      system_performance: {
        processing_time_ms: 85,
        throughput_per_second: 1245,
        ml_model_accuracy: 0.947,
        false_positive_rate: 0.062
      }
    };
  }
}

export default FraudGuardApiService;