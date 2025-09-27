/**
 * Real-time Dashboard Service
 * Handles WebSocket connections and real-time data updates for the FraudGuard dashboard
 */

export interface DashboardMetrics {
  totalTransactions: number;
  fraudDetected: number;
  falsePositives: number;
  accuracyRate: number;
  avgResponseTime: number;
  activeAlerts: number;
  systemHealth: 'healthy' | 'warning' | 'critical';
  trendsData: {
    timestamps: string[];
    transactions: number[];
    fraudCases: number[];
    accuracy: number[];
  };
}

export interface FraudAlert {
  alertId: string;
  alertType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'investigating' | 'resolved' | 'false_positive';
  title: string;
  description: string;
  userId: string;
  timestamp: string;
  riskScore: number;
  evidence: Record<string, any>;
  location?: string;
  assignedTo?: string;
}

export interface NetworkNode {
  id: string;
  label: string;
  type: 'user' | 'device' | 'location' | 'number';
  riskScore: number;
  fraudulent: boolean;
  metadata: Record<string, any>;
  x?: number;
  y?: number;
}

export interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  type: 'call' | 'sms' | 'data';
  suspicious: boolean;
  metadata: Record<string, any>;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  centerNode: string;
  timestamp: string;
}

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface ConnectionConfig {
  maxRetries: number;
  retryDelay: number;
  heartbeatInterval: number;
  reconnectOnClose: boolean;
}

class RealTimeDashboardService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private connectionState: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected';
  private retryCount = 0;
  private config: ConnectionConfig;
  
  // Event handlers
  private onMetricsUpdate: ((metrics: DashboardMetrics) => void) | null = null;
  private onAlertReceived: ((alert: FraudAlert) => void) | null = null;
  private onNetworkUpdate: ((graph: NetworkGraph) => void) | null = null;
  private onConnectionStateChange: ((state: string) => void) | null = null;
  
  constructor(config: Partial<ConnectionConfig> = {}) {
    this.config = {
      maxRetries: 5,
      retryDelay: 2000,
      heartbeatInterval: 30000,
      reconnectOnClose: true,
      ...config
    };
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(url: string = 'ws://localhost:8000/ws/dashboard'): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.connectionState = 'connecting';
      this.onConnectionStateChange?.(this.connectionState);

      try {
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
          console.log('🔗 Dashboard WebSocket connected');
          this.connectionState = 'connected';
          this.onConnectionStateChange?.('connected');
          this.retryCount = 0;
          this.startHeartbeat();
          this.processMessageQueue();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('🔌 Dashboard WebSocket disconnected:', event.code);
          this.connectionState = 'disconnected';
          this.onConnectionStateChange?.('disconnected');
          this.stopHeartbeat();
          
          if (this.config.reconnectOnClose && this.retryCount < this.config.maxRetries) {
            this.scheduleReconnect(url);
          }
        };

        this.ws.onerror = (error) => {
          console.error('🚨 Dashboard WebSocket error:', error);
          this.connectionState = 'error';
          this.onConnectionStateChange?.('error');
          reject(error);
        };

      } catch (error) {
        this.connectionState = 'error';
        this.onConnectionStateChange?.('error');
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket
   */
  public disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.connectionState = 'disconnected';
    this.onConnectionStateChange?.('disconnected');
  }

  /**
   * Send message to server
   */
  public sendMessage(type: string, data: any): void {
    const message: WebSocketMessage = {
      type,
      data,
      timestamp: new Date().toISOString()
    };

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message for later
      this.messageQueue.push(message);
    }
  }

  /**
   * Subscribe to specific alert types
   */
  public subscribeToAlerts(alertTypes: string[]): void {
    this.sendMessage('subscribe_alerts', { alertTypes });
  }

  /**
   * Request network graph for specific user
   */
  public requestNetworkGraph(userId: string, depth: number = 2): void {
    this.sendMessage('request_network', { userId, depth });
  }

  /**
   * Request dashboard metrics refresh
   */
  public refreshMetrics(): void {
    this.sendMessage('refresh_metrics', {});
  }

  /**
   * Set event handlers
   */
  public onMetrics(handler: (metrics: DashboardMetrics) => void): void {
    this.onMetricsUpdate = handler;
  }

  public onAlert(handler: (alert: FraudAlert) => void): void {
    this.onAlertReceived = handler;
  }

  public onNetwork(handler: (graph: NetworkGraph) => void): void {
    this.onNetworkUpdate = handler;
  }

  public onConnectionState(handler: (state: string) => void): void {
    this.onConnectionStateChange = handler;
  }

  /**
   * Get current connection state
   */
  public getConnectionState(): string {
    return this.connectionState;
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'metrics_update':
        this.onMetricsUpdate?.(message.data as DashboardMetrics);
        break;
        
      case 'new_alert':
        this.onAlertReceived?.(message.data as FraudAlert);
        break;
        
      case 'alert_update':
        this.onAlertReceived?.(message.data as FraudAlert);
        break;
        
      case 'network_graph':
        this.onNetworkUpdate?.(message.data as NetworkGraph);
        break;
        
      case 'pong':
        // Heartbeat response
        break;
        
      default:
        console.log('📨 Received unknown message type:', message.type);
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(url: string): void {
    this.retryCount++;
    const delay = this.config.retryDelay * Math.pow(2, this.retryCount - 1); // Exponential backoff
    
    console.log(`🔄 Scheduling reconnect attempt ${this.retryCount} in ${delay}ms`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.connect(url).catch(error => {
        console.error('Reconnection failed:', error);
        if (this.retryCount < this.config.maxRetries) {
          this.scheduleReconnect(url);
        }
      });
    }, delay);
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.sendMessage('ping', {});
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Process queued messages
   */
  private processMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(message));
      }
    }
  }
}

// Mock data generators for development
export class MockDataService {
  private intervalId: NodeJS.Timeout | null = null;
  
  public startMockData(dashboardService: RealTimeDashboardService): void {
    // Generate initial metrics
    this.generateMockMetrics(dashboardService);
    
    // Start periodic updates
    this.intervalId = setInterval(() => {
      // Random metrics updates
      if (Math.random() > 0.7) {
        this.generateMockMetrics(dashboardService);
      }
      
      // Random alerts
      if (Math.random() > 0.9) {
        this.generateMockAlert(dashboardService);
      }
      
      // Random network updates
      if (Math.random() > 0.8) {
        this.generateMockNetwork(dashboardService);
      }
    }, 5000);
  }
  
  public stopMockData(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
  
  private generateMockMetrics(service: RealTimeDashboardService): void {
    const now = new Date();
    const timestamps = Array.from({ length: 24 }, (_, i) => 
      new Date(now.getTime() - (23 - i) * 60 * 60 * 1000).toISOString()
    );
    
    const metrics: DashboardMetrics = {
      totalTransactions: Math.floor(Math.random() * 10000) + 50000,
      fraudDetected: Math.floor(Math.random() * 100) + 50,
      falsePositives: Math.floor(Math.random() * 20) + 5,
      accuracyRate: 0.95 + Math.random() * 0.04,
      avgResponseTime: Math.random() * 100 + 50,
      activeAlerts: Math.floor(Math.random() * 10) + 2,
      systemHealth: Math.random() > 0.8 ? 'healthy' : 'warning',
      trendsData: {
        timestamps,
        transactions: timestamps.map(() => Math.floor(Math.random() * 1000) + 2000),
        fraudCases: timestamps.map(() => Math.floor(Math.random() * 20) + 5),
        accuracy: timestamps.map(() => 0.9 + Math.random() * 0.1)
      }
    };
    
    // Use the public handler method
    const handler = (service as any).onMetricsUpdate;
    if (handler) handler(metrics);
  }
  
  private generateMockAlert(service: RealTimeDashboardService): void {
    const alertTypes = ['fraud_detection', 'velocity_fraud', 'sim_box_fraud', 'premium_rate_fraud'];
    const severities: ('low' | 'medium' | 'high' | 'critical')[] = ['low', 'medium', 'high', 'critical'];
    
    const alert: FraudAlert = {
      alertId: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      alertType: alertTypes[Math.floor(Math.random() * alertTypes.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
      status: 'open',
      title: `Suspicious Activity Detected`,
      description: `Potential fraud detected for user ${Math.random().toString(36).substr(2, 9)}`,
      userId: `user_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      riskScore: Math.random(),
      evidence: {
        callFrequency: Math.floor(Math.random() * 100) + 10,
        unusualLocation: Math.random() > 0.5,
        premiumCalls: Math.floor(Math.random() * 10)
      },
      location: `Cell_${Math.floor(Math.random() * 1000)}`
    };
    
    service.onAlertReceived?.(alert);
  }
  
  private generateMockNetwork(service: RealTimeDashboardService): void {
    const nodeCount = Math.floor(Math.random() * 20) + 10;
    const nodes: NetworkNode[] = [];
    const edges: NetworkEdge[] = [];
    
    // Generate nodes
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        id: `node_${i}`,
        label: `User ${i}`,
        type: Math.random() > 0.7 ? 'device' : 'user',
        riskScore: Math.random(),
        fraudulent: Math.random() > 0.8,
        metadata: {
          callCount: Math.floor(Math.random() * 100),
          location: `Location_${Math.floor(Math.random() * 10)}`
        },
        x: Math.random() * 800,
        y: Math.random() * 600
      });
    }
    
    // Generate edges
    for (let i = 0; i < nodeCount - 1; i++) {
      if (Math.random() > 0.3) {
        edges.push({
          id: `edge_${i}`,
          source: `node_${i}`,
          target: `node_${(i + 1) % nodeCount}`,
          weight: Math.random() * 10,
          type: Math.random() > 0.5 ? 'call' : 'sms',
          suspicious: Math.random() > 0.7,
          metadata: {
            frequency: Math.floor(Math.random() * 50),
            duration: Math.floor(Math.random() * 300)
          }
        });
      }
    }
    
    const networkGraph: NetworkGraph = {
      nodes,
      edges,
      centerNode: 'node_0',
      timestamp: new Date().toISOString()
    };
    
    service.onNetworkUpdate?.(networkGraph);
  }
}

// Singleton instance
const dashboardService = new RealTimeDashboardService();
const mockDataService = new MockDataService();

export { dashboardService, mockDataService };
export default dashboardService;