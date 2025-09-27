/**
 * Enhanced WebSocket Service for FraudGuard 360
 * Real-time data streaming with automatic reconnection and event handling
 */

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
  id?: string;
}

export interface ConnectionStatus {
  connected: boolean;
  reconnecting: boolean;
  lastConnected?: Date;
  lastError?: string;
  reconnectAttempts: number;
}

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  timeout?: number;
}

type EventHandler = (data: any) => void;
type ConnectionHandler = (status: ConnectionStatus) => void;

class EnhancedWebSocketService {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private eventHandlers: Map<string, EventHandler[]> = new Map();
  private connectionHandlers: ConnectionHandler[] = [];
  private status: ConnectionStatus = {
    connected: false,
    reconnecting: false,
    reconnectAttempts: 0
  };
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private destroyed = false;

  constructor(config: WebSocketConfig) {
    this.config = {
      protocols: [],
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      timeout: 10000,
      ...config,
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.destroyed) {
        reject(new Error('WebSocket service has been destroyed'));
        return;
      }

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.cleanup();

      try {
        console.log(`🔌 Connecting to WebSocket: ${this.config.url}`);
        this.ws = new WebSocket(this.config.url, this.config.protocols);

        const timeout = setTimeout(() => {
          if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
            this.ws.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, this.config.timeout);

        this.ws.onopen = () => {
          clearTimeout(timeout);
          console.log('✅ WebSocket connected successfully');
          
          this.status = {
            connected: true,
            reconnecting: false,
            lastConnected: new Date(),
            reconnectAttempts: 0
          };
          
          this.notifyConnectionHandlers();
          this.startHeartbeat();
          this.flushMessageQueue();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          clearTimeout(timeout);
          console.log(`🔌 WebSocket disconnected: ${event.code} - ${event.reason}`);
          
          this.status.connected = false;
          this.notifyConnectionHandlers();
          this.stopHeartbeat();

          if (!this.destroyed && !event.wasClean) {
            this.attemptReconnect();
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(timeout);
          console.error('❌ WebSocket error:', error);
          
          this.status.lastError = 'Connection error';
          this.notifyConnectionHandlers();
          
          if (!this.status.connected) {
            reject(error);
          }
        };

      } catch (error) {
        console.error('❌ Failed to create WebSocket:', error);
        reject(error);
      }
    });
  }

  private handleMessage(event: MessageEvent) {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      // Handle special message types
      if (message.type === 'heartbeat') {
        this.sendHeartbeatResponse();
        return;
      }

      if (message.type === 'error') {
        console.error('WebSocket server error:', message.data);
        return;
      }

      // Dispatch to registered handlers
      const handlers = this.eventHandlers.get(message.type) || [];
      handlers.forEach(handler => {
        try {
          handler(message.data);
        } catch (error) {
          console.error(`Error in event handler for ${message.type}:`, error);
        }
      });

      // Log received message (except frequent ones)
      if (!['heartbeat', 'ping', 'metrics'].includes(message.type)) {
        console.log(`📨 WebSocket message: ${message.type}`, message.data);
      }

    } catch (error) {
      console.error('❌ Failed to parse WebSocket message:', error, event.data);
    }
  }

  private attemptReconnect() {
    if (this.destroyed || this.status.reconnecting) {
      return;
    }

    if (this.status.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      this.status.lastError = 'Max reconnection attempts reached';
      this.notifyConnectionHandlers();
      return;
    }

    this.status.reconnecting = true;
    this.status.reconnectAttempts++;
    this.notifyConnectionHandlers();

    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(1.5, this.status.reconnectAttempts - 1),
      30000
    );

    console.log(`🔄 Attempting to reconnect in ${delay}ms (attempt ${this.status.reconnectAttempts}/${this.config.maxReconnectAttempts})`);

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
        this.status.reconnecting = false;
        this.notifyConnectionHandlers();
      } catch (error) {
        console.error('❌ Reconnection failed:', error);
        this.status.reconnecting = false;
        this.attemptReconnect();
      }
    }, delay);
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.send({
        type: 'heartbeat',
        data: { timestamp: new Date().toISOString() },
        timestamp: new Date().toISOString()
      });
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private sendHeartbeatResponse() {
    this.send({
      type: 'heartbeat_response',
      data: { timestamp: new Date().toISOString() },
      timestamp: new Date().toISOString()
    });
  }

  private flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  private cleanup() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
      }
    }
    
    this.ws = null;
  }

  private notifyConnectionHandlers() {
    this.connectionHandlers.forEach(handler => {
      try {
        handler({ ...this.status });
      } catch (error) {
        console.error('Error in connection handler:', error);
      }
    });
  }

  send(message: WebSocketMessage): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      // Queue message for when connection is restored
      this.messageQueue.push(message);
      console.log(`📤 Message queued: ${message.type}`);
      return false;
    }

    try {
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('❌ Failed to send WebSocket message:', error);
      this.messageQueue.push(message);
      return false;
    }
  }

  on(eventType: string, handler: EventHandler): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    
    this.eventHandlers.get(eventType)!.push(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = this.eventHandlers.get(eventType);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) {
          handlers.splice(index, 1);
        }
      }
    };
  }

  onConnection(handler: ConnectionHandler): () => void {
    this.connectionHandlers.push(handler);
    
    // Call immediately with current status
    handler({ ...this.status });
    
    // Return unsubscribe function
    return () => {
      const index = this.connectionHandlers.indexOf(handler);
      if (index > -1) {
        this.connectionHandlers.splice(index, 1);
      }
    };
  }

  getStatus(): ConnectionStatus {
    return { ...this.status };
  }

  isConnected(): boolean {
    return this.status.connected;
  }

  disconnect() {
    this.destroyed = false; // Allow reconnection
    this.cleanup();
    this.status.connected = false;
    this.status.reconnecting = false;
    this.notifyConnectionHandlers();
  }

  destroy() {
    this.destroyed = true;
    this.cleanup();
    this.eventHandlers.clear();
    this.connectionHandlers.length = 0;
    this.messageQueue.length = 0;
  }
}

// Factory function for creating WebSocket connections
export function createWebSocketService(config: WebSocketConfig): EnhancedWebSocketService {
  return new EnhancedWebSocketService(config);
}

// Predefined services for common FraudGuard endpoints
export class FraudGuardWebSocketService {
  private static instance: FraudGuardWebSocketService;
  private services: Map<string, EnhancedWebSocketService> = new Map();

  private constructor() {}

  static getInstance(): FraudGuardWebSocketService {
    if (!FraudGuardWebSocketService.instance) {
      FraudGuardWebSocketService.instance = new FraudGuardWebSocketService();
    }
    return FraudGuardWebSocketService.instance;
  }

  getAlertsService(): EnhancedWebSocketService {
    if (!this.services.has('alerts')) {
      const service = createWebSocketService({
        url: `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'}/alerts`,
        reconnectInterval: 3000,
        maxReconnectAttempts: 15,
      });
      this.services.set('alerts', service);
    }
    return this.services.get('alerts')!;
  }

  getTransactionsService(): EnhancedWebSocketService {
    if (!this.services.has('transactions')) {
      const service = createWebSocketService({
        url: `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'}/transactions`,
        reconnectInterval: 2000,
        maxReconnectAttempts: 20,
      });
      this.services.set('transactions', service);
    }
    return this.services.get('transactions')!;
  }

  getMetricsService(): EnhancedWebSocketService {
    if (!this.services.has('metrics')) {
      const service = createWebSocketService({
        url: `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'}/metrics`,
        heartbeatInterval: 10000,
        reconnectInterval: 5000,
      });
      this.services.set('metrics', service);
    }
    return this.services.get('metrics')!;
  }

  destroyAll() {
    this.services.forEach(service => service.destroy());
    this.services.clear();
  }
}

export default EnhancedWebSocketService;