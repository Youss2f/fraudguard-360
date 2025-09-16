/**
 * Real-time WebSocket Service for Fraud Detection Dashboard
 * Simulates live fraud alerts, CDR streaming, and dashboard updates
 */

export interface FraudAlert {
  id: string;
  timestamp: string;
  phoneNumber: string;
  riskScore: number;
  alertType: 'FRAUD_DETECTED' | 'SUSPICIOUS_PATTERN' | 'HIGH_VOLUME' | 'LOCATION_ANOMALY';
  description: string;
  location: string;
  status: 'NEW' | 'INVESTIGATING' | 'CONFIRMED' | 'FALSE_POSITIVE';
  assignedTo?: string;
  metadata: {
    callDuration: number;
    destinationNumber: string;
    callCount: number;
    suspiciousPatterns: string[];
  };
}

export interface RealTimeMetrics {
  timestamp: string;
  totalCalls: number;
  fraudAlertsCount: number;
  suspiciousCallsCount: number;
  blockedCallsCount: number;
  riskDistribution: {
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
  topRiskCountries: Array<{
    country: string;
    count: number;
    avgRiskScore: number;
  }>;
  networkStats: {
    totalNodes: number;
    totalEdges: number;
    suspiciousConnections: number;
    isolatedNodes: number;
  };
}

class WebSocketService {
  private listeners: Map<string, Array<(data: any) => void>> = new Map();
  private isConnected = false;
  private intervalIds: NodeJS.Timeout[] = [];

  // Simulate WebSocket connection
  connect(): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(() => {
        this.isConnected = true;
        console.log('🔌 WebSocket connected to fraud detection system');
        this.startDataStreaming();
        resolve();
      }, 1000);
    });
  }

  disconnect(): void {
    this.isConnected = false;
    this.intervalIds.forEach(id => clearInterval(id));
    this.intervalIds = [];
    console.log('🔌 WebSocket disconnected');
  }

  // Subscribe to specific data streams
  subscribe(channel: string, callback: (data: any) => void): void {
    if (!this.listeners.has(channel)) {
      this.listeners.set(channel, []);
    }
    this.listeners.get(channel)!.push(callback);
  }

  unsubscribe(channel: string, callback: (data: any) => void): void {
    const channelListeners = this.listeners.get(channel);
    if (channelListeners) {
      const index = channelListeners.indexOf(callback);
      if (index > -1) {
        channelListeners.splice(index, 1);
      }
    }
  }

  private emit(channel: string, data: any): void {
    const channelListeners = this.listeners.get(channel);
    if (channelListeners) {
      channelListeners.forEach(callback => callback(data));
    }
  }

  private startDataStreaming(): void {
    // Stream fraud alerts every 3-8 seconds
    const alertInterval = setInterval(() => {
      if (this.isConnected && Math.random() < 0.7) {
        this.emit('fraud-alerts', this.generateFraudAlert());
      }
    }, 3000 + Math.random() * 5000);

    // Stream real-time metrics every 5 seconds
    const metricsInterval = setInterval(() => {
      if (this.isConnected) {
        this.emit('real-time-metrics', this.generateRealTimeMetrics());
      }
    }, 5000);

    // Stream network updates every 10-15 seconds
    const networkInterval = setInterval(() => {
      if (this.isConnected && Math.random() < 0.6) {
        this.emit('network-updates', this.generateNetworkUpdate());
      }
    }, 10000 + Math.random() * 5000);

    // Stream CDR records continuously (every 1-3 seconds)
    const cdrInterval = setInterval(() => {
      if (this.isConnected) {
        this.emit('cdr-stream', this.generateCDRRecord());
      }
    }, 1000 + Math.random() * 2000);

    this.intervalIds = [alertInterval, metricsInterval, networkInterval, cdrInterval];
  }

  private generateFraudAlert(): FraudAlert {
    const alertTypes = ['FRAUD_DETECTED', 'SUSPICIOUS_PATTERN', 'HIGH_VOLUME', 'LOCATION_ANOMALY'] as const;
    const locations = [
      'Lagos, Nigeria', 'Mumbai, India', 'Manila, Philippines', 'Karachi, Pakistan',
      'Dhaka, Bangladesh', 'Cairo, Egypt', 'Istanbul, Turkey', 'São Paulo, Brazil',
      'Mexico City, Mexico', 'Bangkok, Thailand'
    ];
    
    const suspiciousPatterns = [
      'Sequential dialing pattern detected',
      'Burst calling behavior',
      'Location spoofing detected',
      'Unusual call duration pattern',
      'High-frequency international calls',
      'Call forwarding chain detected',
      'Caller ID manipulation',
      'Roaming fraud indicators'
    ];

    const alertType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
    const riskScore = 60 + Math.random() * 40; // High risk alerts (60-100%)
    
    return {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      phoneNumber: this.generatePhoneNumber(),
      riskScore: Math.round(riskScore),
      alertType,
      description: this.generateAlertDescription(alertType),
      location: locations[Math.floor(Math.random() * locations.length)],
      status: 'NEW',
      metadata: {
        callDuration: Math.floor(Math.random() * 300) + 30,
        destinationNumber: this.generatePhoneNumber(),
        callCount: Math.floor(Math.random() * 50) + 1,
        suspiciousPatterns: this.getRandomElements(suspiciousPatterns, 1 + Math.floor(Math.random() * 3))
      }
    };
  }

  private generateAlertDescription(alertType: string): string {
    const descriptions = {
      'FRAUD_DETECTED': [
        'Confirmed fraudulent activity pattern detected',
        'Known fraud signature matched',
        'Multi-stage fraud operation identified',
        'Sophisticated fraud scheme detected'
      ],
      'SUSPICIOUS_PATTERN': [
        'Unusual calling behavior detected',
        'Abnormal traffic pattern identified',
        'Potential fraud indicators present',
        'Suspicious network activity detected'
      ],
      'HIGH_VOLUME': [
        'Excessive call volume detected',
        'Bulk calling operation identified',
        'Call center fraud suspected',
        'Mass dialing activity detected'
      ],
      'LOCATION_ANOMALY': [
        'Impossible travel detected',
        'Geographic inconsistency identified',
        'Location spoofing suspected',
        'Roaming fraud indicators present'
      ]
    };

    const typeDescriptions = descriptions[alertType as keyof typeof descriptions] || ['Unknown alert type'];
    return typeDescriptions[Math.floor(Math.random() * typeDescriptions.length)];
  }

  private generateRealTimeMetrics(): RealTimeMetrics {
    const baseMetrics = {
      totalCalls: 50000 + Math.floor(Math.random() * 20000),
      fraudAlertsCount: 150 + Math.floor(Math.random() * 100),
      suspiciousCallsCount: 800 + Math.floor(Math.random() * 400),
      blockedCallsCount: 300 + Math.floor(Math.random() * 200)
    };

    return {
      timestamp: new Date().toISOString(),
      ...baseMetrics,
      riskDistribution: {
        low: 70 + Math.random() * 15,
        medium: 20 + Math.random() * 10,
        high: 8 + Math.random() * 5,
        critical: 2 + Math.random() * 3
      },
      topRiskCountries: [
        { country: 'Nigeria', count: 45 + Math.floor(Math.random() * 20), avgRiskScore: 75 + Math.random() * 20 },
        { country: 'India', count: 38 + Math.floor(Math.random() * 15), avgRiskScore: 70 + Math.random() * 15 },
        { country: 'Philippines', count: 32 + Math.floor(Math.random() * 12), avgRiskScore: 68 + Math.random() * 18 },
        { country: 'Pakistan', count: 28 + Math.floor(Math.random() * 10), avgRiskScore: 65 + Math.random() * 20 },
        { country: 'Bangladesh', count: 22 + Math.floor(Math.random() * 8), avgRiskScore: 63 + Math.random() * 17 }
      ],
      networkStats: {
        totalNodes: 1500 + Math.floor(Math.random() * 500),
        totalEdges: 3200 + Math.floor(Math.random() * 800),
        suspiciousConnections: 120 + Math.floor(Math.random() * 80),
        isolatedNodes: 45 + Math.floor(Math.random() * 30)
      }
    };
  }

  private generateNetworkUpdate(): any {
    return {
      type: 'network-update',
      timestamp: new Date().toISOString(),
      action: Math.random() < 0.6 ? 'node-added' : 'edge-added',
      data: Math.random() < 0.6 ? this.generateNewNode() : this.generateNewEdge()
    };
  }

  private generateNewNode(): any {
    return {
      id: `node_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      label: this.generatePhoneNumber(),
      riskScore: Math.random() * 100,
      location: this.getRandomLocation(),
      callCount: Math.floor(Math.random() * 100) + 1
    };
  }

  private generateNewEdge(): any {
    return {
      id: `edge_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      source: `existing_node_${Math.floor(Math.random() * 100)}`,
      target: `existing_node_${Math.floor(Math.random() * 100)}`,
      callDuration: Math.floor(Math.random() * 300) + 10,
      suspiciousScore: Math.random() * 100
    };
  }

  private generateCDRRecord(): any {
    return {
      id: `cdr_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`,
      timestamp: new Date().toISOString(),
      callerNumber: this.generatePhoneNumber(),
      calleeNumber: this.generatePhoneNumber(),
      duration: Math.floor(Math.random() * 300) + 5,
      location: this.getRandomLocation(),
      riskScore: Math.random() * 100,
      callType: Math.random() < 0.7 ? 'domestic' : 'international'
    };
  }

  private generatePhoneNumber(): string {
    const countryCode = Math.random() < 0.6 ? '+1' : this.getRandomCountryCode();
    const area = Math.floor(Math.random() * 900) + 100;
    const exchange = Math.floor(Math.random() * 900) + 100;
    const number = Math.floor(Math.random() * 9000) + 1000;
    return `${countryCode}-${area}-${exchange}-${number}`;
  }

  private getRandomCountryCode(): string {
    const codes = ['+234', '+91', '+63', '+92', '+880', '+20', '+90', '+55', '+52', '+66'];
    return codes[Math.floor(Math.random() * codes.length)];
  }

  private getRandomLocation(): string {
    const locations = [
      'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
      'Lagos, Nigeria', 'Mumbai, India', 'Manila, Philippines', 'Karachi, Pakistan',
      'London, UK', 'Berlin, Germany', 'Paris, France', 'Tokyo, Japan'
    ];
    return locations[Math.floor(Math.random() * locations.length)];
  }

  private getRandomElements<T>(array: T[], count: number): T[] {
    const shuffled = [...array].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }
}

// Export singleton instance
export const webSocketService = new WebSocketService();
export default webSocketService;