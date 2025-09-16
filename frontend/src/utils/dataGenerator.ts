// Simple data generator for FraudGuard-360 dashboard
// Using built-in Math.random() to avoid external dependencies

export interface CDRRecord {
  id: string;
  timestamp: Date;
  caller: string;
  callee: string;
  duration: number;
  location: {
    lat: number;
    lng: number;
    city: string;
    country: string;
  };
  fraudScore: number;
  fraudIndicators: string[];
  callType: 'voice' | 'sms' | 'data';
  cost: number;
}

export interface FraudAlert {
  id: string;
  timestamp: Date;
  type: 'suspicious_pattern' | 'high_risk_number' | 'unusual_location' | 'rate_anomaly';
  severity: 'low' | 'medium' | 'high' | 'critical';
  caller: string;
  callee?: string;
  description: string;
  riskScore: number;
  patterns: string[];
  investigation: {
    status: 'new' | 'investigating' | 'confirmed' | 'false_positive';
    assignedTo?: string;
    notes?: string;
  };
}

export interface NetworkNode {
  id: string;
  label: string;
  type: 'subscriber' | 'number' | 'location' | 'device';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  fraudScore: number;
  metadata: {
    callCount: number;
    totalDuration: number;
    uniqueContacts: number;
    recentActivity: Date;
    location?: string;
    deviceInfo?: string;
  };
}

export interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  type: 'frequent' | 'suspicious' | 'normal';
  interactions: number;
  lastContact: Date;
  fraudIndicators: string[];
}

class SimpleDataGenerator {
  private suspiciousNumbers: string[] = [];
  private knownFraudsters: string[] = [];
  private locations = [
    { city: 'Lagos', country: 'Nigeria', lat: 6.5244, lng: 3.3792 },
    { city: 'Mumbai', country: 'India', lat: 19.0760, lng: 72.8777 },
    { city: 'Manila', country: 'Philippines', lat: 14.5995, lng: 120.9842 },
    { city: 'Dhaka', country: 'Bangladesh', lat: 23.8103, lng: 90.4125 },
    { city: 'New York', country: 'USA', lat: 40.7128, lng: -74.0060 },
    { city: 'London', country: 'UK', lat: 51.5074, lng: -0.1278 },
    { city: 'Berlin', country: 'Germany', lat: 52.5200, lng: 13.4050 },
    { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },
  ];

  private names = ['John Smith', 'Sarah Johnson', 'Mike Davis', 'Lisa Brown', 'Tom Wilson', 'Emily Garcia'];

  constructor() {
    this.initializeData();
  }

  private initializeData() {
    this.suspiciousNumbers = Array.from({ length: 50 }, () => this.generatePhoneNumber());
    this.knownFraudsters = Array.from({ length: 20 }, () => this.generatePhoneNumber());
  }

  private generateUUID(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  private generatePhoneNumber(): string {
    const areaCode = Math.floor(Math.random() * 900) + 100;
    const exchange = Math.floor(Math.random() * 900) + 100;
    const number = Math.floor(Math.random() * 9000) + 1000;
    return `+1-${areaCode}-${exchange}-${number}`;
  }

  private randomChoice<T>(array: readonly T[]): T {
    return array[Math.floor(Math.random() * array.length)];
  }

  private randomFloat(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }

  private randomInt(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  private recentDate(days: number = 7): Date {
    const now = new Date();
    const msPerDay = 24 * 60 * 60 * 1000;
    return new Date(now.getTime() - Math.random() * days * msPerDay);
  }

  generateCDRRecord(): CDRRecord {
    const isFraudulent = Math.random() < 0.03;
    const isFromKnownFraudster = Math.random() < 0.01;
    
    const caller = isFromKnownFraudster 
      ? this.randomChoice(this.knownFraudsters)
      : this.generatePhoneNumber();
    
    const callee = Math.random() < 0.1 
      ? this.randomChoice(this.suspiciousNumbers)
      : this.generatePhoneNumber();

    const location = this.randomChoice(this.locations);
    const duration = this.randomInt(1, 3600);
    
    let fraudScore = 0;
    const fraudIndicators: string[] = [];
    
    if (isFraudulent || isFromKnownFraudster) {
      fraudScore = this.randomFloat(0.6, 1.0);
      
      if (duration < 10) fraudIndicators.push('short_duration');
      if (Math.random() < 0.3) fraudIndicators.push('unusual_time');
      if (Math.random() < 0.4) fraudIndicators.push('geographic_anomaly');
      if (Math.random() < 0.2) fraudIndicators.push('known_bad_number');
      if (Math.random() < 0.3) fraudIndicators.push('rate_anomaly');
    } else {
      fraudScore = this.randomFloat(0.0, 0.4);
    }

    return {
      id: this.generateUUID(),
      timestamp: this.recentDate(7),
      caller,
      callee,
      duration,
      location: {
        ...location,
        lat: location.lat + this.randomFloat(-0.1, 0.1),
        lng: location.lng + this.randomFloat(-0.1, 0.1),
      },
      fraudScore,
      fraudIndicators,
      callType: this.randomChoice(['voice', 'sms', 'data']),
      cost: this.randomFloat(0.01, 50.0),
    };
  }

  generateFraudAlert(): FraudAlert {
    const types = ['suspicious_pattern', 'high_risk_number', 'unusual_location', 'rate_anomaly'] as const;
    const severities = ['low', 'medium', 'high', 'critical'] as const;
    const type = this.randomChoice(types);
    
    let riskScore = this.randomFloat(0.5, 1.0);
    let severity: typeof severities[number] = 'low';
    
    if (riskScore > 0.9) severity = 'critical';
    else if (riskScore > 0.7) severity = 'high';
    else if (riskScore > 0.5) severity = 'medium';

    const descriptions = {
      suspicious_pattern: 'Unusual call pattern detected - high frequency calls to multiple numbers',
      high_risk_number: 'Call from number associated with previous fraud incidents',
      unusual_location: 'Call originating from high-risk geographical location',
      rate_anomaly: 'Abnormal call rate and duration patterns detected'
    };

    const patternSets = {
      suspicious_pattern: ['burst_calling', 'sequential_dialing', 'round_robin'],
      high_risk_number: ['blacklisted', 'previous_fraud', 'suspicious_registration'],
      unusual_location: ['high_risk_country', 'location_spoofing', 'travel_anomaly'],
      rate_anomaly: ['unusual_frequency', 'abnormal_duration', 'cost_anomaly']
    };

    return {
      id: this.generateUUID(),
      timestamp: this.recentDate(1),
      type,
      severity,
      caller: Math.random() < 0.3 
        ? this.randomChoice(this.knownFraudsters)
        : this.generatePhoneNumber(),
      callee: Math.random() < 0.5 ? this.generatePhoneNumber() : undefined,
      description: descriptions[type],
      riskScore,
      patterns: patternSets[type].slice(0, this.randomInt(1, 3)),
      investigation: {
        status: this.randomChoice(['new', 'investigating', 'confirmed', 'false_positive']),
        assignedTo: Math.random() < 0.7 ? this.randomChoice(this.names) : undefined,
        notes: Math.random() < 0.4 ? 'Investigation in progress...' : undefined,
      },
    };
  }

  generateNetworkGraph(nodeCount: number = 50, edgeCount: number = 100): { nodes: NetworkNode[], edges: NetworkEdge[] } {
    const nodes: NetworkNode[] = [];
    const edges: NetworkEdge[] = [];

    // Generate nodes
    for (let i = 0; i < nodeCount; i++) {
      const riskLevels = ['low', 'medium', 'high', 'critical'] as const;
      const weights = [0.6, 0.25, 0.1, 0.05];
      let riskLevel: typeof riskLevels[number] = 'low';
      
      const rand = Math.random();
      let cumulative = 0;
      for (let j = 0; j < weights.length; j++) {
        cumulative += weights[j];
        if (rand <= cumulative) {
          riskLevel = riskLevels[j];
          break;
        }
      }

      const fraudScoreRanges = {
        low: [0.0, 0.3] as const,
        medium: [0.3, 0.6] as const,
        high: [0.6, 0.8] as const,
        critical: [0.8, 1.0] as const,
      };

      const [min, max] = fraudScoreRanges[riskLevel];
      const fraudScore = this.randomFloat(min, max);

      nodes.push({
        id: `node-${i}`,
        label: this.generatePhoneNumber(),
        type: this.randomChoice(['subscriber', 'number', 'location', 'device']),
        riskLevel,
        fraudScore,
        metadata: {
          callCount: this.randomInt(1, 1000),
          totalDuration: this.randomInt(60, 50000),
          uniqueContacts: this.randomInt(1, 100),
          recentActivity: this.recentDate(30),
          location: this.randomChoice(this.locations).city,
          deviceInfo: this.randomChoice(['iPhone 14', 'Samsung Galaxy', 'Unknown', 'Burner Phone']),
        },
      });
    }

    // Generate edges
    for (let i = 0; i < edgeCount && i < nodeCount * (nodeCount - 1) / 2; i++) {
      const source = this.randomChoice(nodes);
      const target = this.randomChoice(nodes.filter(n => n.id !== source.id));
      
      if (edges.some(e => (e.source === source.id && e.target === target.id) || 
                          (e.source === target.id && e.target === source.id))) {
        continue;
      }

      const interactions = this.randomInt(1, 50);
      const isSuspicious = source.riskLevel === 'high' || target.riskLevel === 'high' || 
                          source.riskLevel === 'critical' || target.riskLevel === 'critical';

      edges.push({
        id: `edge-${i}`,
        source: source.id,
        target: target.id,
        weight: interactions,
        type: isSuspicious ? 'suspicious' : (interactions > 20 ? 'frequent' : 'normal'),
        interactions,
        lastContact: this.recentDate(7),
        fraudIndicators: isSuspicious ? ['frequent_contact', 'suspicious_timing'] : [],
      });
    }

    return { nodes, edges };
  }

  generateRealtimeMetrics() {
    return {
      totalCalls: this.randomInt(50000, 100000),
      fraudulentCalls: this.randomInt(500, 2000),
      activeCases: this.randomInt(20, 100),
      fraudRate: this.randomFloat(0.5, 3.5),
      avgRiskScore: this.randomFloat(0.1, 0.4),
      alertsToday: this.randomInt(10, 200),
      savedAmount: this.randomInt(10000, 500000),
      responseTime: this.randomFloat(1.2, 8.5),
    };
  }
}

export const fraudDataGenerator = new SimpleDataGenerator();