export interface Node {
  id: string;
  label: string;
  features: number[];  // E.g., [call_count, duration_avg, risk_score]
  riskScore: number;
}

export interface Edge {
  source: string;
  target: string;
  weight: number;  // E.g., call duration or frequency
}

export interface Graph {
  nodes: Node[];
  edges: Edge[];
}

export interface GNNResult {
  subscriberId: string;
  riskScore: number;
  communityId: string;  // Detected fraud community
  explanation: string;  // Basic XAI info
}
