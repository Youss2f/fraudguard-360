import React, { useState, useEffect } from 'react';
import './Dashboard.css';

interface Transaction {
  id: string;
  amount: number;
  userId: string;
  fraudProbability: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  status: 'APPROVED' | 'FLAGGED' | 'BLOCKED';
  timestamp: string;
}

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState({
    totalTransactions: 12547,
    flaggedTransactions: 89,
    blockedTransactions: 23,
    avgFraudScore: 12.3,
    alertsLast24h: 156
  });

  const [transactions, setTransactions] = useState<Transaction[]>([
    {
      id: 'tx_001',
      amount: 15000,
      userId: 'user_123',
      fraudProbability: 0.87,
      riskLevel: 'CRITICAL',
      timestamp: new Date().toISOString(),
      status: 'BLOCKED'
    },
    {
      id: 'tx_002',
      amount: 250,
      userId: 'user_789',
      fraudProbability: 0.15,
      riskLevel: 'LOW',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      status: 'APPROVED'
    },
    {
      id: 'tx_003',
      amount: 5000,
      userId: 'user_456',
      fraudProbability: 0.65,
      riskLevel: 'HIGH',
      timestamp: new Date(Date.now() - 600000).toISOString(),
      status: 'FLAGGED'
    }
  ]);

  const formatAmount = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const getRiskClass = (level: string) => {
    return `risk-${level.toLowerCase()}`;
  };

  const getStatusClass = (status: string) => {
    return `status-${status.toLowerCase()}`;
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>FraudGuard 360° Dashboard</h1>
        <p>Real-time fraud detection monitoring</p>
      </header>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Total Transactions</h3>
          <p className="metric-value">{metrics.totalTransactions.toLocaleString()}</p>
        </div>
        
        <div className="metric-card warning">
          <h3>Flagged Today</h3>
          <p className="metric-value">{metrics.flaggedTransactions}</p>
        </div>
        
        <div className="metric-card danger">
          <h3>Blocked Today</h3>
          <p className="metric-value">{metrics.blockedTransactions}</p>
        </div>
        
        <div className="metric-card info">
          <h3>Avg Fraud Score</h3>
          <p className="metric-value">{metrics.avgFraudScore}%</p>
        </div>
      </div>

      <div className="alert-bar">
        <strong>{metrics.alertsLast24h}</strong> fraud alerts in last 24h | 
        Processing time: <strong>127ms</strong> | 
        System status: <span className="status-healthy">Healthy</span>
      </div>

      <div className="transactions-section">
        <h2>Recent High-Risk Transactions</h2>
        <table className="transactions-table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Amount</th>
              <th>User ID</th>
              <th>Fraud Score</th>
              <th>Risk Level</th>
              <th>Status</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((tx) => (
              <tr key={tx.id}>
                <td className="mono">{tx.id}</td>
                <td className="amount">{formatAmount(tx.amount)}</td>
                <td>{tx.userId}</td>
                <td>
                  <div className="fraud-score">
                    <span>{(tx.fraudProbability * 100).toFixed(1)}%</span>
                    <div className="score-bar">
                      <div 
                        className={`score-fill ${getRiskClass(tx.riskLevel)}`}
                        style={{ width: `${tx.fraudProbability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </td>
                <td>
                  <span className={`badge ${getRiskClass(tx.riskLevel)}`}>
                    {tx.riskLevel}
                  </span>
                </td>
                <td>
                  <span className={`badge ${getStatusClass(tx.status)}`}>
                    {tx.status}
                  </span>
                </td>
                <td className="timestamp">
                  {new Date(tx.timestamp).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;