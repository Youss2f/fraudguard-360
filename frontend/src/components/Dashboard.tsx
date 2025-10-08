import React, { useState, useEffect } from 'react';
import './Dashboard.css';

interface DashboardKPIs {
  totalTransactions: number;
  fraudAlerts: number;
  riskScore: number;
  successRate: number;
  timestamp: number;
}

interface Alert {
  id: string;
  type: string;
  severity: 'critical' | 'high' | 'medium';
  amount: number;
  customer_id: string;
  timestamp: string;
  status: 'new' | 'investigating' | 'resolved';
}

interface Transaction {
  id: string;
  amount: number;
  customer_id: string;
  merchant: string;
  risk_score: number;
  status: 'approved' | 'declined' | 'pending';
  timestamp: string;
}

interface ChartData {
  hourlyFraud: Array<{
    time: string;
    fraudCount: number;
    totalTransactions: number;
  }>;
  transactionVolume: Array<{
    day: string;
    volume: number;
    fraudulent: number;
  }>;
  riskDistribution: Array<{
    range: string;
    count: number;
  }>;
}

const Dashboard: React.FC = () => {
  const [kpis, setKpis] = useState<DashboardKPIs | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');

  // Fetch dashboard data
  const fetchKPIs = async () => {
    try {
      const response = await fetch('/dashboard/kpis');
      if (!response.ok) throw new Error('Failed to fetch KPIs');
      const data = await response.json();
      setKpis(data);
    } catch (error) {
      console.error('Error fetching KPIs:', error);
      setError('Failed to load KPIs');
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch('/dashboard/alerts');
      if (!response.ok) throw new Error('Failed to fetch alerts');
      const data = await response.json();
      setAlerts(data.slice(0, 10)); // Show top 10 alerts
    } catch (error) {
      console.error('Error fetching alerts:', error);
      setError('Failed to load alerts');
    }
  };

  const fetchTransactions = async () => {
    try {
      const response = await fetch('/dashboard/transactions');
      if (!response.ok) throw new Error('Failed to fetch transactions');
      const data = await response.json();
      setTransactions(data.slice(0, 10)); // Show top 10 transactions
    } catch (error) {
      console.error('Error fetching transactions:', error);
      setError('Failed to load transactions');
    }
  };

  const fetchChartData = async () => {
    try {
      const response = await fetch('/dashboard/charts');
      if (!response.ok) throw new Error('Failed to fetch chart data');
      const data = await response.json();
      setChartData(data);
    } catch (error) {
      console.error('Error fetching chart data:', error);
      setError('Failed to load chart data');
    }
  };

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          fetchKPIs(),
          fetchAlerts(),
          fetchTransactions(),
          fetchChartData()
        ]);
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Set up real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchKPIs();
      fetchAlerts();
      fetchTransactions();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#dc3545';
      case 'high': return '#fd7e14';
      case 'medium': return '#ffc107';
      default: return '#6c757d';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return '#28a745';
      case 'declined': return '#dc3545';
      case 'pending': return '#ffc107';
      default: return '#6c757d';
    }
  };

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <h3>Error Loading Dashboard</h3>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>FraudGuard 360 Dashboard</h1>
        <div className="dashboard-controls">
          <select 
            value={selectedTimeRange} 
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="time-range-selector"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <button 
            onClick={() => window.location.reload()} 
            className="refresh-button"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      {kpis && (
        <div className="kpi-grid">
          <div className="kpi-card transactions">
            <div className="kpi-header">
              <h3>Total Transactions</h3>
              <span className="kpi-icon">📊</span>
            </div>
            <div className="kpi-value">{kpis.totalTransactions.toLocaleString()}</div>
            <div className="kpi-change positive">+2.5% from yesterday</div>
          </div>

          <div className="kpi-card alerts">
            <div className="kpi-header">
              <h3>Fraud Alerts</h3>
              <span className="kpi-icon">🚨</span>
            </div>
            <div className="kpi-value">{kpis.fraudAlerts}</div>
            <div className="kpi-change negative">+12% from yesterday</div>
          </div>

          <div className="kpi-card risk">
            <div className="kpi-header">
              <h3>Average Risk Score</h3>
              <span className="kpi-icon">⚠️</span>
            </div>
            <div className="kpi-value">{formatPercentage(kpis.riskScore)}</div>
            <div className="kpi-change positive">-5% from yesterday</div>
          </div>

          <div className="kpi-card success">
            <div className="kpi-header">
              <h3>Success Rate</h3>
              <span className="kpi-icon">✅</span>
            </div>
            <div className="kpi-value">{formatPercentage(kpis.successRate)}</div>
            <div className="kpi-change positive">+0.2% from yesterday</div>
          </div>
        </div>
      )}

      <div className="dashboard-content">
        {/* Charts Section */}
        {chartData && (
          <div className="charts-section">
            <div className="chart-container">
              <h3>Hourly Fraud Detection</h3>
              <div className="chart-placeholder">
                <div className="chart-bars">
                  {chartData.hourlyFraud.map((data, index) => (
                    <div key={index} className="chart-bar">
                      <div 
                        className="bar" 
                        style={{ 
                          height: `${(data.fraudCount / 25) * 100}%`,
                          backgroundColor: data.fraudCount > 15 ? '#dc3545' : '#28a745'
                        }}
                      ></div>
                      <div className="bar-label">{data.time}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="chart-container">
              <h3>Risk Score Distribution</h3>
              <div className="risk-distribution">
                {chartData.riskDistribution.map((data, index) => (
                  <div key={index} className="risk-item">
                    <span className="risk-range">{data.range}</span>
                    <div className="risk-bar">
                      <div 
                        className="risk-fill" 
                        style={{ 
                          width: `${(data.count / 12000) * 100}%`,
                          backgroundColor: index > 2 ? '#dc3545' : '#28a745'
                        }}
                      ></div>
                    </div>
                    <span className="risk-count">{data.count.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Alerts and Transactions */}
        <div className="data-tables">
          <div className="table-container">
            <h3>Recent Fraud Alerts</h3>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Alert ID</th>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Amount</th>
                    <th>Customer</th>
                    <th>Status</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.id}>
                      <td className="alert-id">{alert.id}</td>
                      <td>{alert.type}</td>
                      <td>
                        <span 
                          className="severity-badge"
                          style={{ backgroundColor: getSeverityColor(alert.severity) }}
                        >
                          {alert.severity}
                        </span>
                      </td>
                      <td>{formatCurrency(alert.amount)}</td>
                      <td>{alert.customer_id}</td>
                      <td>
                        <span 
                          className="status-badge"
                          style={{ backgroundColor: getStatusColor(alert.status) }}
                        >
                          {alert.status}
                        </span>
                      </td>
                      <td>{formatTimestamp(alert.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="table-container">
            <h3>Recent Transactions</h3>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Amount</th>
                    <th>Customer</th>
                    <th>Merchant</th>
                    <th>Risk Score</th>
                    <th>Status</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((transaction) => (
                    <tr key={transaction.id}>
                      <td className="transaction-id">{transaction.id}</td>
                      <td>{formatCurrency(transaction.amount)}</td>
                      <td>{transaction.customer_id}</td>
                      <td>{transaction.merchant}</td>
                      <td>
                        <span 
                          className="risk-score"
                          style={{ 
                            color: transaction.risk_score > 0.7 ? '#dc3545' : 
                                   transaction.risk_score > 0.3 ? '#ffc107' : '#28a745'
                          }}
                        >
                          {formatPercentage(transaction.risk_score)}
                        </span>
                      </td>
                      <td>
                        <span 
                          className="status-badge"
                          style={{ backgroundColor: getStatusColor(transaction.status) }}
                        >
                          {transaction.status}
                        </span>
                      </td>
                      <td>{formatTimestamp(transaction.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time status indicator */}
      <div className="status-indicator">
        <span className="status-dot active"></span>
        <span>Live Data</span>
        <span className="last-update">
          Last updated: {new Date().toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
};

export default Dashboard;