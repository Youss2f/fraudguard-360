/**
 * FRAUD DETECTION DASHBOARD DEMO
 * Professional demonstration of real-time fraud detection capabilities
 */

import React, { useState, useEffect } from 'react';

interface FraudAlert {
  id: string;
  phoneNumber: string;
  riskScore: number;
  location: string;
  description: string;
  timestamp: string;
}

const FraudDashboardDemo: React.FC = () => {
  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [metrics, setMetrics] = useState({
    totalCalls: 0,
    fraudAlerts: 0,
    suspiciousCalls: 0,
    blockedCalls: 0
  });

  useEffect(() => {
    // Simulate real-time fraud detection
    const generateAlert = () => {
      const fraudAlert: FraudAlert = {
        id: `alert_${Date.now()}`,
        phoneNumber: `+${Math.floor(Math.random() * 90) + 10}-${Math.floor(Math.random() * 900) + 100}-${Math.floor(Math.random() * 900) + 100}-${Math.floor(Math.random() * 9000) + 1000}`,
        riskScore: Math.floor(Math.random() * 60) + 40,
        location: ['Lagos, Nigeria', 'Mumbai, India', 'Manila, Philippines', 'Karachi, Pakistan'][Math.floor(Math.random() * 4)],
        description: ['Suspicious calling pattern detected', 'Burst dialing activity', 'Location spoofing suspected', 'High-risk destination'][Math.floor(Math.random() * 4)],
        timestamp: new Date().toLocaleTimeString()
      };

      setAlerts(prev => [fraudAlert, ...prev.slice(0, 9)]);
      setMetrics(prev => ({
        totalCalls: prev.totalCalls + Math.floor(Math.random() * 100) + 50,
        fraudAlerts: prev.fraudAlerts + 1,
        suspiciousCalls: prev.suspiciousCalls + Math.floor(Math.random() * 5) + 1,
        blockedCalls: prev.blockedCalls + Math.floor(Math.random() * 3) + 1
      }));
    };

    // Generate initial data
    setMetrics({
      totalCalls: 47834,
      fraudAlerts: 156,
      suspiciousCalls: 892,
      blockedCalls: 334
    });

    // Generate alerts every 3-7 seconds
    const interval = setInterval(generateAlert, 3000 + Math.random() * 4000);
    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (score: number) => {
    if (score >= 80) return '#d32f2f';
    if (score >= 60) return '#f57c00';
    if (score >= 40) return '#fbc02d';
    return '#388e3c';
  };

  const cardStyle = {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    margin: '10px'
  };

  const headerStyle = {
    background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
    color: 'white',
    padding: '20px',
    textAlign: 'center' as const,
    fontSize: '24px',
    fontWeight: 'bold'
  };

  return (
    <div style={{ backgroundColor: '#f5f5f5', minHeight: '100vh', fontFamily: 'Arial, sans-serif' }}>
      {/* Header */}
      <div style={headerStyle}>
        🛡️ FraudGuard 360 - Professional Fraud Detection Platform
      </div>

      {/* Metrics Cards */}
      <div style={{ display: 'flex', flexWrap: 'wrap', padding: '20px' }}>
        <div style={{...cardStyle, flex: '1', minWidth: '200px'}}>
          <h3 style={{ margin: '0 0 10px 0', color: '#1976d2' }}>📞 Total Calls</h3>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#1976d2' }}>
            {metrics.totalCalls.toLocaleString()}
          </div>
        </div>

        <div style={{...cardStyle, flex: '1', minWidth: '200px'}}>
          <h3 style={{ margin: '0 0 10px 0', color: '#d32f2f' }}>⚠️ Fraud Alerts</h3>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#d32f2f' }}>
            {metrics.fraudAlerts}
          </div>
        </div>

        <div style={{...cardStyle, flex: '1', minWidth: '200px'}}>
          <h3 style={{ margin: '0 0 10px 0', color: '#f57c00' }}>🔍 Suspicious</h3>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#f57c00' }}>
            {metrics.suspiciousCalls}
          </div>
        </div>

        <div style={{...cardStyle, flex: '1', minWidth: '200px'}}>
          <h3 style={{ margin: '0 0 10px 0', color: '#388e3c' }}>🚫 Blocked</h3>
          <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#388e3c' }}>
            {metrics.blockedCalls}
          </div>
        </div>
      </div>

      {/* Live Alerts */}
      <div style={{...cardStyle, margin: '20px'}}>
        <h2 style={{ margin: '0 0 20px 0', color: '#1976d2' }}>
          🚨 Live Fraud Alerts ({alerts.length})
        </h2>
        
        {alerts.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#666', padding: '40px' }}>
            Monitoring for fraud patterns... System will detect and display alerts in real-time.
          </div>
        ) : (
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {alerts.map(alert => (
              <div key={alert.id} style={{
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                padding: '15px',
                marginBottom: '10px',
                borderLeft: `4px solid ${getRiskColor(alert.riskScore)}`
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 'bold', fontSize: '16px' }}>
                      {alert.phoneNumber}
                    </div>
                    <div style={{ color: '#666', marginTop: '5px' }}>
                      📍 {alert.location} • {alert.timestamp}
                    </div>
                    <div style={{ marginTop: '8px' }}>
                      {alert.description}
                    </div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ 
                      fontSize: '20px', 
                      fontWeight: 'bold', 
                      color: getRiskColor(alert.riskScore) 
                    }}>
                      {alert.riskScore}%
                    </div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      RISK SCORE
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Network Visualization Placeholder */}
      <div style={{...cardStyle, margin: '20px'}}>
        <h2 style={{ margin: '0 0 20px 0', color: '#1976d2' }}>
          🌐 Network Graph Visualization
        </h2>
        <div style={{
          height: '300px',
          backgroundColor: '#f9f9f9',
          border: '2px dashed #ccc',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '8px'
        }}>
          <div style={{ textAlign: 'center', color: '#666' }}>
            <div style={{ fontSize: '48px', marginBottom: '10px' }}>🕷️</div>
            <div style={{ fontSize: '18px', fontWeight: 'bold' }}>Interactive Network Graph</div>
            <div style={{ marginTop: '10px' }}>
              Real-time fraud relationship mapping with Cytoscape.js visualization
            </div>
            <div style={{ marginTop: '10px', fontSize: '14px' }}>
              • Node clustering algorithms • Risk-based color coding • Interactive drill-down
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderTop: '1px solid #e0e0e0',
        padding: '15px 20px',
        textAlign: 'center',
        color: '#666',
        fontSize: '14px'
      }}>
        🛡️ Professional Fraud Detection System • Real-time Monitoring Active • Interview-Ready Demonstration Platform
      </div>
    </div>
  );
};

export default FraudDashboardDemo;