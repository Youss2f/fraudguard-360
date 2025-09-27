import { useState, useEffect } from 'react';
import { useWebSocket } from './useWebSocket';

export interface RealTimeData {
  alerts: any[];
  kpis: {
    totalTransactions: number;
    fraudAlerts: number;
    riskScore: number;
    successRate: number;
  };
  transactions: any[];
  charts: {
    hourlyFraud: any[];
    transactionVolume: any[];
    riskDistribution: any[];
  };
}

export const useRealTimeData = () => {
  const [data, setData] = useState<RealTimeData>({
    alerts: [],
    kpis: {
      totalTransactions: 0,
      fraudAlerts: 0,
      riskScore: 0,
      successRate: 0
    },
    transactions: [],
    charts: {
      hourlyFraud: [],
      transactionVolume: [],
      riskDistribution: []
    }
  });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const webSocket = useWebSocket('ws://localhost:8000/ws');

  useEffect(() => {
    // Initial data fetch
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        const [alertsRes, kpisRes, transactionsRes, chartsRes] = await Promise.all([
          fetch('http://localhost:8000/dashboard/alerts'),
          fetch('http://localhost:8000/dashboard/kpis'),
          fetch('http://localhost:8000/dashboard/transactions'),
          fetch('http://localhost:8000/dashboard/charts')
        ]);

        const alerts = await alertsRes.json();
        const kpis = await kpisRes.json();
        const transactions = await transactionsRes.json();
        const charts = await chartsRes.json();

        setData({ alerts, kpis, transactions, charts });
        setError(null);
      } catch (err) {
        setError('Failed to fetch initial data');
        console.error('Error fetching initial data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (webSocket.data) {
      try {
        const message = webSocket.data;
        
        switch (message.type) {
          case 'kpi_update':
            setData(prev => ({ ...prev, kpis: message.data }));
            break;
          case 'new_alert':
            setData(prev => ({ 
              ...prev, 
              alerts: [message.data, ...prev.alerts.slice(0, 49)] 
            }));
            break;
          case 'new_transaction':
            setData(prev => ({ 
              ...prev, 
              transactions: [message.data, ...prev.transactions.slice(0, 99)] 
            }));
            break;
          case 'chart_update':
            setData(prev => ({ 
              ...prev, 
              charts: { ...prev.charts, ...message.data } 
            }));
            break;
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    }
  }, [webSocket.data]);

  return {
    data,
    loading,
    error,
    isConnected: webSocket.ws?.readyState === WebSocket.OPEN
  };
};