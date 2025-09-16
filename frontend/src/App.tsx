import React, { createContext, useState, useEffect } from 'react';
import GraphVisualizer from './components/GraphVisualizer';
import { Graph } from './types';
import { useWebSocket } from './hooks/useWebSocket';

export const GraphContext = createContext<{ graph: Graph; updateGraph: (newGraph: Graph) => void }>({
  graph: { nodes: [], edges: [] },
  updateGraph: () => {},
});

const App = () => {
  const [graph, setGraph] = useState<Graph>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  console.log('App component is mounting');
  
  // Commented out WebSocket for now since it might cause issues
  // const { data } = useWebSocket(process.env.REACT_APP_WS_URL + '/graph/1');

  // Load some sample data for demonstration
  useEffect(() => {
    console.log('useEffect running - loading sample data');
    const sampleGraph: Graph = {
      nodes: [
        { id: "user1", label: "User 1", features: [0.1, 0.2, 0.3], riskScore: 0.7 },
        { id: "user2", label: "User 2", features: [0.4, 0.5, 0.6], riskScore: 0.3 },
        { id: "user3", label: "User 3", features: [0.7, 0.8, 0.9], riskScore: 0.9 }
      ],
      edges: [
        { source: "user1", target: "user2", weight: 5 },
        { source: "user2", target: "user3", weight: 3 }
      ]
    };
    setGraph(sampleGraph);
  }, []);

  const handleNodeClick = async (nodeId: string) => {
    setLoading(true);
    setError(null);
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/graph/expand/${nodeId}`);
      if (!res.ok) throw new Error('Fetch failed');
      const expanded = await res.json();
      setGraph(expanded);
    } catch (error) {
      console.error('Error fetching subgraph:', error);
      setError('Failed to fetch graph data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>FraudGuard-360 Dashboard</h1>
      {error && (
        <div style={{ background: '#ffebee', color: '#c62828', padding: '10px', marginBottom: '20px', borderRadius: '4px' }}>
          Error: {error}
        </div>
      )}
      {loading && <div style={{ color: '#1976d2', marginBottom: '20px' }}>Loading...</div>}
      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '20px' }}>
        <GraphContext.Provider value={{ graph, updateGraph: setGraph }}>
          <GraphVisualizer graph={graph} onNodeClick={handleNodeClick} />
        </GraphContext.Provider>
      </div>
    </div>
  );
};

export default App;
