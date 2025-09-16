import React from 'react';
import { Graph } from '../types';

interface Props {
  graph: Graph;
  onNodeClick: (nodeId: string) => void;
}

const GraphVisualizer: React.FC<Props> = ({ graph, onNodeClick }) => {
  console.log('GraphVisualizer rendering with graph:', graph);

  // Add a fallback div to ensure something renders
  if (!graph || graph.nodes.length === 0) {
    return (
      <div style={{ 
        width: '100%', 
        height: '600px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: '#f5f5f5',
        border: '2px dashed #ccc',
        color: '#666'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h3>No Graph Data</h3>
          <p>Waiting for graph data to load...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '600px', background: '#f9f9f9', border: '1px solid #ddd', borderRadius: '8px' }}>
      <h3 style={{ padding: '20px', margin: 0, background: '#333', color: 'white' }}>Graph Visualization</h3>
      <div style={{ padding: '20px' }}>
        <div style={{ marginBottom: '20px' }}>
          <strong>Nodes ({graph.nodes.length}):</strong>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
            {graph.nodes.map(node => (
              <div 
                key={node.id}
                onClick={() => onNodeClick(node.id)}
                style={{
                  padding: '10px 15px',
                  background: node.riskScore > 0.5 ? '#ffcdd2' : '#c8e6c9',
                  border: `2px solid ${node.riskScore > 0.5 ? '#f44336' : '#4caf50'}`,
                  borderRadius: '20px',
                  cursor: 'pointer',
                  transition: 'transform 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
              >
                <strong>{node.label}</strong><br/>
                <small>Risk: {(node.riskScore * 100).toFixed(0)}%</small>
              </div>
            ))}
          </div>
        </div>
        
        <div>
          <strong>Connections ({graph.edges.length}):</strong>
          <div style={{ marginTop: '10px' }}>
            {graph.edges.map((edge, idx) => (
              <div key={idx} style={{ 
                padding: '8px', 
                background: '#e3f2fd', 
                margin: '5px 0', 
                borderRadius: '4px',
                border: '1px solid #2196f3'
              }}>
                {edge.source} ↔ {edge.target} (Weight: {edge.weight})
              </div>
            ))}
          </div>
        </div>
        
        <div style={{ marginTop: '20px', padding: '15px', background: '#fff3e0', borderRadius: '4px', border: '1px solid #ff9800' }}>
          <strong>📊 Note:</strong> This is a simplified view. The full cytoscape visualization will be enabled once all dependencies are confirmed working.
        </div>
      </div>
    </div>
  );
};

export default GraphVisualizer;
