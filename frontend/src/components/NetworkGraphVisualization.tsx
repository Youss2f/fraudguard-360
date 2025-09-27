import React, { useEffect, useRef, useState } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape, { Core, ElementDefinition } from 'cytoscape';
import cola from 'cytoscape-cola';
import dagre from 'cytoscape-dagre';
import fcose from 'cytoscape-fcose';

import { NetworkNode, NetworkEdge, fraudDataGenerator } from '../utils/dataGenerator';

// Register layout extensions
if (typeof cytoscape !== 'undefined') {
  cytoscape.use(cola);
  cytoscape.use(dagre);
  cytoscape.use(fcose);
}

interface Props {
  onNodeClick?: (nodeId: string, nodeData: NetworkNode) => void;
  onEdgeClick?: (edgeId: string, edgeData: NetworkEdge) => void;
  filters?: {
    riskLevel?: string[];
    nodeType?: string[];
    edgeType?: string[];
  };
}

const NetworkGraphVisualization: React.FC<Props> = ({ 
  onNodeClick, 
  onEdgeClick, 
  filters = {} 
}) => {
  const [graphData, setGraphData] = useState<{ nodes: NetworkNode[], edges: NetworkEdge[] }>({ nodes: [], edges: [] });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [layout, setLayout] = useState<string>('fcose');
  const [loading, setLoading] = useState(true);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    // Generate initial graph data
    const data = fraudDataGenerator.generateNetworkGraph(30, 45);
    setGraphData(data);
    setLoading(false);
  }, []);

  // Filter data based on current filters
  const filteredData = React.useMemo(() => {
    let filteredNodes = graphData.nodes;
    let filteredEdges = graphData.edges;

    if (filters.riskLevel && filters.riskLevel.length > 0) {
      filteredNodes = filteredNodes.filter(node => 
        filters.riskLevel!.includes(node.riskLevel)
      );
    }

    if (filters.nodeType && filters.nodeType.length > 0) {
      filteredNodes = filteredNodes.filter(node => 
        filters.nodeType!.includes(node.type)
      );
    }

    if (filters.edgeType && filters.edgeType.length > 0) {
      filteredEdges = filteredEdges.filter(edge => 
        filters.edgeType!.includes(edge.type)
      );
    }

    // Only include edges between visible nodes
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    filteredEdges = filteredEdges.filter(edge => 
      nodeIds.has(edge.source) && nodeIds.has(edge.target)
    );

    return { nodes: filteredNodes, edges: filteredEdges };
  }, [graphData, filters]);

  // Convert to Cytoscape format
  const cytoscapeElements: ElementDefinition[] = React.useMemo(() => {
    const elements: ElementDefinition[] = [];

    // Add nodes
    filteredData.nodes.forEach(node => {
      const color = {
        low: '#4CAF50',
        medium: '#FF9800', 
        high: '#FF5722',
        critical: '#D32F2F'
      }[node.riskLevel];

      const size = {
        low: 20,
        medium: 30,
        high: 40,
        critical: 50
      }[node.riskLevel];

      elements.push({
        data: {
          id: node.id,
          label: node.label,
          riskLevel: node.riskLevel,
          fraudScore: node.fraudScore,
          type: node.type,
          metadata: node.metadata,
          color,
          size
        },
        classes: `node-${node.riskLevel} node-${node.type}`
      });
    });

    // Add edges
    filteredData.edges.forEach(edge => {
      const color = {
        normal: '#E0E0E0',
        frequent: '#2196F3',
        suspicious: '#F44336'
      }[edge.type];

      const width = Math.min(Math.max(edge.weight / 10, 1), 8);

      elements.push({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          weight: edge.weight,
          type: edge.type,
          interactions: edge.interactions,
          fraudIndicators: edge.fraudIndicators,
          color,
          width
        },
        classes: `edge-${edge.type}`
      });
    });

    return elements;
  }, [filteredData]);

  const cytoscapeStylesheet = [
    {
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        'border-color': '#333',
        'border-width': 2,
        'width': 'data(size)',
        'height': 'data(size)',
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-halign': 'center',
        'font-size': '10px',
        'font-weight': 'bold',
        'text-outline-width': 1,
        'text-outline-color': '#fff',
        'overlay-padding': '6px',
        'z-index': 10
      }
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': 4,
        'border-color': '#FFD700',
        'background-color': 'data(color)',
        'box-shadow': '0px 0px 20px #FFD700'
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 'data(width)',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'opacity': 0.7
      }
    },
    {
      selector: 'edge:selected',
      style: {
        'line-color': '#FFD700',
        'target-arrow-color': '#FFD700',
        'width': '4px',
        'opacity': 1
      }
    },
    {
      selector: '.node-critical',
      style: {
        'border-width': 4,
        'box-shadow': '0px 0px 15px #D32F2F'
      }
    },
    {
      selector: '.edge-suspicious',
      style: {
        'line-style': 'dashed'
      }
    }
  ];

  const layoutOptions = {
    fcose: {
      name: 'fcose',
      quality: 'default',
      randomize: false,
      animate: true,
      animationDuration: 1000,
      nodeDimensionsIncludeLabels: true,
      uniformNodeDimensions: false,
      packComponents: true,
      nodeRepulsion: 4500,
      idealEdgeLength: 50,
      edgeElasticity: 0.45,
      nestingFactor: 0.1,
      gravity: 0.25,
      numIter: 2500,
      tile: true,
      tilingPaddingVertical: 10,
      tilingPaddingHorizontal: 10
    },
    cola: {
      name: 'cola',
      animate: true,
      animationDuration: 1000,
      refresh: 1,
      maxSimulationTime: 4000,
      ungrabifyWhileSimulating: false,
      fit: true,
      padding: 30,
      nodeDimensionsIncludeLabels: true,
      randomize: false,
      avoidOverlap: true,
      handleDisconnected: true,
      convergenceThreshold: 0.01,
      nodeSpacing: 10,
      flow: undefined,
      alignment: undefined,
      gapInequalities: undefined
    },
    dagre: {
      name: 'dagre',
      animate: true,
      animationDuration: 1000,
      fit: true,
      padding: 30,
      spacingFactor: 1.25,
      nodeDimensionsIncludeLabels: true,
      rankDir: 'TB',
      ranker: 'longest-path'
    }
  };

  const handleNodeClick = (event: any) => {
    const node = event.target;
    const nodeData = node.data();
    
    setSelectedNode(nodeData.id);
    
    if (onNodeClick) {
      const fullNodeData = graphData.nodes.find(n => n.id === nodeData.id);
      if (fullNodeData) {
        onNodeClick(nodeData.id, fullNodeData);
      }
    }
  };

  const handleEdgeClick = (event: any) => {
    const edge = event.target;
    const edgeData = edge.data();
    
    if (onEdgeClick) {
      const fullEdgeData = graphData.edges.find(e => e.id === edgeData.id);
      if (fullEdgeData) {
        onEdgeClick(edgeData.id, fullEdgeData);
      }
    }
  };

  const handleCyReady = (cy: Core) => {
    cyRef.current = cy;
    
    cy.on('tap', 'node', handleNodeClick);
    cy.on('tap', 'edge', handleEdgeClick);
    
    // Add context menu for nodes
    cy.on('cxttap', 'node', (event) => {
      event.preventDefault();
      const node = event.target;
      console.log('Right-clicked node:', node.data());
    });
  };

  const changeLayout = (newLayout: string) => {
    setLayout(newLayout);
    if (cyRef.current) {
      const layoutOpts = (layoutOptions as any)[newLayout];
      if (layoutOpts) {
        cyRef.current.layout(layoutOpts).run();
      }
    }
  };

  const refreshGraph = () => {
    setLoading(true);
    const newData = fraudDataGenerator.generateNetworkGraph(30, 45);
    setGraphData(newData);
    setSelectedNode(null);
    setLoading(false);
  };

  const resetView = () => {
    if (cyRef.current) {
      cyRef.current.fit();
      cyRef.current.center();
    }
  };

  if (loading) {
    return (
      <div style={{ 
        width: '100%', 
        height: '600px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f5f5f5',
        border: '1px solid #ddd',
        borderRadius: '8px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: '16px' }}>Loading Network Graph...</div>
          <div style={{ 
            width: '40px', 
            height: '40px', 
            border: '4px solid #f3f3f3',
            borderTop: '4px solid #3498db',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto'
          }} />
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '600px', position: 'relative' }}>
      {/* Controls */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        zIndex: 1000,
        backgroundColor: 'white',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '8px',
        alignItems: 'center'
      }}>
        <div style={{ fontWeight: 'bold', marginRight: '8px' }}>Layout:</div>
        <select 
          value={layout} 
          onChange={(e) => changeLayout(e.target.value)}
          style={{ padding: '4px 8px', borderRadius: '4px', border: '1px solid #ddd' }}
        >
          <option value="fcose">Force-Directed (F-CoSE)</option>
          <option value="cola">Physics (Cola)</option>
          <option value="dagre">Hierarchical (Dagre)</option>
        </select>
        
        <button 
          onClick={refreshGraph}
          style={{ 
            padding: '6px 12px', 
            backgroundColor: '#2196F3', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Refresh
        </button>
        
        <button 
          onClick={resetView}
          style={{ 
            padding: '6px 12px', 
            backgroundColor: '#4CAF50', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Reset View
        </button>
      </div>

      {/* Legend */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        zIndex: 1000,
        backgroundColor: 'white',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
        fontSize: '12px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Risk Levels</div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
          <div style={{ width: '16px', height: '16px', backgroundColor: '#4CAF50', borderRadius: '50%', marginRight: '8px' }} />
          Low Risk
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
          <div style={{ width: '20px', height: '20px', backgroundColor: '#FF9800', borderRadius: '50%', marginRight: '8px' }} />
          Medium Risk
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
          <div style={{ width: '24px', height: '24px', backgroundColor: '#FF5722', borderRadius: '50%', marginRight: '8px' }} />
          High Risk
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ width: '28px', height: '28px', backgroundColor: '#D32F2F', borderRadius: '50%', marginRight: '8px' }} />
          Critical Risk
        </div>
      </div>

      {/* Graph */}
      <CytoscapeComponent
        elements={cytoscapeElements}
        style={{ width: '100%', height: '100%' }}
        layout={layoutOptions[layout as keyof typeof layoutOptions]}
        stylesheet={cytoscapeStylesheet}
        cy={handleCyReady}
      />

      {/* Selection Info */}
      {selectedNode && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          zIndex: 1000,
          backgroundColor: 'white',
          padding: '12px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          maxWidth: '300px'
        }}>
          {(() => {
            const nodeData = graphData.nodes.find(n => n.id === selectedNode);
            if (!nodeData) return null;
            
            return (
              <div>
                <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Selected Node</div>
                <div><strong>Number:</strong> {nodeData.label}</div>
                <div><strong>Risk Level:</strong> {nodeData.riskLevel.toUpperCase()}</div>
                <div><strong>Fraud Score:</strong> {(nodeData.fraudScore * 100).toFixed(1)}%</div>
                <div><strong>Calls:</strong> {nodeData.metadata.callCount}</div>
                <div><strong>Location:</strong> {nodeData.metadata.location}</div>
                <div><strong>Device:</strong> {nodeData.metadata.deviceInfo}</div>
              </div>
            );
          })()}
        </div>
      )}

      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default NetworkGraphVisualization;