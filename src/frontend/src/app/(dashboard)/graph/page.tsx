'use client';

import { useState, useCallback, useMemo, useRef } from 'react';
import dynamic from 'next/dynamic';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Search,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RefreshCcw,
  Phone,
  Radio,
  AlertTriangle,
  Shield,
  Link as LinkIcon,
  Network,
  Smartphone,
  PhoneCall,
} from 'lucide-react';

// Dynamically import ForceGraph to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
    </div>
  ),
});

// Node types for telecom
type NodeType = 'msisdn' | 'imei' | 'celltower';

interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  riskScore: number;
  callCount: number;
  flagged: boolean;
  fraudType?: 'WANGIRI' | 'SIMBOX' | 'IRSF' | 'NONE';
  val?: number;
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  value: number; // number of calls
  suspicious: boolean;
  avgDuration: number; // average call duration in seconds
  fraudType?: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// Generate mock telecom graph data
const generateTelecomGraphData = (): GraphData => {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];

  const msisdnPrefixes = ['+212661', '+212662', '+212663', '+212670', '+212671'];
  const internationalPrefixes = ['+220', '+236', '+216', '+33', '+34'];
  
  // Create domestic MSISDNs (phone numbers)
  for (let i = 0; i < 25; i++) {
    const riskScore = Math.floor(Math.random() * 100);
    const prefix = msisdnPrefixes[Math.floor(Math.random() * msisdnPrefixes.length)];
    const isFraud = riskScore > 70;
    
    nodes.push({
      id: `msisdn-${i}`,
      label: `${prefix}${Math.floor(Math.random() * 1000000).toString().padStart(6, '0')}`,
      type: 'msisdn',
      riskScore,
      callCount: Math.floor(Math.random() * 100) + 10,
      flagged: isFraud,
      fraudType: isFraud ? (Math.random() > 0.5 ? 'SIMBOX' : 'WANGIRI') : 'NONE',
      val: 10,
    });
  }

  // Create international MSISDNs (premium destinations for fraud)
  for (let i = 0; i < 8; i++) {
    const riskScore = 60 + Math.floor(Math.random() * 40); // High risk
    const prefix = internationalPrefixes[Math.floor(Math.random() * internationalPrefixes.length)];
    
    nodes.push({
      id: `intl-msisdn-${i}`,
      label: `${prefix}${Math.floor(Math.random() * 10000000).toString().padStart(7, '0')}`,
      type: 'msisdn',
      riskScore,
      callCount: Math.floor(Math.random() * 50) + 5,
      flagged: true,
      fraudType: 'IRSF',
      val: 8,
    });
  }

  // Create IMEIs (devices) - some shared by multiple MSISDNs (SIM Box indicator)
  for (let i = 0; i < 12; i++) {
    const riskScore = Math.floor(Math.random() * 100);
    const isSimBox = i < 3; // First 3 are SIM Box devices
    
    nodes.push({
      id: `imei-${i}`,
      label: `${Math.floor(Math.random() * 100000000000000).toString().padStart(15, '0')}`,
      type: 'imei',
      riskScore: isSimBox ? 85 + Math.floor(Math.random() * 15) : riskScore,
      callCount: isSimBox ? 200 + Math.floor(Math.random() * 300) : Math.floor(Math.random() * 50),
      flagged: isSimBox || riskScore > 80,
      fraudType: isSimBox ? 'SIMBOX' : 'NONE',
      val: 8,
    });
  }

  // Create Cell Towers
  const towerLocations = ['Casablanca', 'Rabat', 'Marrakech', 'Fes', 'Tangier', 'Agadir'];
  for (let i = 0; i < towerLocations.length; i++) {
    const riskScore = Math.floor(Math.random() * 50); // Towers are generally low risk
    
    nodes.push({
      id: `tower-${i}`,
      label: `TWR-${towerLocations[i].substring(0, 3).toUpperCase()}-001`,
      type: 'celltower',
      riskScore,
      callCount: Math.floor(Math.random() * 500) + 100,
      flagged: false,
      val: 14,
    });
  }

  // Create links (calls between MSISDNs)
  const msisdnNodes = nodes.filter(n => n.type === 'msisdn');
  const imeiNodes = nodes.filter(n => n.type === 'imei');
  const towerNodes = nodes.filter(n => n.type === 'celltower');

  // MSISDN to MSISDN calls
  msisdnNodes.forEach((caller) => {
    const numCalls = Math.floor(Math.random() * 4) + 1;
    const possibleCallees = msisdnNodes.filter(n => n.id !== caller.id);
    
    for (let i = 0; i < numCalls; i++) {
      const callee = possibleCallees[Math.floor(Math.random() * possibleCallees.length)];
      if (callee && !links.find(l => 
        (l.source === caller.id && l.target === callee.id)
      )) {
        const isSuspicious = caller.flagged || callee.flagged;
        // Wangiri = very short calls (1-3s)
        const avgDuration = isSuspicious && caller.fraudType === 'WANGIRI' 
          ? Math.floor(Math.random() * 3) + 1 
          : Math.floor(Math.random() * 300) + 30;
        
        links.push({
          source: caller.id,
          target: callee.id,
          value: Math.floor(Math.random() * 20) + 1,
          suspicious: isSuspicious,
          avgDuration,
          fraudType: caller.fraudType !== 'NONE' ? caller.fraudType : callee.fraudType,
        });
      }
    }
  });

  // MSISDN to IMEI connections (device usage)
  msisdnNodes.forEach((msisdn) => {
    // Each MSISDN connects to 1-2 IMEIs
    const numDevices = Math.floor(Math.random() * 2) + 1;
    for (let i = 0; i < numDevices; i++) {
      const imei = imeiNodes[Math.floor(Math.random() * imeiNodes.length)];
      if (!links.find(l => l.source === msisdn.id && l.target === imei.id)) {
        links.push({
          source: msisdn.id,
          target: imei.id,
          value: Math.floor(Math.random() * 30) + 5,
          suspicious: msisdn.flagged || imei.flagged,
          avgDuration: 0,
        });
      }
    }
  });

  // SIM Box pattern: Multiple MSISDNs sharing same IMEI
  const simBoxImeis = imeiNodes.filter(n => n.fraudType === 'SIMBOX');
  simBoxImeis.forEach((imei) => {
    // Connect 5-8 MSISDNs to each SIM Box IMEI
    const numSims = Math.floor(Math.random() * 4) + 5;
    const domesticMsisdns = msisdnNodes.filter(n => n.id.startsWith('msisdn-'));
    for (let i = 0; i < numSims; i++) {
      const msisdn = domesticMsisdns[Math.floor(Math.random() * domesticMsisdns.length)];
      if (!links.find(l => l.source === msisdn.id && l.target === imei.id)) {
        links.push({
          source: msisdn.id,
          target: imei.id,
          value: Math.floor(Math.random() * 50) + 20,
          suspicious: true,
          avgDuration: 0,
          fraudType: 'SIMBOX',
        });
      }
    }
  });

  // MSISDN to Cell Tower connections
  msisdnNodes.forEach((msisdn) => {
    const tower = towerNodes[Math.floor(Math.random() * towerNodes.length)];
    if (!links.find(l => l.source === msisdn.id && l.target === tower.id)) {
      links.push({
        source: msisdn.id,
        target: tower.id,
        value: msisdn.callCount,
        suspicious: false,
        avgDuration: 0,
      });
    }
  });

  return { nodes, links };
};

// Node colors by type
const nodeColors: Record<NodeType, string> = {
  msisdn: '#3b82f6',    // Blue for phone numbers
  imei: '#8b5cf6',      // Purple for devices
  celltower: '#10b981', // Green for cell towers
};

const nodeIcons: Record<NodeType, React.ElementType> = {
  msisdn: Phone,
  imei: Smartphone,
  celltower: Radio,
};

export default function TelecomNetworkGraph() {
  const fgRef = useRef<any>(null);
  const [graphData, setGraphData] = useState<GraphData>(generateTelecomGraphData);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<NodeType | 'all'>('all');
  const [showSuspiciousOnly, setShowSuspiciousOnly] = useState(false);

  // Filter graph data
  const filteredData = useMemo(() => {
    let nodes = graphData.nodes;
    let links = graphData.links;

    if (filterType !== 'all') {
      nodes = nodes.filter(n => n.type === filterType);
      const nodeIds = new Set(nodes.map(n => n.id));
      links = links.filter(l => 
        nodeIds.has(l.source as string) || nodeIds.has(l.target as string)
      );
    }

    if (searchQuery) {
      nodes = nodes.filter(n => 
        n.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
        n.id.toLowerCase().includes(searchQuery.toLowerCase())
      );
      const nodeIds = new Set(nodes.map(n => n.id));
      links = links.filter(l => 
        nodeIds.has(l.source as string) || nodeIds.has(l.target as string)
      );
    }

    if (showSuspiciousOnly) {
      nodes = nodes.filter(n => n.flagged);
      const nodeIds = new Set(nodes.map(n => n.id));
      links = links.filter(l => l.suspicious);
    }

    return { nodes, links };
  }, [graphData, filterType, searchQuery, showSuspiciousOnly]);

  const handleNodeClick = useCallback((node: any) => {
    const graphNode = node as GraphNode;
    setSelectedNode(graphNode);
    
    if (fgRef.current && node.x !== undefined && node.y !== undefined) {
      fgRef.current.centerAt(node.x, node.y, 1000);
      fgRef.current.zoom(2, 1000);
    }
  }, []);

  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.label;
    const fontSize = 10 / globalScale;
    ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
    
    const baseColor = nodeColors[node.type as NodeType] || '#6b7280';
    const size = (node.val || 5) / globalScale;
    
    ctx.beginPath();
    ctx.arc(node.x, node.y, size * 2, 0, 2 * Math.PI);
    ctx.fillStyle = node.flagged ? '#ef4444' : baseColor;
    ctx.fill();
    
    if (selectedNode?.id === node.id) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();
    }
    
    if (node.flagged) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size * 3, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
      ctx.fill();
    }
    
    if (globalScale > 0.5) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      // Truncate IMEI for readability
      const displayLabel = node.type === 'imei' ? `...${label.slice(-6)}` : label;
      ctx.fillText(displayLabel, node.x, node.y + size * 2.5);
    }
  }, [selectedNode]);

  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const start = link.source;
    const end = link.target;
    
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    
    // Color by fraud type
    if (link.fraudType === 'WANGIRI') {
      ctx.strokeStyle = 'rgba(249, 115, 22, 0.7)'; // Orange
    } else if (link.fraudType === 'SIMBOX') {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)'; // Red
    } else if (link.fraudType === 'IRSF') {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.7)'; // Purple
    } else if (link.suspicious) {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
    } else {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    }
    
    ctx.lineWidth = link.suspicious ? 2 / globalScale : 1 / globalScale;
    ctx.stroke();
  }, []);

  const handleZoomIn = () => fgRef.current?.zoom(fgRef.current.zoom() * 1.5, 400);
  const handleZoomOut = () => fgRef.current?.zoom(fgRef.current.zoom() / 1.5, 400);
  const handleFitView = () => fgRef.current?.zoomToFit(400);

  const stats = useMemo(() => ({
    totalNodes: filteredData.nodes.length,
    totalLinks: filteredData.links.length,
    flaggedNodes: filteredData.nodes.filter(n => n.flagged).length,
    suspiciousLinks: filteredData.links.filter(l => l.suspicious).length,
    wangiriCount: filteredData.nodes.filter(n => n.fraudType === 'WANGIRI').length,
    simboxCount: filteredData.nodes.filter(n => n.fraudType === 'SIMBOX').length,
  }), [filteredData]);

  const formatDuration = (seconds: number) => {
    if (seconds === 0) return 'N/A';
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Main Graph Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border/50">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Telecom Network Graph</h1>
            <p className="text-sm text-muted-foreground">
              Visualize call patterns and fraud networks
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setGraphData(generateTelecomGraphData())}>
              <RefreshCcw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-4 p-4 border-b border-border/50 bg-muted/30">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search MSISDN, IMEI..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          <Select value={filterType} onValueChange={(v) => setFilterType(v as NodeType | 'all')}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="msisdn">Phone Numbers</SelectItem>
              <SelectItem value="imei">Devices (IMEI)</SelectItem>
              <SelectItem value="celltower">Cell Towers</SelectItem>
            </SelectContent>
          </Select>

          <Button
            variant={showSuspiciousOnly ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowSuspiciousOnly(!showSuspiciousOnly)}
          >
            <AlertTriangle className="mr-2 h-4 w-4" />
            Fraud Only
          </Button>

          <Separator orientation="vertical" className="h-8" />

          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleFitView}>
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Graph */}
        <div className="flex-1 bg-background">
          <ForceGraph2D
            ref={fgRef}
            graphData={filteredData}
            nodeCanvasObject={paintNode}
            linkCanvasObject={paintLink}
            onNodeClick={handleNodeClick}
            nodeLabel={(node: any) => `${node.label} (Risk: ${node.riskScore})`}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={(link: any) => link.suspicious ? 4 : 0}
            linkDirectionalParticleColor={(link: any) => {
              if (link.fraudType === 'WANGIRI') return '#f97316';
              if (link.fraudType === 'SIMBOX') return '#ef4444';
              return '#ef4444';
            }}
            backgroundColor="transparent"
            d3VelocityDecay={0.3}
          />
        </div>

        {/* Stats Bar */}
        <div className="flex items-center gap-6 p-3 border-t border-border/50 bg-muted/30">
          <div className="flex items-center gap-2 text-sm">
            <Phone className="h-3 w-3 text-blue-500" />
            <span className="text-muted-foreground">Nodes:</span>
            <span className="font-mono font-medium">{stats.totalNodes}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <PhoneCall className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Calls:</span>
            <span className="font-mono font-medium">{stats.totalLinks}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <AlertTriangle className="h-3 w-3 text-destructive" />
            <span className="text-muted-foreground">Flagged:</span>
            <span className="font-mono font-medium text-destructive">{stats.flaggedNodes}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="outline" className="bg-orange-500/10 text-orange-500 border-orange-500/20 text-xs">
              Wangiri: {stats.wangiriCount}
            </Badge>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 text-xs">
              SIM Box: {stats.simboxCount}
            </Badge>
          </div>
        </div>
      </div>

      {/* Node Details Panel */}
      <div className="w-80 border-l border-border/50 bg-card/50">
        <div className="p-4 border-b border-border/50">
          <h2 className="font-semibold">Entity Details</h2>
          <p className="text-xs text-muted-foreground">
            Click a node to view details
          </p>
        </div>

        <ScrollArea className="h-[calc(100vh-12rem)]">
          {selectedNode ? (
            <div className="p-4 space-y-4">
              {/* Node Header */}
              <div className="flex items-center gap-3">
                <div 
                  className="h-10 w-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${nodeColors[selectedNode.type]}20` }}
                >
                  {(() => {
                    const Icon = nodeIcons[selectedNode.type];
                    return <Icon className="h-5 w-5" style={{ color: nodeColors[selectedNode.type] }} />;
                  })()}
                </div>
                <div>
                  <p className="font-mono text-sm">
                    {selectedNode.type === 'imei' 
                      ? `...${selectedNode.label.slice(-8)}` 
                      : selectedNode.label}
                  </p>
                  <Badge variant="outline" className="capitalize text-xs">
                    {selectedNode.type === 'msisdn' ? 'Phone Number' : 
                     selectedNode.type === 'imei' ? 'Device' : 'Cell Tower'}
                  </Badge>
                </div>
              </div>

              <Separator />

              {/* Fraud Type */}
              {selectedNode.fraudType && selectedNode.fraudType !== 'NONE' && (
                <Card className="border-destructive/50 bg-destructive/5">
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-destructive" />
                      <span className="text-sm font-medium">Fraud Detected</span>
                    </div>
                    <Badge variant="destructive" className="mt-2">
                      {selectedNode.fraudType}
                    </Badge>
                  </CardContent>
                </Card>
              )}

              {/* Risk Score */}
              <Card className={cn(
                'border-2',
                selectedNode.riskScore >= 70 && 'border-destructive/50 bg-destructive/5',
                selectedNode.riskScore >= 40 && selectedNode.riskScore < 70 && 'border-amber-500/50 bg-amber-500/5',
                selectedNode.riskScore < 40 && 'border-emerald-500/50 bg-emerald-500/5',
              )}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Risk Score</p>
                      <p className="text-2xl font-bold">{selectedNode.riskScore}</p>
                    </div>
                    {selectedNode.flagged && (
                      <Badge variant="destructive">
                        <AlertTriangle className="mr-1 h-3 w-3" />
                        Flagged
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-3">
                <Card>
                  <CardContent className="p-3">
                    <p className="text-xs text-muted-foreground">
                      {selectedNode.type === 'celltower' ? 'Total CDRs' : 'Calls'}
                    </p>
                    <p className="text-lg font-semibold">{selectedNode.callCount}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-3">
                    <p className="text-xs text-muted-foreground">Connections</p>
                    <p className="text-lg font-semibold">
                      {filteredData.links.filter(l => 
                        l.source === selectedNode.id || l.target === selectedNode.id ||
                        (l.source as any)?.id === selectedNode.id || (l.target as any)?.id === selectedNode.id
                      ).length}
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Full ID */}
              <div className="p-3 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground mb-1">
                  {selectedNode.type === 'msisdn' ? 'MSISDN' : 
                   selectedNode.type === 'imei' ? 'IMEI' : 'Tower ID'}
                </p>
                <p className="font-mono text-sm break-all">{selectedNode.label}</p>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <Button variant="outline" className="flex-1" size="sm">
                  Investigate
                </Button>
                <Button variant="destructive" size="sm">
                  Block
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-center p-4">
              <Network className="h-10 w-10 text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">
                Select a node to view call patterns and fraud indicators
              </p>
            </div>
          )}
        </ScrollArea>

        {/* Legend */}
        <div className="p-4 border-t border-border/50">
          <p className="text-xs font-medium text-muted-foreground mb-3">Legend</p>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-blue-500" />
              <span className="text-xs">Phone (MSISDN)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-purple-500" />
              <span className="text-xs">Device (IMEI)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-emerald-500" />
              <span className="text-xs">Cell Tower</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-destructive" />
              <span className="text-xs">Flagged</span>
            </div>
          </div>
          <Separator className="my-3" />
          <p className="text-xs font-medium text-muted-foreground mb-2">Link Colors</p>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2">
              <div className="h-0.5 w-4 bg-orange-500" />
              <span className="text-xs">Wangiri</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-0.5 w-4 bg-red-500" />
              <span className="text-xs">SIM Box</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-0.5 w-4 bg-purple-500" />
              <span className="text-xs">IRSF</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-0.5 w-4 bg-slate-400" />
              <span className="text-xs">Normal</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
