'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Database,
  Plus,
  RefreshCcw,
  CheckCircle,
  AlertCircle,
  Clock,
  Server,
  Cloud,
  HardDrive,
} from 'lucide-react';

const dataSources = [
  {
    id: 'DS-001',
    name: 'PostgreSQL - Primary',
    type: 'database',
    status: 'connected',
    lastSync: '2024-01-15T14:30:00Z',
    records: '45.2M',
    latency: '12ms',
  },
  {
    id: 'DS-002',
    name: 'Apache Kafka',
    type: 'streaming',
    status: 'connected',
    lastSync: '2024-01-15T14:35:00Z',
    records: 'Real-time',
    latency: '3ms',
  },
  {
    id: 'DS-003',
    name: 'Redis Cache',
    type: 'cache',
    status: 'connected',
    lastSync: '2024-01-15T14:35:00Z',
    records: '1.2M keys',
    latency: '<1ms',
  },
  {
    id: 'DS-004',
    name: 'Neo4j Graph DB',
    type: 'graph',
    status: 'connected',
    lastSync: '2024-01-15T14:20:00Z',
    records: '8.5M nodes',
    latency: '45ms',
  },
  {
    id: 'DS-005',
    name: 'External API - Card Network',
    type: 'api',
    status: 'degraded',
    lastSync: '2024-01-15T14:00:00Z',
    records: 'N/A',
    latency: '250ms',
  },
];

const typeIcons = {
  database: Database,
  streaming: Server,
  cache: HardDrive,
  graph: Database,
  api: Cloud,
};

export default function SourcesPage() {
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Data Sources</h1>
          <p className="text-muted-foreground">
            Manage connected data sources and integrations
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Add Source
        </Button>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="bg-emerald-500/5 border-emerald-500/20">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                <CheckCircle className="h-5 w-5 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Connected</p>
                <p className="text-xl font-bold text-emerald-500">
                  {dataSources.filter(s => s.status === 'connected').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-amber-500/5 border-amber-500/20">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
                <AlertCircle className="h-5 w-5 text-amber-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Degraded</p>
                <p className="text-xl font-bold text-amber-500">
                  {dataSources.filter(s => s.status === 'degraded').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-primary/5 border-primary/20">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <Database className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Sources</p>
                <p className="text-xl font-bold">{dataSources.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-blue-500/5 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Clock className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg Latency</p>
                <p className="text-xl font-bold">62ms</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Sources List */}
      <div className="grid gap-4">
        {dataSources.map((source) => {
          const Icon = typeIcons[source.type as keyof typeof typeIcons] || Database;
          
          return (
            <Card key={source.id} className="hover:bg-muted/30 transition-colors">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="h-12 w-12 rounded-lg bg-muted/50 flex items-center justify-center">
                      <Icon className="h-6 w-6 text-muted-foreground" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold">{source.name}</h3>
                        {source.status === 'connected' ? (
                          <Badge className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">
                            <CheckCircle className="mr-1 h-3 w-3" />
                            Connected
                          </Badge>
                        ) : (
                          <Badge className="bg-amber-500/10 text-amber-500 border-amber-500/20">
                            <AlertCircle className="mr-1 h-3 w-3" />
                            Degraded
                          </Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                        <span>Type: {source.type}</span>
                        <span>Records: {source.records}</span>
                        <span>Latency: {source.latency}</span>
                        <span>Last sync: {new Date(source.lastSync).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm">
                      <RefreshCcw className="mr-2 h-4 w-4" />
                      Sync
                    </Button>
                    <Button variant="outline" size="sm">
                      Configure
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
