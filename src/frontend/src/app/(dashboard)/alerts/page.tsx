'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Shield,
  Bell,
  Filter,
  MoreHorizontal,
  Eye,
  ArrowRight,
} from 'lucide-react';

interface Alert {
  id: string;
  type: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  timestamp: string;
  status: 'new' | 'investigating' | 'resolved' | 'dismissed';
  transactionId?: string;
  userId?: string;
}

const mockAlerts: Alert[] = [
  {
    id: 'ALT-001',
    type: 'critical',
    title: 'Potential Fraud Ring Detected',
    description: 'Multiple accounts sharing same IP and device fingerprint detected in velocity spike',
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    status: 'new',
    transactionId: 'TXN-839201',
  },
  {
    id: 'ALT-002',
    type: 'high',
    title: 'Account Takeover Attempt',
    description: 'Login from new device and location with subsequent high-value transaction',
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    status: 'investigating',
    userId: 'USR-12345',
  },
  {
    id: 'ALT-003',
    type: 'high',
    title: 'Card Testing Pattern',
    description: 'Series of small transactions followed by large purchase attempt',
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    status: 'new',
    transactionId: 'TXN-839187',
  },
  {
    id: 'ALT-004',
    type: 'medium',
    title: 'Unusual Transaction Location',
    description: 'Transaction originated from a country not in user\'s history',
    timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    status: 'investigating',
    transactionId: 'TXN-839156',
  },
  {
    id: 'ALT-005',
    type: 'medium',
    title: 'Velocity Threshold Exceeded',
    description: 'User exceeded 5 transactions per minute threshold',
    timestamp: new Date(Date.now() - 90 * 60 * 1000).toISOString(),
    status: 'resolved',
    userId: 'USR-67890',
  },
  {
    id: 'ALT-006',
    type: 'low',
    title: 'New Device Detected',
    description: 'First transaction from a new device for established user',
    timestamp: new Date(Date.now() - 120 * 60 * 1000).toISOString(),
    status: 'dismissed',
    userId: 'USR-11111',
  },
];

const alertTypeConfig = {
  critical: { color: 'bg-red-500', textColor: 'text-red-500', bgColor: 'bg-red-500/10', label: 'Critical' },
  high: { color: 'bg-orange-500', textColor: 'text-orange-500', bgColor: 'bg-orange-500/10', label: 'High' },
  medium: { color: 'bg-amber-500', textColor: 'text-amber-500', bgColor: 'bg-amber-500/10', label: 'Medium' },
  low: { color: 'bg-blue-500', textColor: 'text-blue-500', bgColor: 'bg-blue-500/10', label: 'Low' },
};

const statusConfig = {
  new: { icon: Bell, color: 'text-primary', label: 'New' },
  investigating: { icon: Eye, color: 'text-amber-500', label: 'Investigating' },
  resolved: { icon: CheckCircle, color: 'text-emerald-500', label: 'Resolved' },
  dismissed: { icon: XCircle, color: 'text-muted-foreground', label: 'Dismissed' },
};

export default function AlertsPage() {
  const [alerts] = useState<Alert[]>(mockAlerts);
  const [selectedTab, setSelectedTab] = useState('all');

  const filteredAlerts = alerts.filter((alert) => {
    if (selectedTab === 'all') return true;
    return alert.status === selectedTab;
  });

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
  };

  const stats = {
    critical: alerts.filter(a => a.type === 'critical' && a.status !== 'resolved' && a.status !== 'dismissed').length,
    high: alerts.filter(a => a.type === 'high' && a.status !== 'resolved' && a.status !== 'dismissed').length,
    new: alerts.filter(a => a.status === 'new').length,
    investigating: alerts.filter(a => a.status === 'investigating').length,
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Alerts</h1>
          <p className="text-muted-foreground">
            Monitor and respond to security alerts
          </p>
        </div>
        <Button variant="outline" size="sm">
          <Filter className="mr-2 h-4 w-4" />
          Filter
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="bg-red-500/5 border-red-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Critical</p>
                <p className="text-2xl font-bold text-red-500">{stats.critical}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500/50" />
            </div>
          </CardContent>
        </Card>
        <Card className="bg-orange-500/5 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">High Priority</p>
                <p className="text-2xl font-bold text-orange-500">{stats.high}</p>
              </div>
              <Shield className="h-8 w-8 text-orange-500/50" />
            </div>
          </CardContent>
        </Card>
        <Card className="bg-primary/5 border-primary/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">New Alerts</p>
                <p className="text-2xl font-bold text-primary">{stats.new}</p>
              </div>
              <Bell className="h-8 w-8 text-primary/50" />
            </div>
          </CardContent>
        </Card>
        <Card className="bg-amber-500/5 border-amber-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Investigating</p>
                <p className="text-2xl font-bold text-amber-500">{stats.investigating}</p>
              </div>
              <Eye className="h-8 w-8 text-amber-500/50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Alert List */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList>
          <TabsTrigger value="all">All ({alerts.length})</TabsTrigger>
          <TabsTrigger value="new">New ({alerts.filter(a => a.status === 'new').length})</TabsTrigger>
          <TabsTrigger value="investigating">Investigating ({alerts.filter(a => a.status === 'investigating').length})</TabsTrigger>
          <TabsTrigger value="resolved">Resolved ({alerts.filter(a => a.status === 'resolved').length})</TabsTrigger>
        </TabsList>

        <TabsContent value={selectedTab} className="mt-4">
          <Card>
            <ScrollArea className="h-[500px]">
              <div className="divide-y divide-border/50">
                {filteredAlerts.map((alert, index) => {
                  const typeConfig = alertTypeConfig[alert.type];
                  const statusCfg = statusConfig[alert.status];
                  const StatusIcon = statusCfg.icon;

                  return (
                    <motion.div
                      key={alert.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="flex items-start gap-4 p-4 hover:bg-muted/30 transition-colors cursor-pointer"
                    >
                      {/* Severity Indicator */}
                      <div className={cn('h-10 w-10 rounded-lg flex items-center justify-center', typeConfig.bgColor)}>
                        <AlertTriangle className={cn('h-5 w-5', typeConfig.textColor)} />
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-mono text-xs text-muted-foreground">{alert.id}</span>
                          <Badge className={cn('text-[10px]', typeConfig.bgColor, typeConfig.textColor)}>
                            {typeConfig.label}
                          </Badge>
                          <div className={cn('flex items-center gap-1 text-xs', statusCfg.color)}>
                            <StatusIcon className="h-3 w-3" />
                            {statusCfg.label}
                          </div>
                        </div>
                        <h4 className="font-medium">{alert.title}</h4>
                        <p className="text-sm text-muted-foreground mt-0.5 truncate">{alert.description}</p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatTime(alert.timestamp)}
                          </span>
                          {alert.transactionId && (
                            <span className="font-mono">{alert.transactionId}</span>
                          )}
                          {alert.userId && (
                            <span className="font-mono">{alert.userId}</span>
                          )}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="sm">
                          View
                          <ArrowRight className="ml-1 h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </ScrollArea>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
