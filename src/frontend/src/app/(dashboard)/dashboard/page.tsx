'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  ArrowUpRight,
  ArrowDownRight,
  ShieldAlert,
  AlertTriangle,
  Zap,
  TrendingUp,
  ArrowRight,
  RefreshCcw,
  Clock,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

// Mock data generators for Telecom CDRs
const generateCallTrafficData = () => {
  const data = [];
  const now = new Date();
  for (let i = 23; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000);
    data.push({
      time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      volume: Math.floor(Math.random() * 500) + 200,  // Calls per second
      flagged: Math.floor(Math.random() * 50) + 10,   // Wangiri/SIM Box attempts
      blocked: Math.floor(Math.random() * 20) + 5,
    });
  }
  return data;
};

const generateRecentCDRs = () => {
  const callTypes = ['Voice', 'SMS'];
  const fraudTypes = ['Normal', 'Wangiri', 'SIM Box', 'IRSF'];
  const statuses = ['approved', 'flagged', 'blocked', 'pending'] as const;
  const cdrs = [];
  
  // Sample MSISDNs
  const generateMSISDN = () => `+212${Math.floor(Math.random() * 100000000 + 600000000)}`;
  
  for (let i = 0; i < 10; i++) {
    const risk = Math.random();
    let status: typeof statuses[number];
    let fraudType = 'Normal';
    
    if (risk > 0.85) {
      status = 'blocked';
      fraudType = Math.random() > 0.5 ? 'Wangiri' : 'SIM Box';
    } else if (risk > 0.7) {
      status = 'flagged';
      fraudType = Math.random() > 0.5 ? 'Wangiri' : 'SIM Box';
    } else if (risk > 0.6) {
      status = 'pending';
    } else {
      status = 'approved';
    }

    cdrs.push({
      id: `CDR-${String(Math.floor(Math.random() * 900000) + 100000)}`,
      callerMsisdn: generateMSISDN(),
      calleeMsisdn: fraudType === 'Wangiri' ? `+220${Math.floor(Math.random() * 10000000)}` : generateMSISDN(),
      callType: callTypes[Math.floor(Math.random() * callTypes.length)],
      duration: fraudType === 'Wangiri' ? Math.floor(Math.random() * 3) + 1 : Math.floor(Math.random() * 300) + 10,
      fraudType,
      status,
      riskScore: Math.floor(risk * 100),
      timestamp: new Date(Date.now() - Math.floor(Math.random() * 3600000)).toISOString(),
    });
  }
  
  return cdrs.sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
};

const fraudDistribution = [
  { type: 'Normal', count: 8500, color: '#10b981' },
  { type: 'Wangiri', count: 320, color: '#ef4444' },
  { type: 'SIM Box', count: 180, color: '#f97316' },
  { type: 'IRSF', count: 45, color: '#eab308' },
  { type: 'Other', count: 25, color: '#8b5cf6' },
];

interface MetricCardProps {
  title: string;
  value: string | number;
  change: number;
  icon: React.ElementType;
  trend: 'up' | 'down';
  trendGood?: boolean;
  suffix?: string;
}

function MetricCard({ title, value, change, icon: Icon, trend, trendGood = true, suffix }: MetricCardProps) {
  const isPositive = trend === 'up';
  const isGood = trendGood ? isPositive : !isPositive;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="relative overflow-hidden border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">{title}</p>
              <div className="flex items-baseline gap-2">
                <p className="text-3xl font-bold tracking-tight">{value}</p>
                {suffix && <span className="text-sm text-muted-foreground">{suffix}</span>}
              </div>
              <div className={cn(
                'flex items-center gap-1 text-sm font-medium',
                isGood ? 'text-emerald-500' : 'text-destructive'
              )}>
                {isPositive ? (
                  <ArrowUpRight className="h-4 w-4" />
                ) : (
                  <ArrowDownRight className="h-4 w-4" />
                )}
                <span>{Math.abs(change)}%</span>
                <span className="text-muted-foreground font-normal">vs last hour</span>
              </div>
            </div>
            <div className={cn(
              'flex h-12 w-12 items-center justify-center rounded-full',
              'bg-primary/10'
            )}>
              <Icon className="h-6 w-6 text-primary" />
            </div>
          </div>
        </CardContent>
        {/* Gradient accent */}
        <div className="absolute bottom-0 left-0 h-1 w-full bg-gradient-to-r from-primary/50 via-primary to-primary/50" />
      </Card>
    </motion.div>
  );
}

export default function PulseDashboard() {
  const [callTrafficData, setCallTrafficData] = useState(generateCallTrafficData);
  const [cdrs, setCdrs] = useState(generateRecentCDRs);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCallTrafficData(generateCallTrafficData());
      setCdrs(generateRecentCDRs());
      setLastUpdate(new Date());
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = (status: string, riskScore: number) => {
    const variants: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline', className: string }> = {
      approved: { variant: 'secondary', className: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' },
      flagged: { variant: 'secondary', className: 'bg-amber-500/10 text-amber-500 border-amber-500/20' },
      blocked: { variant: 'destructive', className: 'bg-destructive/10 text-destructive border-destructive/20' },
      pending: { variant: 'outline', className: 'bg-blue-500/10 text-blue-500 border-blue-500/20' },
    };
    const config = variants[status] || variants.pending;
    
    return (
      <Badge variant={config.variant} className={cn('font-medium', config.className)}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
        <span className="ml-1.5 font-mono text-[10px] opacity-70">{riskScore}</span>
      </Badge>
    );
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Telecom FSOC</h1>
          <p className="text-muted-foreground">
            Real-time telecom fraud detection monitoring
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="h-4 w-4" />
            <span>Last updated: {lastUpdate.toLocaleTimeString()}</span>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => {
              setCallTrafficData(generateCallTrafficData());
              setCdrs(generateRecentCDRs());
              setLastUpdate(new Date());
            }}
          >
            <RefreshCcw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Call Traffic"
          value="4.2K"
          change={12.5}
          icon={TrendingUp}
          trend="up"
          suffix=" CPS"
        />
        <MetricCard
          title="Revenue Saved"
          value="$127K"
          change={8.3}
          icon={ShieldAlert}
          trend="up"
          trendGood={true}
        />
        <MetricCard
          title="Fraud Detected"
          value="47"
          change={-15}
          icon={AlertTriangle}
          trend="down"
          trendGood={true}
        />
        <MetricCard
          title="Avg. Duration"
          value="142"
          change={-5.2}
          icon={Zap}
          trend="down"
          suffix="s"
        />
      </div>

      {/* Charts Row */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Call Traffic Chart */}
        <Card className="lg:col-span-2 border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-lg font-semibold">Call Traffic (CPS)</CardTitle>
                <CardDescription>Real-time call flow analysis</CardDescription>
              </div>
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-primary" />
                  <span className="text-muted-foreground">Total Calls</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-amber-500" />
                  <span className="text-muted-foreground">Wangiri</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-destructive" />
                  <span className="text-muted-foreground">SIM Box</span>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={callTrafficData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="flaggedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#eab308" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#eab308" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis 
                    dataKey="time" 
                    stroke="hsl(var(--muted-foreground))" 
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="hsl(var(--muted-foreground))" 
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    labelStyle={{ color: 'hsl(var(--foreground))' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="cps"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    fill="url(#volumeGradient)"
                  />
                  <Area
                    type="monotone"
                    dataKey="wangiri"
                    stroke="#eab308"
                    strokeWidth={2}
                    fill="url(#flaggedGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Fraud Distribution */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg font-semibold">Fraud Distribution</CardTitle>
            <CardDescription>CDR fraud type breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fraudDistribution} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} horizontal={false} />
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="range" 
                    type="category" 
                    width={50}
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number) => [value.toLocaleString(), 'CDRs']}
                  />
                  <Bar 
                    dataKey="count" 
                    radius={[0, 4, 4, 0]}
                    fill="hsl(var(--primary))"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Live Ticker */}
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="h-2 w-2 rounded-full bg-emerald-500"
                />
                <CardTitle className="text-lg font-semibold">Live CDR Feed</CardTitle>
              </div>
              <Badge variant="secondary" className="font-mono text-xs">
                {cdrs.length} recent
              </Badge>
            </div>
            <Button variant="ghost" size="sm" className="text-muted-foreground">
              View All
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[320px]">
            <div className="divide-y divide-border/50">
              <AnimatePresence mode="popLayout">
                {cdrs.map((cdr, index) => (
                  <motion.div
                    key={cdr.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ delay: index * 0.03 }}
                    className="flex items-center justify-between px-6 py-3 hover:bg-muted/30 transition-colors cursor-pointer"
                  >
                    <div className="flex items-center gap-4">
                      <div className={cn(
                        'flex h-10 w-10 items-center justify-center rounded-lg',
                        cdr.status === 'blocked' && 'bg-destructive/10',
                        cdr.status === 'flagged' && 'bg-amber-500/10',
                        cdr.status === 'approved' && 'bg-emerald-500/10',
                        cdr.status === 'pending' && 'bg-blue-500/10',
                      )}>
                        <TrendingUp className={cn(
                          'h-5 w-5',
                          cdr.status === 'blocked' && 'text-destructive',
                          cdr.status === 'flagged' && 'text-amber-500',
                          cdr.status === 'approved' && 'text-emerald-500',
                          cdr.status === 'pending' && 'text-blue-500',
                        )} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm font-medium">{cdr.callerMsisdn}</span>
                          <span className="text-muted-foreground">→</span>
                          <span className="font-mono text-sm">{cdr.calleeMsisdn}</span>
                          <Badge variant="outline" className="text-[10px] font-normal">
                            {cdr.callType}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">{formatTime(cdr.timestamp)} • {formatDuration(cdr.duration)}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <Badge variant={cdr.fraudType === 'NONE' ? 'secondary' : 'destructive'} className="text-xs">
                        {cdr.fraudType === 'NONE' ? 'Clean' : cdr.fraudType}
                      </Badge>
                      {getStatusBadge(cdr.status, cdr.riskScore)}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
