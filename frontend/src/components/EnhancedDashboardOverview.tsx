/**
 * Enhanced Dashboard Overview
 * Professional dashboard with real-time metrics and comprehensive fraud analytics
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  IconButton,
  Button,
  LinearProgress,
  Chip,
  Alert,
  Tooltip,
  styled,
  alpha,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  Security,
  Person,
  Timeline,
  Refresh,
  MoreVert,
  Shield,
  Speed,
  NetworkCheck,
  Analytics,
  ReportProblem,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface DashboardOverviewProps {
  onOpenSection?: (section: string) => void;
}

const MetricCard = styled(Card)(({ theme }) => ({
  height: '120px',
  background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
  transition: 'all 0.3s ease',
  cursor: 'pointer',
  '&:hover': {
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
    transform: 'translateY(-2px)',
  },
}));

const ChartCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
  '& .recharts-wrapper': {
    fontFamily: '"Segoe UI", "Roboto", sans-serif',
  },
}));

const AlertCard = styled(Card)(({ theme }) => ({
  borderLeft: '4px solid #f44336',
  transition: 'all 0.2s ease',
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: alpha('#f44336', 0.02),
  },
  '&.warning': {
    borderLeftColor: '#ff9800',
  },
  '&.info': {
    borderLeftColor: '#2196f3',
  },
  '&.success': {
    borderLeftColor: '#4caf50',
  },
}));

const StatusIndicator = styled(Box)<{ status: 'online' | 'warning' | 'error' }>(({ status }) => ({
  width: 8,
  height: 8,
  borderRadius: '50%',
  backgroundColor: status === 'online' ? '#4caf50' : status === 'warning' ? '#ff9800' : '#f44336',
  animation: status !== 'online' ? 'pulse 2s infinite' : 'none',
  '@keyframes pulse': {
    '0%': { opacity: 1 },
    '50%': { opacity: 0.5 },
    '100%': { opacity: 1 },
  },
}));

export default function DashboardOverview({ onOpenSection }: DashboardOverviewProps) {
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Mock data - replace with real API calls
  const [metrics, setMetrics] = useState({
    totalAlerts: 247,
    fraudDetected: 23,
    riskScore: 78,
    systemHealth: 96,
    activeUsers: 1534,
    processedCalls: 45623,
    blockedCalls: 892,
    savingsAmount: 284500,
  });

  const [trends] = useState({
    alertsTrend: 12.5,
    fraudTrend: -8.3,
    riskTrend: 15.2,
    healthTrend: 2.1,
  });

  const chartData = [
    { name: 'Mon', alerts: 45, fraud: 12, blocked: 23 },
    { name: 'Tue', alerts: 52, fraud: 15, blocked: 28 },
    { name: 'Wed', alerts: 38, fraud: 8, blocked: 19 },
    { name: 'Thu', alerts: 71, fraud: 22, blocked: 35 },
    { name: 'Fri', alerts: 63, fraud: 18, blocked: 31 },
    { name: 'Sat', alerts: 29, fraud: 6, blocked: 14 },
    { name: 'Sun', alerts: 34, fraud: 9, blocked: 17 },
  ];

  const riskDistribution = [
    { name: 'Low Risk', value: 68, color: '#4caf50' },
    { name: 'Medium Risk', value: 23, color: '#ff9800' },
    { name: 'High Risk', value: 9, color: '#f44336' },
  ];

  const recentAlerts = [
    {
      id: '1',
      type: 'High Risk',
      message: 'Suspicious call pattern detected from user ID 4521',
      timestamp: '2 minutes ago',
      severity: 'error' as const,
      icon: Error,
    },
    {
      id: '2',
      type: 'Fraud Detected',
      message: 'Potential SIM box fraud in Region A',
      timestamp: '5 minutes ago',
      severity: 'error' as const,
      icon: ReportProblem,
    },
    {
      id: '3',
      type: 'System Alert',
      message: 'ML model accuracy decreased to 89%',
      timestamp: '12 minutes ago',
      severity: 'warning' as const,
      icon: Warning,
    },
    {
      id: '4',
      type: 'Info',
      message: 'Daily fraud report generated successfully',
      timestamp: '1 hour ago',
      severity: 'info' as const,
      icon: Info,
    },
  ];

  const handleRefresh = async () => {
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLastUpdate(new Date());
      setIsLoading(false);
    }, 1500);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      notation: 'compact',
    }).format(amount);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
            Fraud Detection Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={isLoading}
            sx={{ borderRadius: '8px' }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            sx={{ 
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            }}
            onClick={() => onOpenSection?.('live-alerts')}
          >
            View All Alerts
          </Button>
        </Box>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard onClick={() => onOpenSection?.('live-alerts')}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                    {formatNumber(metrics.totalAlerts)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Alerts
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp sx={{ fontSize: 16, color: trends.alertsTrend > 0 ? '#f44336' : '#4caf50', mr: 0.5 }} />
                    <Typography variant="caption" color={trends.alertsTrend > 0 ? 'error' : 'success.main'}>
                      {Math.abs(trends.alertsTrend)}% from yesterday
                    </Typography>
                  </Box>
                </Box>
                <Warning sx={{ fontSize: 32, color: '#ff9800', opacity: 0.7 }} />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard onClick={() => onOpenSection?.('fraud-detection')}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                    {formatNumber(metrics.fraudDetected)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Fraud Cases
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingDown sx={{ fontSize: 16, color: '#4caf50', mr: 0.5 }} />
                    <Typography variant="caption" color="success.main">
                      {Math.abs(trends.fraudTrend)}% reduction
                    </Typography>
                  </Box>
                </Box>
                <Security sx={{ fontSize: 32, color: '#f44336', opacity: 0.7 }} />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard onClick={() => onOpenSection?.('risk-scoring')}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                    {metrics.riskScore}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Risk Score
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp sx={{ fontSize: 16, color: '#f44336', mr: 0.5 }} />
                    <Typography variant="caption" color="error">
                      {trends.riskTrend}% increase
                    </Typography>
                  </Box>
                </Box>
                <Shield sx={{ fontSize: 32, color: '#667eea', opacity: 0.7 }} />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard onClick={() => onOpenSection?.('system-health')}>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                    {metrics.systemHealth}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    System Health
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <StatusIndicator status="online" />
                    <Typography variant="caption" color="success.main" sx={{ ml: 1 }}>
                      All systems operational
                    </Typography>
                  </Box>
                </Box>
                <Speed sx={{ fontSize: 32, color: '#4caf50', opacity: 0.7 }} />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <ChartCard>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Fraud Detection Trends (Last 7 Days)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={alpha('#1a1a2e', 0.1)} />
                  <XAxis dataKey="name" stroke={alpha('#1a1a2e', 0.6)} />
                  <YAxis stroke={alpha('#1a1a2e', 0.6)} />
                  <RechartsTooltip 
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="alerts" 
                    stroke="#ff9800" 
                    strokeWidth={3}
                    name="Total Alerts"
                    dot={{ fill: '#ff9800', strokeWidth: 2, r: 4 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="fraud" 
                    stroke="#f44336" 
                    strokeWidth={3}
                    name="Fraud Detected"
                    dot={{ fill: '#f44336', strokeWidth: 2, r: 4 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="blocked" 
                    stroke="#4caf50" 
                    strokeWidth={3}
                    name="Calls Blocked"
                    dot={{ fill: '#4caf50', strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </ChartCard>
        </Grid>

        <Grid item xs={12} md={4}>
          <ChartCard>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Risk Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                  >
                    {riskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </ChartCard>
        </Grid>
      </Grid>

      {/* Recent Alerts and Additional Metrics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <ChartCard>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Recent Alerts
              </Typography>
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {recentAlerts.map((alert) => (
                  <AlertCard 
                    key={alert.id} 
                    className={alert.severity}
                    sx={{ mb: 2 }}
                    onClick={() => onOpenSection?.('live-alerts')}
                  >
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                        <alert.icon 
                          fontSize="small" 
                          color={alert.severity}
                        />
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                            {alert.type}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            {alert.message}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {alert.timestamp}
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </AlertCard>
                ))}
              </Box>
            </CardContent>
          </ChartCard>
        </Grid>

        <Grid item xs={12} md={6}>
          <Grid container spacing={2}>
            {/* Additional Metrics */}
            <Grid item xs={6}>
              <MetricCard>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
                    {formatNumber(metrics.activeUsers)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Users
                  </Typography>
                </CardContent>
              </MetricCard>
            </Grid>
            
            <Grid item xs={6}>
              <MetricCard>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
                    {formatNumber(metrics.processedCalls)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Calls Processed
                  </Typography>
                </CardContent>
              </MetricCard>
            </Grid>

            <Grid item xs={6}>
              <MetricCard>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
                    {formatNumber(metrics.blockedCalls)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Calls Blocked
                  </Typography>
                </CardContent>
              </MetricCard>
            </Grid>

            <Grid item xs={6}>
              <MetricCard>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
                    {formatCurrency(metrics.savingsAmount)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Savings Today
                  </Typography>
                </CardContent>
              </MetricCard>
            </Grid>

            {/* Quick Actions */}
            <Grid item xs={12}>
              <Card>
                <CardContent sx={{ p: 2 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    Quick Actions
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    <Chip 
                      label="Create Report" 
                      clickable 
                      onClick={() => onOpenSection?.('custom-reports')}
                      sx={{ bgcolor: alpha('#667eea', 0.1), color: '#667eea' }}
                    />
                    <Chip 
                      label="View Cases" 
                      clickable 
                      onClick={() => onOpenSection?.('case-management')}
                      sx={{ bgcolor: alpha('#4caf50', 0.1), color: '#4caf50' }}
                    />
                    <Chip 
                      label="Export Data" 
                      clickable 
                      onClick={() => onOpenSection?.('data-export')}
                      sx={{ bgcolor: alpha('#ff9800', 0.1), color: '#ff9800' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
}