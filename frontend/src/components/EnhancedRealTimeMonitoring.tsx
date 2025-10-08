/**
 * Real-time Monitoring Dashboard
 * Live monitoring of fraud detection systems with real-time updates
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  IconButton,
  Button,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Tooltip,
  LinearProgress,
  styled,
  alpha,
  useTheme,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Refresh,
  Settings,
  Fullscreen,
  Warning,
  Error,
  CheckCircle,
  Speed,
  NetworkCheck,
  Security,
  Visibility,
  Timeline,
  TrendingUp,
  SignalCellular4Bar,
  DataUsage,
  Storage,
  Memory,
  Computer,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from 'recharts';

interface RealTimeMonitoringProps {
  onOpenAlert?: (alertId: string) => void;
  onOpenSystemHealth?: () => void;
}

const MonitoringCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
  },
}));

const MetricDisplay = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexDirection: 'column',
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.primary.main, 0.05),
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
}));

const StatusIndicator = styled(Box)<{ status: 'online' | 'warning' | 'error' | 'processing' }>(({ status }) => ({
  width: 12,
  height: 12,
  borderRadius: '50%',
  backgroundColor: 
    status === 'online' ? '#4caf50' : 
    status === 'warning' ? '#ff9800' : 
    status === 'error' ? '#f44336' : '#2196f3',
  animation: status !== 'online' ? 'pulse 2s infinite' : 'none',
  '@keyframes pulse': {
    '0%': { opacity: 1 },
    '50%': { opacity: 0.5 },
    '100%': { opacity: 1 },
  },
}));

const AlertItem = styled(ListItem)<{ severity: 'high' | 'medium' | 'low' }>(({ severity }) => ({
  borderLeft: `4px solid ${
    severity === 'high' ? '#f44336' : 
    severity === 'medium' ? '#ff9800' : '#4caf50'
  }`,
  marginBottom: 8,
  borderRadius: '0 8px 8px 0',
  backgroundColor: alpha(
    severity === 'high' ? '#f44336' : 
    severity === 'medium' ? '#ff9800' : '#4caf50', 
    0.05
  ),
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  '&:hover': {
    backgroundColor: alpha(
      severity === 'high' ? '#f44336' : 
      severity === 'medium' ? '#ff9800' : '#4caf50', 
      0.1
    ),
  },
}));

export default function RealTimeMonitoring({ onOpenAlert, onOpenSystemHealth }: RealTimeMonitoringProps) {
  const theme = useTheme();
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Real-time data states
  const [systemMetrics, setSystemMetrics] = useState({
    cpuUsage: 72,
    memoryUsage: 68,
    networkThroughput: 85,
    alertsPerMinute: 12,
    fraudDetectionRate: 94.5,
    systemHealth: 96,
    activeConnections: 1247,
    processedEvents: 45629,
  });

  const [liveAlerts, setLiveAlerts] = useState([
    {
      id: '1',
      type: 'High Risk Pattern',
      message: 'Suspicious calling pattern detected from user 4521',
      severity: 'high' as const,
      timestamp: new Date(Date.now() - 2 * 60000),
      source: 'ML Engine',
    },
    {
      id: '2',
      type: 'Fraud Detection',
      message: 'Potential SIM box activity in Region A',
      severity: 'high' as const,
      timestamp: new Date(Date.now() - 5 * 60000),
      source: 'Graph Analytics',
    },
    {
      id: '3',
      type: 'Threshold Alert',
      message: 'Call volume spike detected - 150% above baseline',
      severity: 'medium' as const,
      timestamp: new Date(Date.now() - 8 * 60000),
      source: 'Stream Processor',
    },
    {
      id: '4',
      type: 'System Warning',
      message: 'Model accuracy dropped to 89% - retraining recommended',
      severity: 'medium' as const,
      timestamp: new Date(Date.now() - 12 * 60000),
      source: 'ML Monitor',
    },
    {
      id: '5',
      type: 'Info',
      message: 'Daily backup completed successfully',
      severity: 'low' as const,
      timestamp: new Date(Date.now() - 20 * 60000),
      source: 'System',
    },
  ]);

  const [performanceData, setPerformanceData] = useState([
    { time: '14:50', alerts: 8, fraud: 2, cpu: 65, memory: 72 },
    { time: '14:51', alerts: 12, fraud: 3, cpu: 68, memory: 74 },
    { time: '14:52', alerts: 6, fraud: 1, cpu: 70, memory: 73 },
    { time: '14:53', alerts: 15, fraud: 4, cpu: 75, memory: 76 },
    { time: '14:54', alerts: 9, fraud: 2, cpu: 72, memory: 75 },
    { time: '14:55', alerts: 18, fraud: 5, cpu: 78, memory: 77 },
    { time: '14:56', alerts: 11, fraud: 3, cpu: 74, memory: 76 },
    { time: '14:57', alerts: 13, fraud: 4, cpu: 76, memory: 78 },
    { time: '14:58', alerts: 7, fraud: 1, cpu: 73, memory: 75 },
    { time: '14:59', alerts: 12, fraud: 3, cpu: 72, memory: 68 },
  ]);

  // Simulate real-time updates
  useEffect(() => {
    if (!isMonitoring || !autoRefresh) return;

    const interval = setInterval(() => {
      // Update metrics
      setSystemMetrics(prev => ({
        ...prev,
        cpuUsage: Math.max(40, Math.min(95, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(50, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 8)),
        networkThroughput: Math.max(60, Math.min(100, prev.networkThroughput + (Math.random() - 0.5) * 15)),
        alertsPerMinute: Math.max(5, Math.min(30, prev.alertsPerMinute + (Math.random() - 0.5) * 5)),
        processedEvents: prev.processedEvents + Math.floor(Math.random() * 50) + 10,
      }));

      // Update performance data
      setPerformanceData(prev => {
        const newData = [...prev.slice(1)];
        const lastTime = prev[prev.length - 1].time;
        const [hours, minutes] = lastTime.split(':').map(Number);
        const newMinutes = (minutes + 1) % 60;
        const newHours = newMinutes === 0 ? (hours + 1) % 24 : hours;
        const newTime = `${newHours.toString().padStart(2, '0')}:${newMinutes.toString().padStart(2, '0')}`;
        
        newData.push({
          time: newTime,
          alerts: Math.floor(Math.random() * 20) + 5,
          fraud: Math.floor(Math.random() * 6) + 1,
          cpu: Math.floor(Math.random() * 30) + 60,
          memory: Math.floor(Math.random() * 25) + 65,
        });
        return newData;
      });

      setLastUpdate(new Date());
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [isMonitoring, autoRefresh]);

  // Add new alert occasionally
  useEffect(() => {
    if (!isMonitoring) return;

    const alertInterval = setInterval(() => {
      if (Math.random() < 0.3) { // 30% chance of new alert
        const alertTypes = [
          { type: 'Fraud Detection', severity: 'high' as const, source: 'ML Engine' },
          { type: 'Suspicious Pattern', severity: 'medium' as const, source: 'Graph Analytics' },
          { type: 'System Alert', severity: 'low' as const, source: 'Monitor' },
        ];
        
        const randomAlert = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        const newAlert = {
          id: Date.now().toString(),
          type: randomAlert.type,
          message: `New ${randomAlert.type.toLowerCase()} detected`,
          severity: randomAlert.severity,
          timestamp: new Date(),
          source: randomAlert.source,
        };

        setLiveAlerts(prev => [newAlert, ...prev.slice(0, 9)]); // Keep only last 10 alerts
      }
    }, 15000); // Check for new alerts every 15 seconds

    return () => clearInterval(alertInterval);
  }, [isMonitoring]);

  const handleToggleMonitoring = () => {
    setIsMonitoring(!isMonitoring);
  };

  const handleRefresh = () => {
    setLastUpdate(new Date());
    // Force update all data
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
  };

  const getSystemStatus = () => {
    if (systemMetrics.systemHealth >= 95) return 'online';
    if (systemMetrics.systemHealth >= 85) return 'warning';
    return 'error';
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
            Real-time Monitoring
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <StatusIndicator status={getSystemStatus()} />
            <Typography variant="body2" color="text.secondary">
              Last updated: {formatTime(lastUpdate)}
            </Typography>
            <Chip 
              label={isMonitoring ? 'LIVE' : 'PAUSED'} 
              color={isMonitoring ? 'success' : 'default'}
              size="small"
              sx={{ fontWeight: 600 }}
            />
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                size="small"
              />
            }
            label="Auto-refresh"
            sx={{ mr: 2 }}
          />
          
          <IconButton onClick={handleRefresh} disabled={!isMonitoring}>
            <Refresh />
          </IconButton>
          
          <Button
            variant={isMonitoring ? 'outlined' : 'contained'}
            startIcon={isMonitoring ? <Pause /> : <PlayArrow />}
            onClick={handleToggleMonitoring}
            sx={{ borderRadius: '8px' }}
          >
            {isMonitoring ? 'Pause' : 'Start'} Monitoring
          </Button>
        </Box>
      </Box>

      {/* System Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MonitoringCard>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  CPU Usage
                </Typography>
                <Computer sx={{ fontSize: 20, color: '#667eea' }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                {systemMetrics.cpuUsage}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemMetrics.cpuUsage} 
                sx={{ 
                  height: 6, 
                  borderRadius: 3,
                  backgroundColor: alpha('#667eea', 0.1),
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: systemMetrics.cpuUsage > 80 ? '#f44336' : '#667eea',
                  },
                }}
              />
            </CardContent>
          </MonitoringCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MonitoringCard>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Memory Usage
                </Typography>
                <Memory sx={{ fontSize: 20, color: '#4caf50' }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                {systemMetrics.memoryUsage}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemMetrics.memoryUsage} 
                sx={{ 
                  height: 6, 
                  borderRadius: 3,
                  backgroundColor: alpha('#4caf50', 0.1),
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: systemMetrics.memoryUsage > 80 ? '#f44336' : '#4caf50',
                  },
                }}
              />
            </CardContent>
          </MonitoringCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MonitoringCard>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Network I/O
                </Typography>
                <NetworkCheck sx={{ fontSize: 20, color: '#ff9800' }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                {systemMetrics.networkThroughput}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemMetrics.networkThroughput} 
                sx={{ 
                  height: 6, 
                  borderRadius: 3,
                  backgroundColor: alpha('#ff9800', 0.1),
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: '#ff9800',
                  },
                }}
              />
            </CardContent>
          </MonitoringCard>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MonitoringCard>
            <CardContent sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  System Health
                </Typography>
                <Speed sx={{ fontSize: 20, color: getSystemStatus() === 'online' ? '#4caf50' : '#f44336' }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#1a1a2e', mb: 1 }}>
                {systemMetrics.systemHealth}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemMetrics.systemHealth} 
                sx={{ 
                  height: 6, 
                  borderRadius: 3,
                  backgroundColor: alpha('#4caf50', 0.1),
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getSystemStatus() === 'online' ? '#4caf50' : '#f44336',
                  },
                }}
              />
            </CardContent>
          </MonitoringCard>
        </Grid>
      </Grid>

      {/* Charts and Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <MonitoringCard>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Real-time Performance Metrics
              </Typography>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={alpha('#1a1a2e', 0.1)} />
                  <XAxis dataKey="time" stroke={alpha('#1a1a2e', 0.6)} />
                  <YAxis stroke={alpha('#1a1a2e', 0.6)} />
                  <RechartsTooltip 
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="alerts" 
                    stroke="#ff9800" 
                    strokeWidth={2}
                    name="Alerts/min"
                    dot={{ fill: '#ff9800', strokeWidth: 2, r: 3 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="fraud" 
                    stroke="#f44336" 
                    strokeWidth={2}
                    name="Fraud Detected"
                    dot={{ fill: '#f44336', strokeWidth: 2, r: 3 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="cpu" 
                    stroke="#667eea" 
                    strokeWidth={2}
                    name="CPU %"
                    dot={{ fill: '#667eea', strokeWidth: 2, r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </MonitoringCard>
        </Grid>

        <Grid item xs={12} md={4}>
          <MonitoringCard>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Live Alerts
                </Typography>
                <Chip 
                  label={liveAlerts.length}
                  color="error"
                  size="small"
                />
              </Box>
              
              <Box sx={{ maxHeight: 320, overflow: 'auto' }}>
                <List sx={{ p: 0 }}>
                  {liveAlerts.map((alert) => (
                    <AlertItem 
                      key={alert.id} 
                      severity={alert.severity}
                      onClick={() => onOpenAlert?.(alert.id)}
                    >
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        {alert.severity === 'high' && <Error color="error" fontSize="small" />}
                        {alert.severity === 'medium' && <Warning color="warning" fontSize="small" />}
                        {alert.severity === 'low' && <CheckCircle color="success" fontSize="small" />}
                      </ListItemIcon>
                      
                      <ListItemText
                        primary={
                          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            {alert.type}
                          </Typography>
                        }
                        secondary={
                          <>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                              {alert.message}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {alert.source} • {formatTime(alert.timestamp)}
                            </Typography>
                          </>
                        }
                      />
                    </AlertItem>
                  ))}
                </List>
              </Box>
            </CardContent>
          </MonitoringCard>
        </Grid>
      </Grid>

      {/* Additional Metrics Row */}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricDisplay>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
              {systemMetrics.alertsPerMinute}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Alerts/Minute
            </Typography>
          </MetricDisplay>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricDisplay>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
              {systemMetrics.fraudDetectionRate}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Detection Rate
            </Typography>
          </MetricDisplay>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricDisplay>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
              {new Intl.NumberFormat().format(systemMetrics.activeConnections)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Active Connections
            </Typography>
          </MetricDisplay>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricDisplay>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1a1a2e' }}>
              {new Intl.NumberFormat().format(systemMetrics.processedEvents)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Events Processed
            </Typography>
          </MetricDisplay>
        </Grid>
      </Grid>
    </Box>
  );
}