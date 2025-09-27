/**
 * System Monitoring Panel
 * Real-time system monitoring dashboard with metrics, logs, and performance monitoring
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  CircularProgress,
  styled,
} from '@mui/material';
import {
  Close,
  Monitor,
  Memory,
  Storage,
  NetworkCheck,
  Speed,
  Timeline,
  Warning,
  Error,
  CheckCircle,
  Refresh,
  Download,
  Settings,
  Computer,
  CloudQueue,
  Security,
  TrendingUp,
  TrendingDown,
  Remove,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const MetricCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  '&:hover': {
    borderColor: customColors.primary[300],
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
  },
}));

const StatusIndicator = styled(Box)<{ status: 'healthy' | 'warning' | 'critical' }>(({ status }) => ({
  width: 12,
  height: 12,
  borderRadius: '50%',
  backgroundColor: 
    status === 'healthy' ? customColors.success[500] :
    status === 'warning' ? customColors.warning[500] :
    '#d32f2f',
  marginRight: 8,
}));

interface SystemMonitoringPanelProps {
  open: boolean;
  onClose: () => void;
}

interface SystemMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  threshold: number;
}

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';
  service: string;
  message: string;
  details?: string;
}

const SystemMonitoringPanel: React.FC<SystemMonitoringPanelProps> = ({
  open,
  onClose
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([
    {
      id: 'cpu',
      name: 'CPU Usage',
      value: 67,
      unit: '%',
      status: 'warning',
      trend: 'up',
      threshold: 80
    },
    {
      id: 'memory',
      name: 'Memory Usage',
      value: 45,
      unit: '%',
      status: 'healthy',
      trend: 'stable',
      threshold: 85
    },
    {
      id: 'disk',
      name: 'Disk Usage',
      value: 32,
      unit: '%',
      status: 'healthy',
      trend: 'up',
      threshold: 90
    },
    {
      id: 'network',
      name: 'Network I/O',
      value: 156,
      unit: 'MB/s',
      status: 'healthy',
      trend: 'down',
      threshold: 1000
    },
    {
      id: 'transactions',
      name: 'Transactions/sec',
      value: 2847,
      unit: 'TPS',
      status: 'healthy',
      trend: 'up',
      threshold: 5000
    },
    {
      id: 'alerts',
      name: 'Active Alerts',
      value: 12,
      unit: 'alerts',
      status: 'warning',
      trend: 'up',
      threshold: 20
    }
  ]);

  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: '1',
      timestamp: new Date('2024-01-15T14:30:00'),
      level: 'INFO',
      service: 'FraudEngine',
      message: 'Transaction processing completed successfully',
      details: 'Processed 1,247 transactions in batch #2024-001'
    },
    {
      id: '2',
      timestamp: new Date('2024-01-15T14:28:15'),
      level: 'WARN',
      service: 'AlertSystem',
      message: 'High volume of alerts detected',
      details: 'Alert threshold exceeded: 12 alerts in the last 5 minutes'
    },
    {
      id: '3',
      timestamp: new Date('2024-01-15T14:25:42'),
      level: 'ERROR',
      service: 'DatabaseService',
      message: 'Connection timeout to secondary database',
      details: 'Failed to connect to db-replica-02 after 30 seconds'
    },
    {
      id: '4',
      timestamp: new Date('2024-01-15T14:22:11'),
      level: 'INFO',
      service: 'ApiGateway',
      message: 'API rate limit updated',
      details: 'Increased rate limit to 1000 requests/minute'
    },
    {
      id: '5',
      timestamp: new Date('2024-01-15T14:20:05'),
      level: 'DEBUG',
      service: 'MLService',
      message: 'Model inference completed',
      details: 'GraphSAGE model processed 500 nodes in 2.3 seconds'
    }
  ]);

  const [services] = useState([
    { name: 'API Gateway', status: 'healthy', uptime: '99.9%', responseTime: '45ms' },
    { name: 'Fraud Engine', status: 'healthy', uptime: '99.8%', responseTime: '127ms' },
    { name: 'ML Service', status: 'warning', uptime: '98.5%', responseTime: '234ms' },
    { name: 'Database Primary', status: 'healthy', uptime: '100%', responseTime: '12ms' },
    { name: 'Database Replica', status: 'critical', uptime: '87.2%', responseTime: '0ms' },
    { name: 'Message Queue', status: 'healthy', uptime: '99.7%', responseTime: '8ms' }
  ]);

  // Simulate real-time updates
  useEffect(() => {
    if (!open) return;

    const interval = setInterval(() => {
      setSystemMetrics(prev => prev.map(metric => ({
        ...metric,
        value: Math.max(0, Math.min(100, metric.value + (Math.random() - 0.5) * 10)),
        status: metric.value > metric.threshold * 0.9 ? 'critical' : 
                metric.value > metric.threshold * 0.7 ? 'warning' : 'healthy'
      })));

      setLastUpdate(new Date());
    }, 5000);

    return () => clearInterval(interval);
  }, [open]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setIsRefreshing(false);
      setLastUpdate(new Date());
      // In a real app, this would fetch fresh data from the API
    }, 1000);
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp fontSize="small" />;
      case 'down': return <TrendingDown fontSize="small" />;
      default: return <Remove fontSize="small" />;
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return '#d32f2f';
      case 'WARN': return customColors.warning[600];
      case 'INFO': return customColors.primary[600];
      case 'DEBUG': return customColors.neutral[600];
      default: return customColors.neutral[600];
    }
  };

  const renderSystemOverview = () => (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          System Overview
        </Typography>
        <Box display="flex" gap={1}>
          <Button
            size="small"
            startIcon={isRefreshing ? <CircularProgress size={16} /> : <Refresh />}
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            Refresh
          </Button>
          <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }}>
            Last updated: {lastUpdate.toLocaleTimeString()}
          </Typography>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {systemMetrics.map((metric) => (
          <Grid item xs={12} sm={6} md={4} key={metric.id}>
            <MetricCard>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {metric.name}
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 600, my: 1 }}>
                      {metric.value}{metric.unit}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1}>
                    <StatusIndicator status={metric.status} />
                    {getTrendIcon(metric.trend)}
                  </Box>
                </Box>
                
                <LinearProgress
                  variant="determinate"
                  value={metric.value}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: customColors.neutral[200],
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: 
                        metric.status === 'critical' ? '#d32f2f' :
                        metric.status === 'warning' ? customColors.warning[500] :
                        customColors.success[500]
                    }
                  }}
                />
                
                <Box display="flex" justifyContent="space-between" mt={1}>
                  <Typography variant="caption" color="text.secondary">
                    Threshold: {metric.threshold}{metric.unit}
                  </Typography>
                  <Chip 
                    label={metric.status.toUpperCase()} 
                    size="small"
                    sx={{
                      bgcolor: 
                        metric.status === 'critical' ? '#ffebee' :
                        metric.status === 'warning' ? customColors.warning[50] :
                        customColors.success[50],
                      color:
                        metric.status === 'critical' ? '#d32f2f' :
                        metric.status === 'warning' ? customColors.warning[700] :
                        customColors.success[700]
                    }}
                  />
                </Box>
              </CardContent>
            </MetricCard>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12}>
          <MetricCard>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Service Health Status
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Service</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">Uptime</TableCell>
                      <TableCell align="right">Response Time</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {services.map((service, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <StatusIndicator status={service.status as any} />
                            {service.name}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={service.status.toUpperCase()}
                            size="small"
                            sx={{
                              bgcolor: 
                                service.status === 'critical' ? '#ffebee' :
                                service.status === 'warning' ? customColors.warning[50] :
                                customColors.success[50],
                              color:
                                service.status === 'critical' ? '#d32f2f' :
                                service.status === 'warning' ? customColors.warning[700] :
                                customColors.success[700]
                            }}
                          />
                        </TableCell>
                        <TableCell align="right">{service.uptime}</TableCell>
                        <TableCell align="right">{service.responseTime}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </MetricCard>
        </Grid>
      </Grid>
    </Box>
  );

  const renderSystemLogs = () => (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          System Logs
        </Typography>
        <Box display="flex" gap={1}>
          <Button size="small" startIcon={<Download />}>
            Export Logs
          </Button>
          <Button size="small" startIcon={<Settings />}>
            Log Settings
          </Button>
        </Box>
      </Box>

      <MetricCard>
        <CardContent>
          <List sx={{ maxHeight: '400px', overflow: 'auto' }}>
            {logs.map((log) => (
              <ListItem key={log.id} divider>
                <ListItemIcon>
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: getLogLevelColor(log.level)
                    }}
                  />
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={2}>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {log.timestamp.toLocaleTimeString()}
                      </Typography>
                      <Chip
                        label={log.level}
                        size="small"
                        sx={{
                          backgroundColor: getLogLevelColor(log.level),
                          color: 'white',
                          fontWeight: 600,
                          minWidth: 60
                        }}
                      />
                      <Chip
                        label={log.service}
                        size="small"
                        variant="outlined"
                      />
                      <Typography variant="body2">
                        {log.message}
                      </Typography>
                    </Box>
                  }
                  secondary={log.details && (
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ mt: 0.5, fontFamily: 'monospace', fontSize: '0.8rem' }}
                    >
                      {log.details}
                    </Typography>
                  )}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </MetricCard>
    </Box>
  );

  const renderPerformanceMetrics = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Performance Metrics
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <MetricCard>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Transaction Processing
              </Typography>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Throughput</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>2,847 TPS</Typography>
                </Box>
                <LinearProgress variant="determinate" value={68} sx={{ height: 6, borderRadius: 3 }} />
              </Box>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Average Response Time</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>127ms</Typography>
                </Box>
                <LinearProgress variant="determinate" value={25} sx={{ height: 6, borderRadius: 3 }} />
              </Box>
              <Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Error Rate</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>0.03%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={0.03} 
                  sx={{ 
                    height: 6, 
                    borderRadius: 3,
                    '& .MuiLinearProgress-bar': { backgroundColor: customColors.success[500] }
                  }} 
                />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>

        <Grid item xs={12} md={6}>
          <MetricCard>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Fraud Detection Engine
              </Typography>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Model Accuracy</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>97.8%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={97.8} 
                  sx={{ 
                    height: 6, 
                    borderRadius: 3,
                    '& .MuiLinearProgress-bar': { backgroundColor: customColors.success[500] }
                  }} 
                />
              </Box>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">False Positive Rate</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>2.1%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={2.1} 
                  sx={{ 
                    height: 6, 
                    borderRadius: 3,
                    '& .MuiLinearProgress-bar': { backgroundColor: customColors.warning[500] }
                  }} 
                />
              </Box>
              <Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Processing Speed</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>1,245 req/sec</Typography>
                </Box>
                <LinearProgress variant="determinate" value={78} sx={{ height: 6, borderRadius: 3 }} />
              </Box>
            </CardContent>
          </MetricCard>
        </Grid>
      </Grid>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12}>
          <Alert severity="info">
            <Typography variant="body2">
              <strong>Performance Summary:</strong> All systems are operating within normal parameters. 
              The fraud detection engine is performing optimally with high accuracy and low false positive rates.
            </Typography>
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );

  const tabLabels = ['System Overview', 'System Logs', 'Performance'];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '80vh',
          backgroundColor: customColors.background.default,
        }
      }}
    >
      <DialogTitle sx={{ 
        backgroundColor: customColors.background.ribbon,
        borderBottom: `2px solid ${customColors.primary[500]}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Box display="flex" alignItems="center" gap={2}>
          <Monitor color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            System Monitoring Dashboard
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>

      <Box sx={{ backgroundColor: customColors.background.ribbon }}>
        <Tabs
          value={currentTab}
          onChange={(_, newValue) => setCurrentTab(newValue)}
          variant="fullWidth"
        >
          {tabLabels.map((label, index) => (
            <Tab key={label} label={label} />
          ))}
        </Tabs>
      </Box>

      <DialogContent sx={{ p: 0, height: '60vh', overflow: 'auto' }}>
        {currentTab === 0 && renderSystemOverview()}
        {currentTab === 1 && renderSystemLogs()}
        {currentTab === 2 && renderPerformanceMetrics()}
      </DialogContent>

      <DialogActions sx={{ 
        p: 2, 
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SystemMonitoringPanel;