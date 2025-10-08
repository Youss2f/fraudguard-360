import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
  Button,
  IconButton,
  Tooltip,
  Badge,
  Switch,
  FormControlLabel,
  Divider
} from '@mui/material';
import {
  Timeline,
  Security,
  Warning,
  TrendingUp,
  Speed,
  NetworkCheck,
  Refresh,
  Pause,
  PlayArrow,
  Notifications,
  Error,
  CheckCircle,
  Memory,
  Storage,
  Router
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';

interface RealTimeMetrics {
  timestamp: Date;
  transactionsPerSecond: number;
  alertsGenerated: number;
  suspiciousActivities: number;
  systemLoad: number;
  memoryUsage: number;
  networkLatency: number;
}

interface SystemStatus {
  service: string;
  status: 'healthy' | 'warning' | 'error';
  uptime: string;
  version: string;
  lastCheck: Date;
}

interface LiveAlert {
  id: string;
  type: 'high' | 'medium' | 'low';
  message: string;
  timestamp: Date;
  source: string;
  acknowledged: boolean;
}

const RealTimeMonitoring: React.FC = () => {
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [metrics, setMetrics] = useState<RealTimeMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<RealTimeMetrics>({
    timestamp: new Date(),
    transactionsPerSecond: 1250,
    alertsGenerated: 23,
    suspiciousActivities: 7,
    systemLoad: 68,
    memoryUsage: 72,
    networkLatency: 45
  });

  const [systemServices, setSystemServices] = useState<SystemStatus[]>([
    { service: 'API Gateway', status: 'healthy', uptime: '99.9%', version: '1.2.3', lastCheck: new Date() },
    { service: 'ML Engine', status: 'healthy', uptime: '99.7%', version: '2.1.0', lastCheck: new Date() },
    { service: 'Stream Processor', status: 'warning', uptime: '98.5%', version: '1.8.2', lastCheck: new Date() },
    { service: 'Graph Database', status: 'healthy', uptime: '99.8%', version: '4.4.5', lastCheck: new Date() },
    { service: 'Message Queue', status: 'healthy', uptime: '99.9%', version: '2.8.0', lastCheck: new Date() },
    { service: 'Alert Manager', status: 'error', uptime: '95.2%', version: '1.5.1', lastCheck: new Date() }
  ]);

  const [liveAlerts, setLiveAlerts] = useState<LiveAlert[]>([
    { id: '1', type: 'high', message: 'Suspicious transaction pattern detected for User ID 12345', timestamp: new Date(), source: 'ML Engine', acknowledged: false },
    { id: '2', type: 'medium', message: 'High frequency calls from number +1-555-0123', timestamp: new Date(Date.now() - 60000), source: 'Stream Processor', acknowledged: false },
    { id: '3', type: 'high', message: 'Potential SIM box detected at location: Chicago, IL', timestamp: new Date(Date.now() - 120000), source: 'Location Analytics', acknowledged: true },
    { id: '4', type: 'low', message: 'Minor system performance degradation detected', timestamp: new Date(Date.now() - 180000), source: 'System Monitor', acknowledged: false }
  ]);

  // Generate mock real-time data
  useEffect(() => {
    if (!isMonitoring || !autoRefresh) return;

    const interval = setInterval(() => {
      const newMetric: RealTimeMetrics = {
        timestamp: new Date(),
        transactionsPerSecond: 1200 + Math.random() * 300,
        alertsGenerated: Math.floor(Math.random() * 50),
        suspiciousActivities: Math.floor(Math.random() * 20),
        systemLoad: 60 + Math.random() * 30,
        memoryUsage: 65 + Math.random() * 25,
        networkLatency: 35 + Math.random() * 30
      };

      setCurrentMetrics(newMetric);
      setMetrics(prev => [...prev.slice(-19), newMetric]);
    }, 2000);

    return () => clearInterval(interval);
  }, [isMonitoring, autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#4caf50';
      case 'warning': return '#ff9800';
      case 'error': return '#f44336';
      default: return '#757575';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle sx={{ color: '#4caf50' }} />;
      case 'warning': return <Warning sx={{ color: '#ff9800' }} />;
      case 'error': return <Error sx={{ color: '#f44336' }} />;
      default: return <CheckCircle sx={{ color: '#757575' }} />;
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const pieData = [
    { name: 'Normal', value: 85, color: '#4caf50' },
    { name: 'Suspicious', value: 12, color: '#ff9800' },
    { name: 'Fraudulent', value: 3, color: '#f44336' }
  ];

  return (
    <Box sx={{ p: 3, minHeight: '100vh', bgcolor: '#f5f5f5' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          Real-Time Monitoring
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                color="primary"
              />
            }
            label="Auto Refresh"
          />
          <Button
            variant="outlined"
            startIcon={isMonitoring ? <Pause /> : <PlayArrow />}
            onClick={() => setIsMonitoring(!isMonitoring)}
            color={isMonitoring ? 'secondary' : 'primary'}
          >
            {isMonitoring ? 'Pause' : 'Resume'}
          </Button>
          <Button
            variant="contained"
            startIcon={<Refresh />}
            onClick={() => window.location.reload()}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {currentMetrics.transactionsPerSecond.toFixed(0)}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Transactions/sec
                  </Typography>
                </Box>
                <Speed sx={{ fontSize: 48, opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #f57c00 0%, #ffb74d 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {currentMetrics.alertsGenerated}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Active Alerts
                  </Typography>
                </Box>
                <Warning sx={{ fontSize: 48, opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #d32f2f 0%, #f44336 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {currentMetrics.suspiciousActivities}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Suspicious Activities
                  </Typography>
                </Box>
                <Security sx={{ fontSize: 48, opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #388e3c 0%, #4caf50 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {currentMetrics.systemLoad.toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    System Load
                  </Typography>
                </Box>
                <Memory sx={{ fontSize: 48, opacity: 0.8 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Transaction Flow (Last 20 intervals)
            </Typography>
            <ResponsiveContainer width="100%" height="85%">
              <AreaChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis />
                <RechartsTooltip 
                  labelFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <Area 
                  type="monotone" 
                  dataKey="transactionsPerSecond" 
                  stroke="#1976d2" 
                  fill="rgba(25, 118, 210, 0.3)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Transaction Classification
            </Typography>
            <ResponsiveContainer width="100%" height="70%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* System Status and Live Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              System Services Status
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Service</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Uptime</TableCell>
                    <TableCell>Version</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {systemServices.map((service, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getStatusIcon(service.status)}
                          {service.service}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={service.status.toUpperCase()}
                          size="small"
                          sx={{
                            backgroundColor: getStatusColor(service.status),
                            color: 'white',
                            fontWeight: 'bold'
                          }}
                        />
                      </TableCell>
                      <TableCell>{service.uptime}</TableCell>
                      <TableCell>{service.version}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Live Alerts
              <Badge badgeContent={liveAlerts.filter(a => !a.acknowledged).length} color="error" sx={{ ml: 2 }}>
                <Notifications />
              </Badge>
            </Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {liveAlerts.map((alert) => (
                <Alert
                  key={alert.id}
                  severity={getAlertColor(alert.type) as any}
                  sx={{ mb: 1, opacity: alert.acknowledged ? 0.6 : 1 }}
                  action={
                    !alert.acknowledged && (
                      <Button 
                        size="small" 
                        onClick={() => {
                          setLiveAlerts(prev => 
                            prev.map(a => a.id === alert.id ? { ...a, acknowledged: true } : a)
                          );
                        }}
                      >
                        ACK
                      </Button>
                    )
                  }
                >
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {alert.message}
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      {alert.source} • {alert.timestamp.toLocaleTimeString()}
                    </Typography>
                  </Box>
                </Alert>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* System Performance Metrics */}
      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Memory Usage
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                {currentMetrics.memoryUsage.toFixed(0)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={currentMetrics.memoryUsage} 
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Network Latency
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                {currentMetrics.networkLatency.toFixed(0)}ms
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={(currentMetrics.networkLatency / 100) * 100} 
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Active Connections
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                1,247
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Real-time monitoring connections
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RealTimeMonitoring;