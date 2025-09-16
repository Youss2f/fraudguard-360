/**
 * Comprehensive Fraud Detection Dashboard
 * Professional fraud monitoring interface with real-time metrics, alerts, and investigation tools
 */

import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Box,
  Chip,
  IconButton,
  Button,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Tooltip,
  Badge,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Divider,
  Alert,
  Fab
} from '@mui/material';
import {
  TrendingUp,
  Security,
  Warning,
  Block,
  Refresh,
  Search,
  FilterList,
  Visibility,
  Assignment,
  Phone,
  Public,
  Timeline,
  Notifications,
  Settings,
  Assessment,
  Schedule,
  LocationOn,
  Person,
  MoreVert
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
  ResponsiveContainer
} from 'recharts';
import { webSocketService, FraudAlert, RealTimeMetrics } from '../services/WebSocketService';

interface DashboardProps {
  onGraphViewClick?: () => void;
}

interface Alert extends FraudAlert {
  timeAgo: string;
}

const FraudDetectionDashboard: React.FC<DashboardProps> = ({ onGraphViewClick }) => {
  const [metrics, setMetrics] = useState<RealTimeMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filteredAlerts, setFilteredAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [isConnected, setIsConnected] = useState(false);
  const [historicalData, setHistoricalData] = useState<any[]>([]);

  useEffect(() => {
    // Connect to WebSocket service
    const initializeConnection = async () => {
      try {
        await webSocketService.connect();
        setIsConnected(true);

        // Subscribe to real-time data streams
        webSocketService.subscribe('real-time-metrics', handleMetricsUpdate);
        webSocketService.subscribe('fraud-alerts', handleNewAlert);

        // Generate some historical data for charts
        generateHistoricalData();
      } catch (error) {
        console.error('Failed to connect to WebSocket service:', error);
      }
    };

    initializeConnection();

    return () => {
      webSocketService.disconnect();
      setIsConnected(false);
    };
  }, []);

  useEffect(() => {
    // Filter alerts based on search and filters
    let filtered = alerts.filter(alert => {
      const matchesSearch = alert.phoneNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          alert.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          alert.location.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesStatus = statusFilter === 'all' || alert.status === statusFilter;
      
      const matchesRisk = riskFilter === 'all' || 
                         (riskFilter === 'high' && alert.riskScore >= 70) ||
                         (riskFilter === 'medium' && alert.riskScore >= 40 && alert.riskScore < 70) ||
                         (riskFilter === 'low' && alert.riskScore < 40);

      return matchesSearch && matchesStatus && matchesRisk;
    });

    setFilteredAlerts(filtered);
  }, [alerts, searchTerm, statusFilter, riskFilter]);

  const handleMetricsUpdate = (newMetrics: RealTimeMetrics) => {
    setMetrics(newMetrics);
    
    // Add to historical data for charts
    setHistoricalData(prev => {
      const updated = [...prev, {
        time: new Date(newMetrics.timestamp).toLocaleTimeString(),
        fraudAlerts: newMetrics.fraudAlertsCount,
        suspiciousCalls: newMetrics.suspiciousCallsCount,
        totalCalls: newMetrics.totalCalls,
        blockedCalls: newMetrics.blockedCallsCount
      }];
      
      // Keep only last 20 data points
      return updated.slice(-20);
    });
  };

  const handleNewAlert = (newAlert: FraudAlert) => {
    const alertWithTimeAgo = {
      ...newAlert,
      timeAgo: 'Just now'
    };

    setAlerts(prev => {
      const updated = [alertWithTimeAgo, ...prev];
      return updated.slice(0, 100); // Keep only latest 100 alerts
    });
  };

  const generateHistoricalData = () => {
    const data = [];
    for (let i = 19; i >= 0; i--) {
      const time = new Date(Date.now() - i * 60000).toLocaleTimeString();
      data.push({
        time,
        fraudAlerts: 120 + Math.floor(Math.random() * 80),
        suspiciousCalls: 800 + Math.floor(Math.random() * 400),
        totalCalls: 50000 + Math.floor(Math.random() * 20000),
        blockedCalls: 250 + Math.floor(Math.random() * 150)
      });
    }
    setHistoricalData(data);
  };

  const getRiskColor = (riskScore: number): string => {
    if (riskScore >= 80) return '#d32f2f';
    if (riskScore >= 60) return '#f57c00';
    if (riskScore >= 40) return '#fbc02d';
    return '#388e3c';
  };

  const getStatusColor = (status: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'NEW': return 'error';
      case 'INVESTIGATING': return 'warning';
      case 'CONFIRMED': return 'error';
      case 'FALSE_POSITIVE': return 'success';
      default: return 'default';
    }
  };

  const handleAssignAlert = (alertId: string, assignee: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId 
        ? { ...alert, assignedTo: assignee, status: 'INVESTIGATING' as const }
        : alert
    ));
  };

  const handleStatusChange = (alertId: string, newStatus: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId 
        ? { ...alert, status: newStatus as any }
        : alert
    ));
  };

  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <Box sx={{ flexGrow: 1, p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            🛡️ Fraud Detection Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip 
              icon={isConnected ? <Security /> : <Warning />}
              label={isConnected ? 'System Online' : 'System Offline'}
              color={isConnected ? 'success' : 'error'}
              variant="outlined"
            />
            <Typography variant="body2" color="text.secondary">
              Last updated: {metrics ? new Date(metrics.timestamp).toLocaleTimeString() : 'Never'}
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<Timeline />}
            onClick={onGraphViewClick}
          >
            Network Graph
          </Button>
          <IconButton>
            <Settings />
          </IconButton>
          <IconButton>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Total Calls
                  </Typography>
                  <Typography variant="h4">
                    {metrics?.totalCalls.toLocaleString() || '0'}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  <Phone />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Fraud Alerts
                  </Typography>
                  <Typography variant="h4" color="error.main">
                    {metrics?.fraudAlertsCount || '0'}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'error.main' }}>
                  <Warning />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Suspicious Calls
                  </Typography>
                  <Typography variant="h4" color="warning.main">
                    {metrics?.suspiciousCallsCount || '0'}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'warning.main' }}>
                  <Security />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Blocked Calls
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {metrics?.blockedCallsCount || '0'}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'success.main' }}>
                  <Block />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Real-time Activity Trends
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Line type="monotone" dataKey="fraudAlerts" stroke="#f44336" strokeWidth={2} name="Fraud Alerts" />
                <Line type="monotone" dataKey="suspiciousCalls" stroke="#ff9800" strokeWidth={2} name="Suspicious Calls" />
                <Line type="monotone" dataKey="blockedCalls" stroke="#4caf50" strokeWidth={2} name="Blocked Calls" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Risk Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={metrics ? [
                    { name: 'Low Risk', value: metrics.riskDistribution.low, color: '#4caf50' },
                    { name: 'Medium Risk', value: metrics.riskDistribution.medium, color: '#fbc02d' },
                    { name: 'High Risk', value: metrics.riskDistribution.high, color: '#f57c00' },
                    { name: 'Critical', value: metrics.riskDistribution.critical, color: '#d32f2f' }
                  ] : []}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label
                >
                  {metrics && [
                    { name: 'Low Risk', value: metrics.riskDistribution.low, color: '#4caf50' },
                    { name: 'Medium Risk', value: metrics.riskDistribution.medium, color: '#fbc02d' },
                    { name: 'High Risk', value: metrics.riskDistribution.high, color: '#f57c00' },
                    { name: 'Critical', value: metrics.riskDistribution.critical, color: '#d32f2f' }
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Alerts Section */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">
                Live Fraud Alerts ({filteredAlerts.length})
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  size="small"
                  placeholder="Search alerts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  InputProps={{
                    startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                  }}
                />
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Status</InputLabel>
                  <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="NEW">New</MenuItem>
                    <MenuItem value="INVESTIGATING">Investigating</MenuItem>
                    <MenuItem value="CONFIRMED">Confirmed</MenuItem>
                    <MenuItem value="FALSE_POSITIVE">False Positive</MenuItem>
                  </Select>
                </FormControl>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Risk</InputLabel>
                  <Select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="high">High (70%+)</MenuItem>
                    <MenuItem value="medium">Medium (40-70%)</MenuItem>
                    <MenuItem value="low">Low (&lt;40%)</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </Box>

            <TableContainer sx={{ maxHeight: 500 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Time</TableCell>
                    <TableCell>Phone Number</TableCell>
                    <TableCell>Risk Score</TableCell>
                    <TableCell>Alert Type</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredAlerts.slice(0, 20).map((alert) => (
                    <TableRow key={alert.id} hover>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {alert.timeAgo}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {alert.phoneNumber}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={alert.riskScore}
                            sx={{
                              width: 60,
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: getRiskColor(alert.riskScore)
                              }
                            }}
                          />
                          <Typography variant="body2" color={getRiskColor(alert.riskScore)}>
                            {alert.riskScore}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={alert.alertType.replace('_', ' ')}
                          size="small"
                          color={alert.alertType === 'FRAUD_DETECTED' ? 'error' : 'warning'}
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <LocationOn fontSize="small" color="action" />
                          <Typography variant="body2">
                            {alert.location}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={alert.status}
                          size="small"
                          color={getStatusColor(alert.status)}
                        />
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => setSelectedAlert(alert)}
                        >
                          <Visibility />
                        </IconButton>
                        <IconButton size="small">
                          <Assignment />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Top Risk Countries
            </Typography>
            <List>
              {metrics?.topRiskCountries.map((country, index) => (
                <ListItem key={country.country}>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: getRiskColor(country.avgRiskScore), fontSize: '0.8rem' }}>
                      {index + 1}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={country.country}
                    secondary={
                      <Box>
                        <Typography variant="body2">
                          {country.count} alerts • {country.avgRiskScore.toFixed(1)}% avg risk
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={country.avgRiskScore}
                          sx={{
                            mt: 1,
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: getRiskColor(country.avgRiskScore)
                            }
                          }}
                        />
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Paper>

          {metrics && (
            <Paper sx={{ p: 2, mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Network Statistics
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Total Nodes:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {metrics.networkStats.totalNodes.toLocaleString()}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Total Connections:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {metrics.networkStats.totalEdges.toLocaleString()}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="warning.main">Suspicious:</Typography>
                  <Typography variant="body2" fontWeight="bold" color="warning.main">
                    {metrics.networkStats.suspiciousConnections}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="info.main">Isolated Nodes:</Typography>
                  <Typography variant="body2" fontWeight="bold" color="info.main">
                    {metrics.networkStats.isolatedNodes}
                  </Typography>
                </Box>
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Floating Action Button for Emergency Actions */}
      <Fab
        color="error"
        aria-label="emergency-block"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
      >
        <Block />
      </Fab>
    </Box>
  );
};

export default FraudDetectionDashboard;