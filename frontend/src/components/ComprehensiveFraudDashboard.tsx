import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Alert,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  Tooltip,
  Badge,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tabs,
  Tab,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Warning as WarningIcon,
  Security as SecurityIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  GetApp as ExportIcon,
  Visibility as ViewIcon,
  Block as BlockIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  LocationOn as LocationIcon,
  Phone as PhoneIcon,
  AccessTime as TimeIcon,
  AttachMoney as MoneyIcon,
  Person as PersonIcon,
  Speed as SpeedIcon,
  Notifications as NotificationsIcon,
  Dashboard as DashboardIcon,
  Assessment as AssessmentIcon
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
  ScatterChart,
  Scatter
} from 'recharts';
import { format, subDays, subHours } from 'date-fns';

// Types
interface FraudAlert {
  id: string;
  userId: string;
  fraudType: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  riskScore: number;
  confidence: number;
  timestamp: string;
  description: string;
  evidence: any;
  status: 'OPEN' | 'INVESTIGATING' | 'RESOLVED' | 'FALSE_POSITIVE';
  location?: string;
  callId?: string;
  cost?: number;
}

interface FraudStatistics {
  totalAlerts: number;
  criticalAlerts: number;
  highAlerts: number;
  mediumAlerts: number;
  lowAlerts: number;
  avgRiskScore: number;
  maxRiskScore: number;
  uniqueUsersAffected: number;
  totalPotentialLoss: number;
  windowStart: string;
  windowEnd: string;
}

interface UserRiskProfile {
  userId: string;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  fraudProbability: number;
  recentAlerts: number;
  fraudTypes: string[];
  lastAssessment: string;
  predictionHistory: any[];
}

// Color schemes
const SEVERITY_COLORS = {
  CRITICAL: '#f44336',
  HIGH: '#ff9800',
  MEDIUM: '#ffeb3b',
  LOW: '#4caf50'
};

const FRAUD_TYPE_COLORS = {
  VELOCITY_FRAUD: '#e91e63',
  PREMIUM_RATE_FRAUD: '#9c27b0',
  SIM_BOX_FRAUD: '#673ab7',
  ROAMING_FRAUD: '#3f51b5',
  ACCOUNT_TAKEOVER: '#2196f3',
  LOCATION_ANOMALY: '#00bcd4'
};

const ComprehensiveFraudDashboard: React.FC = () => {
  // State management
  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [statistics, setStatistics] = useState<FraudStatistics | null>(null);
  const [userProfiles, setUserProfiles] = useState<Map<string, UserRiskProfile>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<FraudAlert | null>(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [timeRange, setTimeRange] = useState('24h');
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');
  const [filterFraudType, setFilterFraudType] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  // WebSocket connection for real-time updates
  const [ws, setWs] = useState<WebSocket | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const websocket = new WebSocket('ws://localhost:8001/ws/fraud-alerts');
      
      websocket.onopen = () => {
        console.log('WebSocket connected');
        setWs(websocket);
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'fraud_alert') {
            setAlerts(prev => [data.alert, ...prev.slice(0, 99)]); // Keep last 100 alerts
          } else if (data.type === 'statistics_update') {
            setStatistics(data.statistics);
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      };

      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setWs(null);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Data fetching functions
  const fetchAlerts = useCallback(async () => {
    try {
      const response = await fetch(`/api/alerts?timeRange=${timeRange}&limit=100`);
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
      setError('Failed to fetch alerts');
    }
  }, [timeRange]);

  const fetchStatistics = useCallback(async () => {
    try {
      const response = await fetch(`/api/statistics?timeRange=${timeRange}`);
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Error fetching statistics:', error);
      setError('Failed to fetch statistics');
    }
  }, [timeRange]);

  const fetchUserProfile = useCallback(async (userId: string) => {
    try {
      const response = await fetch(`/api/user-risk-profile/${userId}`);
      const profile = await response.json();
      setUserProfiles(prev => new Map(prev.set(userId, profile)));
    } catch (error) {
      console.error('Error fetching user profile:', error);
    }
  }, []);

  // Initial data load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        await Promise.all([fetchAlerts(), fetchStatistics()]);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [fetchAlerts, fetchStatistics]);

  // Auto-refresh mechanism
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(async () => {
      await Promise.all([fetchAlerts(), fetchStatistics()]);
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchAlerts, fetchStatistics]);

  // Filtered alerts
  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      const matchesSeverity = filterSeverity === 'ALL' || alert.severity === filterSeverity;
      const matchesFraudType = filterFraudType === 'ALL' || alert.fraudType === filterFraudType;
      const matchesSearch = !searchTerm || 
        alert.userId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        alert.description.toLowerCase().includes(searchTerm.toLowerCase());
      
      return matchesSeverity && matchesFraudType && matchesSearch;
    });
  }, [alerts, filterSeverity, filterFraudType, searchTerm]);

  // Statistics calculations
  const alertsByHour = useMemo(() => {
    const hours = Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      alerts: 0,
      critical: 0,
      high: 0,
      medium: 0,
      low: 0
    }));

    filteredAlerts.forEach(alert => {
      const hour = new Date(alert.timestamp).getHours();
      hours[hour].alerts++;
      const severity = alert.severity.toLowerCase();
      if (severity === 'critical') hours[hour].critical++;
      else if (severity === 'high') hours[hour].high++;
      else if (severity === 'medium') hours[hour].medium++;
      else if (severity === 'low') hours[hour].low++;
    });

    return hours;
  }, [filteredAlerts]);

  const fraudTypeDistribution = useMemo(() => {
    const distribution: Record<string, number> = {};
    filteredAlerts.forEach(alert => {
      distribution[alert.fraudType] = (distribution[alert.fraudType] || 0) + 1;
    });

    return Object.entries(distribution).map(([type, count]) => ({
      name: type.replace(/_/g, ' '),
      value: count,
      color: (FRAUD_TYPE_COLORS as any)[type] || '#gray'
    }));
  }, [filteredAlerts]);

  const riskScoreDistribution = useMemo(() => {
    const buckets = [0, 0, 0, 0, 0]; // 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    
    filteredAlerts.forEach(alert => {
      const bucket = Math.min(Math.floor(alert.riskScore * 5), 4);
      buckets[bucket]++;
    });

    return buckets.map((count, index) => ({
      range: `${(index * 0.2).toFixed(1)}-${((index + 1) * 0.2).toFixed(1)}`,
      count
    }));
  }, [filteredAlerts]);

  // Event handlers
  const handleAlertClick = async (alert: FraudAlert) => {
    setSelectedAlert(alert);
    if (!userProfiles.has(alert.userId)) {
      await fetchUserProfile(alert.userId);
    }
  };

  const handleAlertStatusUpdate = async (alertId: string, newStatus: string) => {
    try {
      await fetch(`/api/alerts/${alertId}/status`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });
      
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId ? { ...alert, status: newStatus as any } : alert
      ));
      
      if (selectedAlert?.id === alertId) {
        setSelectedAlert(prev => prev ? { ...prev, status: newStatus as any } : null);
      }
    } catch (error) {
      console.error('Error updating alert status:', error);
    }
  };

  const handleExportData = () => {
    const csvContent = [
      ['ID', 'User ID', 'Fraud Type', 'Severity', 'Risk Score', 'Timestamp', 'Status'],
      ...filteredAlerts.map(alert => [
        alert.id,
        alert.userId,
        alert.fraudType,
        alert.severity,
        alert.riskScore.toFixed(3),
        alert.timestamp,
        alert.status
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraud_alerts_${format(new Date(), 'yyyy-MM-dd_HH-mm')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Render functions
  const renderStatisticsCards = () => (
    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Total Alerts
                </Typography>
                <Typography variant="h4">
                  {statistics?.totalAlerts || 0}
                </Typography>
              </Box>
              <SecurityIcon color="primary" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Critical Alerts
                </Typography>
                <Typography variant="h4" color="error">
                  {statistics?.criticalAlerts || 0}
                </Typography>
              </Box>
              <WarningIcon color="error" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Avg Risk Score
                </Typography>
                <Typography variant="h4">
                  {(statistics?.avgRiskScore || 0).toFixed(2)}
                </Typography>
              </Box>
              <SpeedIcon color="warning" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" gutterBottom>
                  Potential Loss
                </Typography>
                <Typography variant="h4" color="error">
                  ${(statistics?.totalPotentialLoss || 0).toLocaleString()}
                </Typography>
              </Box>
              <MoneyIcon color="error" sx={{ fontSize: 40 }} />
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderAlertsTable = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>User ID</TableCell>
            <TableCell>Fraud Type</TableCell>
            <TableCell>Severity</TableCell>
            <TableCell>Risk Score</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {filteredAlerts.slice(0, 20).map((alert) => (
            <TableRow key={alert.id} hover>
              <TableCell>
                {format(new Date(alert.timestamp), 'MM/dd HH:mm:ss')}
              </TableCell>
              <TableCell>
                <Typography variant="body2" color="primary">
                  {alert.userId}
                </Typography>
              </TableCell>
              <TableCell>
                <Chip
                  label={alert.fraudType.replace(/_/g, ' ')}
                  size="small"
                  style={{ backgroundColor: (FRAUD_TYPE_COLORS as any)[alert.fraudType] || '#gray', color: 'white' }}
                />
              </TableCell>
              <TableCell>
                <Chip
                  label={alert.severity}
                  size="small"
                  style={{ backgroundColor: SEVERITY_COLORS[alert.severity], color: 'white' }}
                />
              </TableCell>
              <TableCell>
                <Box display="flex" alignItems="center">
                  <Box width="60px" mr={1}>
                    <LinearProgress
                      variant="determinate"
                      value={alert.riskScore * 100}
                      color={alert.riskScore > 0.7 ? 'error' : alert.riskScore > 0.4 ? 'warning' : 'success'}
                    />
                  </Box>
                  <Typography variant="body2">
                    {(alert.riskScore * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </TableCell>
              <TableCell>
                <Chip
                  label={alert.status}
                  size="small"
                  variant="outlined"
                  color={alert.status === 'RESOLVED' ? 'success' : 
                         alert.status === 'FALSE_POSITIVE' ? 'default' : 'primary'}
                />
              </TableCell>
              <TableCell>
                <IconButton
                  size="small"
                  onClick={() => handleAlertClick(alert)}
                  title="View Details"
                >
                  <ViewIcon />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  const renderCharts = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Alerts by Hour
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={alertsByHour}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="critical"
                  stackId="1"
                  stroke={SEVERITY_COLORS.CRITICAL}
                  fill={SEVERITY_COLORS.CRITICAL}
                />
                <Area
                  type="monotone"
                  dataKey="high"
                  stackId="1"
                  stroke={SEVERITY_COLORS.HIGH}
                  fill={SEVERITY_COLORS.HIGH}
                />
                <Area
                  type="monotone"
                  dataKey="medium"
                  stackId="1"
                  stroke={SEVERITY_COLORS.MEDIUM}
                  fill={SEVERITY_COLORS.MEDIUM}
                />
                <Area
                  type="monotone"
                  dataKey="low"
                  stackId="1"
                  stroke={SEVERITY_COLORS.LOW}
                  fill={SEVERITY_COLORS.LOW}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Fraud Type Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={fraudTypeDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {fraudTypeDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Risk Score Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={riskScoreDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderAlertDetails = () => {
    if (!selectedAlert) return null;

    const userProfile = userProfiles.get(selectedAlert.userId);

    return (
      <Dialog
        open={!!selectedAlert}
        onClose={() => setSelectedAlert(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">
              Alert Details - {selectedAlert.id}
            </Typography>
            <Chip
              label={selectedAlert.severity}
              style={{ backgroundColor: SEVERITY_COLORS[selectedAlert.severity], color: 'white' }}
            />
          </Box>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Basic Information
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon><PersonIcon /></ListItemIcon>
                  <ListItemText
                    primary="User ID"
                    secondary={selectedAlert.userId}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><SecurityIcon /></ListItemIcon>
                  <ListItemText
                    primary="Fraud Type"
                    secondary={selectedAlert.fraudType.replace(/_/g, ' ')}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon><TimeIcon /></ListItemIcon>
                  <ListItemText
                    primary="Timestamp"
                    secondary={format(new Date(selectedAlert.timestamp), 'PPpp')}
                  />
                </ListItem>
                {selectedAlert.location && (
                  <ListItem>
                    <ListItemIcon><LocationIcon /></ListItemIcon>
                    <ListItemText
                      primary="Location"
                      secondary={selectedAlert.location}
                    />
                  </ListItem>
                )}
                {selectedAlert.cost && (
                  <ListItem>
                    <ListItemIcon><MoneyIcon /></ListItemIcon>
                    <ListItemText
                      primary="Cost"
                      secondary={`$${selectedAlert.cost.toFixed(2)}`}
                    />
                  </ListItem>
                )}
              </List>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Risk Assessment
              </Typography>
              <Box mb={2}>
                <Typography variant="body2" gutterBottom>
                  Risk Score: {(selectedAlert.riskScore * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={selectedAlert.riskScore * 100}
                  color={selectedAlert.riskScore > 0.7 ? 'error' : selectedAlert.riskScore > 0.4 ? 'warning' : 'success'}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
              <Box mb={2}>
                <Typography variant="body2" gutterBottom>
                  Confidence: {(selectedAlert.confidence * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={selectedAlert.confidence * 100}
                  color="primary"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
              
              {userProfile && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    User Risk Profile
                  </Typography>
                  <Chip
                    label={`Risk Level: ${userProfile.riskLevel}`}
                    color={userProfile.riskLevel === 'HIGH' ? 'error' : 
                           userProfile.riskLevel === 'MEDIUM' ? 'warning' : 'success'}
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2">
                    Recent Alerts: {userProfile.recentAlerts}
                  </Typography>
                  <Typography variant="body2">
                    Fraud Probability: {(userProfile.fraudProbability * 100).toFixed(1)}%
                  </Typography>
                </Box>
              )}
            </Grid>

            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Description
              </Typography>
              <Typography variant="body2" paragraph>
                {selectedAlert.description}
              </Typography>

              <Typography variant="subtitle2" gutterBottom>
                Evidence
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                <pre style={{ fontSize: '12px', margin: 0, whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(selectedAlert.evidence, null, 2)}
                </pre>
              </Paper>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedAlert(null)}>
            Close
          </Button>
          <Button
            color="success"
            onClick={() => handleAlertStatusUpdate(selectedAlert.id, 'RESOLVED')}
            disabled={selectedAlert.status === 'RESOLVED'}
          >
            Mark Resolved
          </Button>
          <Button
            color="warning"
            onClick={() => handleAlertStatusUpdate(selectedAlert.id, 'FALSE_POSITIVE')}
            disabled={selectedAlert.status === 'FALSE_POSITIVE'}
          >
            False Positive
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Box textAlign="center">
          <LinearProgress sx={{ mb: 2, width: '200px' }} />
          <Typography>Loading fraud detection data...</Typography>
        </Box>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Fraud Detection Dashboard
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => Promise.all([fetchAlerts(), fetchStatistics()])}
          >
            Refresh
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<ExportIcon />}
            onClick={handleExportData}
          >
            Export
          </Button>
        </Box>
      </Box>

      {/* Controls */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth size="small">
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="1h">Last Hour</MenuItem>
              <MenuItem value="24h">Last 24 Hours</MenuItem>
              <MenuItem value="7d">Last 7 Days</MenuItem>
              <MenuItem value="30d">Last 30 Days</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth size="small">
            <InputLabel>Severity</InputLabel>
            <Select
              value={filterSeverity}
              label="Severity"
              onChange={(e) => setFilterSeverity(e.target.value)}
            >
              <MenuItem value="ALL">All Severities</MenuItem>
              <MenuItem value="CRITICAL">Critical</MenuItem>
              <MenuItem value="HIGH">High</MenuItem>
              <MenuItem value="MEDIUM">Medium</MenuItem>
              <MenuItem value="LOW">Low</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth size="small">
            <InputLabel>Fraud Type</InputLabel>
            <Select
              value={filterFraudType}
              label="Fraud Type"
              onChange={(e) => setFilterFraudType(e.target.value)}
            >
              <MenuItem value="ALL">All Types</MenuItem>
              <MenuItem value="VELOCITY_FRAUD">Velocity Fraud</MenuItem>
              <MenuItem value="PREMIUM_RATE_FRAUD">Premium Rate</MenuItem>
              <MenuItem value="SIM_BOX_FRAUD">SIM Box</MenuItem>
              <MenuItem value="ROAMING_FRAUD">Roaming Fraud</MenuItem>
              <MenuItem value="ACCOUNT_TAKEOVER">Account Takeover</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <TextField
            fullWidth
            size="small"
            label="Search"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="User ID or description..."
          />
        </Grid>
      </Grid>

      {/* Statistics Cards */}
      {renderStatisticsCards()}

      {/* Main Content Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
          <Tab label="Alerts" icon={<NotificationsIcon />} />
          <Tab label="Analytics" icon={<AssessmentIcon />} />
          <Tab label="Dashboard" icon={<DashboardIcon />} />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Fraud Alerts ({filteredAlerts.length})
            </Typography>
            {renderAlertsTable()}
          </CardContent>
        </Card>
      )}

      {selectedTab === 1 && renderCharts()}

      {selectedTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            {renderCharts()}
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Status
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <Badge color={ws ? 'success' : 'error'} variant="dot">
                        <NotificationsIcon />
                      </Badge>
                    </ListItemIcon>
                    <ListItemText
                      primary="Real-time Alerts"
                      secondary={ws ? 'Connected' : 'Disconnected'}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Badge color="success" variant="dot">
                        <SecurityIcon />
                      </Badge>
                    </ListItemIcon>
                    <ListItemText
                      primary="ML Models"
                      secondary="Active"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Badge color="success" variant="dot">
                        <AssessmentIcon />
                      </Badge>
                    </ListItemIcon>
                    <ListItemText
                      primary="Analytics Engine"
                      secondary="Running"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Alert Details Dialog */}
      {renderAlertDetails()}
    </Box>
  );
};

export default ComprehensiveFraudDashboard;