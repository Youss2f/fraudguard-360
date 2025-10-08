/**
 * Excel-Style Fraud Detection Dashboard
 * Real-time fraud detection, ML model monitoring, and immediate response capabilities
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Avatar,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  Badge,
  Alert,
  LinearProgress,
  styled,
  alpha,
  Paper,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  Security,
  Warning,
  Block,
  CheckCircle,
  TrendingUp,
  Search,
  FilterList,
  Refresh,
  Visibility,
  MoreVert,
  Person,
  AttachMoney,
  LocationOn,
  Schedule,
  Speed,
  BugReport,
  Shield,
  Psychology,
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
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';
import { excelColors } from '../theme/excelTheme';

// Styled components
const FraudDetectionCard = styled(Card)({
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
});

const MLModelCard = styled(Paper)({
  padding: '16px',
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.primary.light}`,
  borderRadius: 8,
});

const RealTimeAlert = styled(Alert)({
  marginBottom: '8px',
  '& .MuiAlert-icon': {
    fontSize: '20px',
  },
});

// Mock data for fraud detection
const generateMLModelMetrics = () => [
  { name: 'Transaction Anomaly', accuracy: 94.2, precision: 92.8, recall: 89.3, f1Score: 91.0, status: 'active' },
  { name: 'Identity Verification', accuracy: 97.1, precision: 95.6, recall: 94.2, f1Score: 94.9, status: 'active' },
  { name: 'Payment Pattern', accuracy: 89.7, precision: 88.2, recall: 91.5, f1Score: 89.8, status: 'training' },
  { name: 'Geolocation Risk', accuracy: 92.4, precision: 90.1, recall: 87.9, f1Score: 89.0, status: 'active' },
  { name: 'Device Fingerprint', accuracy: 96.3, precision: 94.7, recall: 93.1, f1Score: 93.9, status: 'active' },
];

const generateRealTimeFraudAlerts = () => [
  {
    id: 'FD-001',
    type: 'High Risk Transaction',
    severity: 'critical',
    confidence: 97.8,
    amount: '$15,847',
    user: 'John.Smith@email.com',
    location: 'Lagos, Nigeria',
    model: 'Transaction Anomaly',
    timestamp: new Date(Date.now() - 2 * 60 * 1000),
    status: 'pending',
    riskFactors: ['Unusual location', 'Large amount', 'New device']
  },
  {
    id: 'FD-002',
    type: 'Identity Theft',
    severity: 'high',
    confidence: 89.3,
    amount: '$2,340',
    user: 'jane.doe@email.com',
    location: 'Unknown VPN',
    model: 'Identity Verification',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    status: 'investigating',
    riskFactors: ['VPN usage', 'Rapid transactions', 'Failed verification']
  },
  {
    id: 'FD-003',
    type: 'Card Testing',
    severity: 'medium',
    confidence: 76.2,
    amount: '$1.00',
    user: 'test.user@email.com',
    location: 'Multiple IPs',
    model: 'Payment Pattern',
    timestamp: new Date(Date.now() - 8 * 60 * 1000),
    status: 'blocked',
    riskFactors: ['Multiple small transactions', 'Different cards', 'Bot pattern']
  },
  {
    id: 'FD-004',
    type: 'Account Takeover',
    severity: 'critical',
    confidence: 94.1,
    amount: '$8,920',
    user: 'victim@email.com',
    location: 'Moscow, Russia',
    model: 'Device Fingerprint',
    timestamp: new Date(Date.now() - 12 * 60 * 1000),
    status: 'blocked',
    riskFactors: ['Login from new device', 'Password change', 'Immediate transfer']
  },
];

const generateFraudMetrics = () => ({
  totalDetected: 1247,
  totalBlocked: 1089,
  totalSaved: 4782340,
  falsePositives: 23,
  accuracy: 94.7,
  responseTime: 0.23
});

const generateModelPerformanceData = () => [
  { time: '00:00', detections: 45, accuracy: 94.2, latency: 0.21 },
  { time: '04:00', detections: 32, accuracy: 95.1, latency: 0.19 },
  { time: '08:00', detections: 78, accuracy: 93.8, latency: 0.25 },
  { time: '12:00', detections: 125, accuracy: 94.9, latency: 0.22 },
  { time: '16:00', detections: 98, accuracy: 95.3, latency: 0.18 },
  { time: '20:00', detections: 67, accuracy: 94.1, latency: 0.24 },
];

const ExcelFraudDetection: React.FC = () => {
  const [mlModels, setMLModels] = useState(generateMLModelMetrics());
  const [realTimeAlerts, setRealTimeAlerts] = useState(generateRealTimeFraudAlerts());
  const [fraudMetrics, setFraudMetrics] = useState(generateFraudMetrics());
  const [modelPerformance, setModelPerformance] = useState(generateModelPerformanceData());
  const [selectedTab, setSelectedTab] = useState(0);
  const [alertFilter, setAlertFilter] = useState('all');
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);

  // Real-time updates
  useEffect(() => {
    if (!isAutoRefresh) return;

    const interval = setInterval(() => {
      // Simulate new fraud alerts
      const newAlert = {
        id: `FD-${String(Math.floor(Math.random() * 999)).padStart(3, '0')}`,
        type: ['High Risk Transaction', 'Identity Theft', 'Card Testing', 'Account Takeover'][Math.floor(Math.random() * 4)],
        severity: ['critical', 'high', 'medium'][Math.floor(Math.random() * 3)] as 'critical' | 'high' | 'medium',
        confidence: Math.floor(Math.random() * 30) + 70,
        amount: `$${(Math.random() * 10000 + 100).toLocaleString()}`,
        user: `user${Math.floor(Math.random() * 1000)}@email.com`,
        location: ['New York, US', 'London, UK', 'Lagos, Nigeria', 'Unknown VPN'][Math.floor(Math.random() * 4)],
        model: mlModels[Math.floor(Math.random() * mlModels.length)].name,
        timestamp: new Date(),
        status: 'pending' as 'pending',
        riskFactors: ['Unusual location', 'Large amount', 'New device'].slice(0, Math.floor(Math.random() * 3) + 1)
      };

      setRealTimeAlerts(prev => [newAlert, ...prev.slice(0, 19)]); // Keep last 20
      
      // Update metrics
      setFraudMetrics(prev => ({
        ...prev,
        totalDetected: prev.totalDetected + 1,
        totalBlocked: prev.totalBlocked + (newAlert.severity === 'critical' ? 1 : 0),
        totalSaved: prev.totalSaved + (newAlert.severity === 'critical' ? Math.floor(Math.random() * 10000) : 0)
      }));
    }, 10000); // New alert every 10 seconds

    return () => clearInterval(interval);
  }, [isAutoRefresh, mlModels]);

  const handleAlertAction = (alertId: string, action: 'block' | 'approve' | 'investigate') => {
    setRealTimeAlerts(prev => prev.map(alert => 
      alert.id === alertId 
        ? { ...alert, status: action === 'approve' ? 'approved' : action === 'block' ? 'blocked' : 'investigating' }
        : alert
    ));
  };

  const handleModelAction = (modelName: string, action: 'retrain' | 'pause' | 'activate') => {
    setMLModels(prev => prev.map(model => 
      model.name === modelName 
        ? { ...model, status: action === 'pause' ? 'inactive' : action === 'retrain' ? 'training' : 'active' }
        : model
    ));
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return excelColors.error.main;
      case 'high': return excelColors.warning.main;
      case 'medium': return excelColors.info.main;
      default: return excelColors.success.main;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'blocked': return excelColors.error.main;
      case 'investigating': return excelColors.warning.main;
      case 'approved': return excelColors.success.main;
      default: return excelColors.info.main;
    }
  };

  const filteredAlerts = realTimeAlerts.filter(alert => 
    alertFilter === 'all' || alert.severity === alertFilter
  );

  return (
    <Box sx={{ p: 3, backgroundColor: excelColors.background.default, minHeight: '100vh' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
          🛡️ AI-Powered Fraud Detection Center
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant={isAutoRefresh ? 'contained' : 'outlined'}
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
            startIcon={<Refresh />}
          >
            {isAutoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
          </Button>
          <Button variant="outlined" startIcon={<FilterList />}>
            Configure Filters
          </Button>
        </Box>
      </Box>

      {/* Real-time Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={2}>
          <FraudDetectionCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <Security sx={{ fontSize: 40, color: excelColors.primary.main, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, color: excelColors.primary.main }}>
                {fraudMetrics.totalDetected.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Threats Detected Today
              </Typography>
            </CardContent>
          </FraudDetectionCard>
        </Grid>
        <Grid item xs={12} md={2}>
          <FraudDetectionCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <Block sx={{ fontSize: 40, color: excelColors.error.main, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, color: excelColors.error.main }}>
                {fraudMetrics.totalBlocked.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Threats Blocked
              </Typography>
            </CardContent>
          </FraudDetectionCard>
        </Grid>
        <Grid item xs={12} md={2}>
          <FraudDetectionCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <AttachMoney sx={{ fontSize: 40, color: excelColors.success.main, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, color: excelColors.success.main }}>
                ${fraudMetrics.totalSaved.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Losses Prevented
              </Typography>
            </CardContent>
          </FraudDetectionCard>
        </Grid>
        <Grid item xs={12} md={2}>
          <FraudDetectionCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <CheckCircle sx={{ fontSize: 40, color: excelColors.success.main, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, color: excelColors.success.main }}>
                {fraudMetrics.accuracy}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Model Accuracy
              </Typography>
            </CardContent>
          </FraudDetectionCard>
        </Grid>
        <Grid item xs={12} md={2}>
          <FraudDetectionCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 40, color: excelColors.info.main, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, color: excelColors.info.main }}>
                {fraudMetrics.responseTime}s
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Response Time
              </Typography>
            </CardContent>
          </FraudDetectionCard>
        </Grid>
        <Grid item xs={12} md={2}>
          <FraudDetectionCard>
            <CardContent sx={{ textAlign: 'center' }}>
              <BugReport sx={{ fontSize: 40, color: excelColors.warning.main, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, color: excelColors.warning.main }}>
                {fraudMetrics.falsePositives}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                False Positives
              </Typography>
            </CardContent>
          </FraudDetectionCard>
        </Grid>
      </Grid>

      <Tabs value={selectedTab} onChange={(e, v) => setSelectedTab(v)} sx={{ mb: 3 }}>
        <Tab label="Real-time Alerts" />
        <Tab label="ML Models" />
        <Tab label="Performance Analytics" />
      </Tabs>

      {/* Real-time Alerts Tab */}
      {selectedTab === 0 && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Live Fraud Alerts</Typography>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Filter</InputLabel>
              <Select value={alertFilter} onChange={(e) => setAlertFilter(e.target.value)} label="Filter">
                <MenuItem value="all">All Alerts</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <FraudDetectionCard>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Alert ID</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Severity</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>User</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Model</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredAlerts.map((alert) => (
                    <TableRow key={alert.id} hover>
                      <TableCell sx={{ fontWeight: 600 }}>{alert.id}</TableCell>
                      <TableCell>{alert.type}</TableCell>
                      <TableCell>
                        <Chip 
                          label={alert.severity.toUpperCase()} 
                          sx={{ 
                            backgroundColor: alpha(getSeverityColor(alert.severity), 0.1),
                            color: getSeverityColor(alert.severity),
                            fontWeight: 600
                          }}
                          size="small"
                        />
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>{alert.confidence}%</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: excelColors.error.main }}>
                        {alert.amount}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Avatar sx={{ width: 24, height: 24, mr: 1, fontSize: '0.75rem' }}>
                            {alert.user.charAt(0).toUpperCase()}
                          </Avatar>
                          {alert.user}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <LocationOn sx={{ fontSize: 16, mr: 0.5, color: excelColors.text.secondary }} />
                          {alert.location}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip label={alert.model} variant="outlined" size="small" />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Schedule sx={{ fontSize: 16, mr: 0.5, color: excelColors.text.secondary }} />
                          {alert.timestamp.toLocaleTimeString()}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={alert.status.toUpperCase()} 
                          sx={{ 
                            backgroundColor: alpha(getStatusColor(alert.status), 0.1),
                            color: getStatusColor(alert.status),
                            fontWeight: 600
                          }}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <IconButton 
                            size="small" 
                            onClick={() => handleAlertAction(alert.id, 'block')}
                            sx={{ color: excelColors.error.main }}
                          >
                            <Block />
                          </IconButton>
                          <IconButton 
                            size="small" 
                            onClick={() => handleAlertAction(alert.id, 'investigate')}
                            sx={{ color: excelColors.warning.main }}
                          >
                            <Search />
                          </IconButton>
                          <IconButton 
                            size="small" 
                            onClick={() => handleAlertAction(alert.id, 'approve')}
                            sx={{ color: excelColors.success.main }}
                          >
                            <CheckCircle />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </FraudDetectionCard>
        </Box>
      )}

      {/* ML Models Tab */}
      {selectedTab === 1 && (
        <Box>
          <Typography variant="h6" sx={{ mb: 2 }}>AI/ML Model Performance</Typography>
          <Grid container spacing={3}>
            {mlModels.map((model) => (
              <Grid item xs={12} md={6} key={model.name}>
                <MLModelCard>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Psychology sx={{ fontSize: 24, mr: 1, color: excelColors.primary.main }} />
                      <Typography variant="h6">{model.name}</Typography>
                    </Box>
                    <Chip 
                      label={model.status.toUpperCase()} 
                      color={model.status === 'active' ? 'success' : model.status === 'training' ? 'warning' : 'default'}
                      size="small"
                    />
                  </Box>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">Accuracy</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>{model.accuracy}%</Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={model.accuracy} 
                        sx={{ mt: 1, height: 6, borderRadius: 3 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">Precision</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>{model.precision}%</Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={model.precision} 
                        sx={{ mt: 1, height: 6, borderRadius: 3 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">Recall</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>{model.recall}%</Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={model.recall} 
                        sx={{ mt: 1, height: 6, borderRadius: 3 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">F1 Score</Typography>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>{model.f1Score}%</Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={model.f1Score} 
                        sx={{ mt: 1, height: 6, borderRadius: 3 }}
                      />
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button 
                      size="small" 
                      variant="outlined" 
                      onClick={() => handleModelAction(model.name, 'retrain')}
                    >
                      Retrain
                    </Button>
                    <Button 
                      size="small" 
                      variant="outlined" 
                      onClick={() => handleModelAction(model.name, model.status === 'active' ? 'pause' : 'activate')}
                    >
                      {model.status === 'active' ? 'Pause' : 'Activate'}
                    </Button>
                  </Box>
                </MLModelCard>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Performance Analytics Tab */}
      {selectedTab === 2 && (
        <Box>
          <Typography variant="h6" sx={{ mb: 2 }}>Model Performance Analytics</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <FraudDetectionCard>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Detection Performance Over Time</Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={modelPerformance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="detections" fill={excelColors.primary.main} name="Detections" />
                      <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke={excelColors.success.main} name="Accuracy %" />
                      <Line yAxisId="right" type="monotone" dataKey="latency" stroke={excelColors.warning.main} name="Latency (s)" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </FraudDetectionCard>
            </Grid>
            <Grid item xs={12} md={4}>
              <FraudDetectionCard>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Alert Distribution</Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Critical', value: realTimeAlerts.filter(a => a.severity === 'critical').length, color: excelColors.error.main },
                          { name: 'High', value: realTimeAlerts.filter(a => a.severity === 'high').length, color: excelColors.warning.main },
                          { name: 'Medium', value: realTimeAlerts.filter(a => a.severity === 'medium').length, color: excelColors.info.main },
                        ]}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label
                      >
                        {[
                          { name: 'Critical', value: realTimeAlerts.filter(a => a.severity === 'critical').length, color: excelColors.error.main },
                          { name: 'High', value: realTimeAlerts.filter(a => a.severity === 'high').length, color: excelColors.warning.main },
                          { name: 'Medium', value: realTimeAlerts.filter(a => a.severity === 'medium').length, color: excelColors.info.main },
                        ].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </FraudDetectionCard>
            </Grid>
          </Grid>
        </Box>
      )}
    </Box>
  );
};

export default ExcelFraudDetection;