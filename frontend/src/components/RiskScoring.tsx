import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  Tooltip,
  Tabs,
  Tab,
  Divider
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Speed,
  Settings,
  Tune,
  Assessment,
  Timeline,
  Person,
  Phone,
  LocationOn,
  AttachMoney,
  Security,
  Visibility,
  Edit,
  Refresh,
  Download,
  Add,
  Save
} from '@mui/icons-material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

interface RiskModel {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'training';
  accuracy: number;
  lastUpdated: Date;
  version: string;
  threshold: number;
  weights: {
    behavioral: number;
    geographical: number;
    transactional: number;
    network: number;
    temporal: number;
  };
}

interface RiskScore {
  userId: string;
  userPhone: string;
  currentScore: number;
  previousScore: number;
  trend: 'up' | 'down' | 'stable';
  category: 'low' | 'medium' | 'high' | 'critical';
  lastCalculated: Date;
  factors: {
    behavioral: number;
    geographical: number;
    transactional: number;
    network: number;
    temporal: number;
  };
  alerts: number;
}

const RiskScoring: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [modelConfigOpen, setModelConfigOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<RiskModel | null>(null);
  const [thresholdConfigOpen, setThresholdConfigOpen] = useState(false);
  const [autoRecalculate, setAutoRecalculate] = useState(true);

  // Mock data
  const [riskModels, setRiskModels] = useState<RiskModel[]>([
    {
      id: 'model-001',
      name: 'Comprehensive Fraud Model v2.1',
      description: 'Advanced ML model combining behavioral, transactional, and network analysis',
      status: 'active',
      accuracy: 94.2,
      lastUpdated: new Date(Date.now() - 86400000),
      version: '2.1.3',
      threshold: 75,
      weights: {
        behavioral: 0.25,
        geographical: 0.15,
        transactional: 0.30,
        network: 0.20,
        temporal: 0.10
      }
    },
    {
      id: 'model-002',
      name: 'Real-time Transaction Model',
      description: 'Specialized model for real-time transaction risk assessment',
      status: 'active',
      accuracy: 91.8,
      lastUpdated: new Date(Date.now() - 172800000),
      version: '1.8.1',
      threshold: 80,
      weights: {
        behavioral: 0.20,
        geographical: 0.25,
        transactional: 0.35,
        network: 0.15,
        temporal: 0.05
      }
    },
    {
      id: 'model-003',
      name: 'Behavioral Analysis Model',
      description: 'Focus on user behavior patterns and anomaly detection',
      status: 'training',
      accuracy: 89.5,
      lastUpdated: new Date(Date.now() - 259200000),
      version: '3.0.0-beta',
      threshold: 70,
      weights: {
        behavioral: 0.45,
        geographical: 0.10,
        transactional: 0.20,
        network: 0.15,
        temporal: 0.10
      }
    }
  ]);

  const [riskScores, setRiskScores] = useState<RiskScore[]>([
    {
      userId: 'user_12345',
      userPhone: '+1-555-0123',
      currentScore: 92,
      previousScore: 78,
      trend: 'up',
      category: 'critical',
      lastCalculated: new Date(Date.now() - 300000),
      factors: {
        behavioral: 85,
        geographical: 95,
        transactional: 90,
        network: 88,
        temporal: 82
      },
      alerts: 3
    },
    {
      userId: 'user_67890',
      userPhone: '+1-555-0456',
      currentScore: 76,
      previousScore: 79,
      trend: 'down',
      category: 'high',
      lastCalculated: new Date(Date.now() - 600000),
      factors: {
        behavioral: 72,
        geographical: 68,
        transactional: 85,
        network: 75,
        temporal: 80
      },
      alerts: 1
    },
    {
      userId: 'user_24680',
      userPhone: '+1-555-0789',
      currentScore: 45,
      previousScore: 42,
      trend: 'up',
      category: 'medium',
      lastCalculated: new Date(Date.now() - 900000),
      factors: {
        behavioral: 40,
        geographical: 52,
        transactional: 48,
        network: 45,
        temporal: 38
      },
      alerts: 0
    }
  ]);

  const getRiskColor = (score: number) => {
    if (score >= 80) return '#d32f2f';
    if (score >= 60) return '#f57c00';
    if (score >= 40) return '#fbc02d';
    return '#388e3c';
  };

  const getRiskCategory = (score: number) => {
    if (score >= 80) return 'critical';
    if (score >= 60) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  };

  const getRiskCategoryColor = (category: string) => {
    switch (category) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp sx={{ color: '#d32f2f' }} />;
      case 'down': return <TrendingDown sx={{ color: '#4caf50' }} />;
      default: return <span>—</span>;
    }
  };

  const getModelStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'training': return 'warning';
      case 'inactive': return 'error';
      default: return 'default';
    }
  };

  // Statistics
  const riskStats = {
    total: riskScores.length,
    critical: riskScores.filter(s => s.category === 'critical').length,
    high: riskScores.filter(s => s.category === 'high').length,
    medium: riskScores.filter(s => s.category === 'medium').length,
    low: riskScores.filter(s => s.category === 'low').length,
    averageScore: Math.round(riskScores.reduce((sum, s) => sum + s.currentScore, 0) / riskScores.length),
    trending: riskScores.filter(s => s.trend === 'up').length
  };

  const distributionData = [
    { name: 'Critical', value: riskStats.critical, color: '#d32f2f' },
    { name: 'High', value: riskStats.high, color: '#f57c00' },
    { name: 'Medium', value: riskStats.medium, color: '#fbc02d' },
    { name: 'Low', value: riskStats.low, color: '#388e3c' }
  ];

  const trendData = [
    { time: '00:00', avgScore: 45 },
    { time: '04:00', avgScore: 42 },
    { time: '08:00', avgScore: 52 },
    { time: '12:00', avgScore: 68 },
    { time: '16:00', avgScore: 71 },
    { time: '20:00', avgScore: 63 }
  ];

  const handleRecalculateAll = () => {
    // Simulate recalculation
    setRiskScores(prev => prev.map(score => ({
      ...score,
      lastCalculated: new Date(),
      // Slight random variation
      currentScore: Math.max(0, Math.min(100, score.currentScore + (Math.random() - 0.5) * 10))
    })));
  };

  const handleModelConfig = (model: RiskModel) => {
    setSelectedModel(model);
    setModelConfigOpen(true);
  };

  return (
    <Box sx={{ p: 3, minHeight: '100vh', bgcolor: '#f5f5f5' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          Risk Scoring
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRecalculate}
                onChange={(e) => setAutoRecalculate(e.target.checked)}
                color="primary"
              />
            }
            label="Auto Recalculate"
          />
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRecalculateAll}
          >
            Recalculate All
          </Button>
          <Button variant="contained" startIcon={<Settings />}>
            Model Settings
          </Button>
          <Button variant="contained" startIcon={<Download />} color="secondary">
            Export Scores
          </Button>
        </Box>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {riskStats.total}
              </Typography>
              <Typography variant="body2">Total Users</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #d32f2f 0%, #f44336 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {riskStats.critical}
              </Typography>
              <Typography variant="body2">Critical Risk</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #f57c00 0%, #ff9800 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {riskStats.high}
              </Typography>
              <Typography variant="body2">High Risk</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #fbc02d 0%, #ffeb3b 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {riskStats.medium}
              </Typography>
              <Typography variant="body2">Medium Risk</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #388e3c 0%, #4caf50 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {riskStats.averageScore}
              </Typography>
              <Typography variant="body2">Avg. Score</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #7b1fa2 0%, #9c27b0 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {riskStats.trending}
              </Typography>
              <Typography variant="body2">Trending Up</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 350 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Risk Distribution
            </Typography>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart>
                <Pie
                  data={distributionData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {distributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: 350 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Average Risk Score Trend (24 Hours)
            </Typography>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <RechartsTooltip />
                <Line 
                  type="monotone" 
                  dataKey="avgScore" 
                  stroke="#1976d2" 
                  strokeWidth={3}
                  dot={{ fill: '#1976d2', strokeWidth: 2, r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Risk Scores" />
        <Tab label="Models" />
        <Tab label="Thresholds" />
      </Tabs>

      {selectedTab === 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
            Individual Risk Scores
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>User</TableCell>
                  <TableCell>Current Score</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Trend</TableCell>
                  <TableCell>Last Updated</TableCell>
                  <TableCell>Alerts</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {riskScores
                  .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((score) => (
                  <TableRow key={score.userId} hover>
                    <TableCell>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                          {score.userId}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {score.userPhone}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Typography 
                          variant="h6" 
                          sx={{ 
                            fontWeight: 'bold',
                            color: getRiskColor(score.currentScore)
                          }}
                        >
                          {score.currentScore}
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={score.currentScore}
                          sx={{ 
                            width: 100, 
                            height: 8,
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: getRiskColor(score.currentScore)
                            }
                          }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={score.category.toUpperCase()}
                        size="small"
                        color={getRiskCategoryColor(score.category) as any}
                        sx={{ fontWeight: 'bold' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getTrendIcon(score.trend)}
                        <Typography variant="body2">
                          {score.previousScore} → {score.currentScore}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {score.lastCalculated.toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {score.alerts > 0 ? (
                        <Chip
                          label={score.alerts}
                          size="small"
                          color="error"
                          icon={<Warning />}
                        />
                      ) : (
                        <CheckCircle sx={{ color: '#4caf50' }} />
                      )}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="View Details">
                          <IconButton size="small">
                            <Visibility />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Recalculate">
                          <IconButton size="small">
                            <Refresh />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={riskScores.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={(_, newPage) => setPage(newPage)}
            onRowsPerPageChange={(e) => {
              setRowsPerPage(parseInt(e.target.value, 10));
              setPage(0);
            }}
          />
        </Paper>
      )}

      {selectedTab === 1 && (
        <Paper sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              Risk Assessment Models
            </Typography>
            <Button variant="contained" startIcon={<Add />}>
              Deploy New Model
            </Button>
          </Box>
          
          <Grid container spacing={3}>
            {riskModels.map((model) => (
              <Grid item xs={12} md={6} key={model.id}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                          {model.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          {model.description}
                        </Typography>
                      </Box>
                      <Chip
                        label={model.status.toUpperCase()}
                        size="small"
                        color={getModelStatusColor(model.status) as any}
                        sx={{ fontWeight: 'bold' }}
                      />
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Accuracy
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                          {model.accuracy}%
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Version
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                          {model.version}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Threshold
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                          {model.threshold}
                        </Typography>
                      </Box>
                    </Box>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Last Updated: {model.lastUpdated.toLocaleDateString()}
                    </Typography>

                    <Divider sx={{ my: 2 }} />

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Tune />}
                        onClick={() => handleModelConfig(model)}
                      >
                        Configure
                      </Button>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button size="small" variant="outlined">
                          Test
                        </Button>
                        <Button size="small" variant="contained">
                          Deploy
                        </Button>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {selectedTab === 2 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 'bold' }}>
            Risk Thresholds Configuration
          </Typography>
          
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Risk Categories
              </Typography>
              
              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Critical Risk (80-100)
                </Typography>
                <Slider
                  defaultValue={80}
                  min={0}
                  max={100}
                  marks={[
                    { value: 0, label: '0' },
                    { value: 50, label: '50' },
                    { value: 100, label: '100' }
                  ]}
                  sx={{ color: '#d32f2f', mb: 3 }}
                />
              </Box>

              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  High Risk (60-79)
                </Typography>
                <Slider
                  defaultValue={60}
                  min={0}
                  max={100}
                  sx={{ color: '#f57c00', mb: 3 }}
                />
              </Box>

              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Medium Risk (40-59)
                </Typography>
                <Slider
                  defaultValue={40}
                  min={0}
                  max={100}
                  sx={{ color: '#fbc02d', mb: 3 }}
                />
              </Box>

              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Low Risk (0-39)
                </Typography>
                <Slider
                  defaultValue={0}
                  min={0}
                  max={100}
                  sx={{ color: '#388e3c', mb: 3 }}
                />
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Alert Triggers
              </Typography>
              
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Auto-generate alerts for Critical risk"
                sx={{ mb: 2, display: 'block' }}
              />
              
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Auto-generate alerts for High risk"
                sx={{ mb: 2, display: 'block' }}
              />
              
              <FormControlLabel
                control={<Switch />}
                label="Auto-generate alerts for Medium risk"
                sx={{ mb: 2, display: 'block' }}
              />

              <Divider sx={{ my: 3 }} />

              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Recalculation Settings
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Recalculation Frequency</InputLabel>
                <Select defaultValue="hourly" label="Recalculation Frequency">
                  <MenuItem value="realtime">Real-time</MenuItem>
                  <MenuItem value="5min">Every 5 minutes</MenuItem>
                  <MenuItem value="15min">Every 15 minutes</MenuItem>
                  <MenuItem value="hourly">Hourly</MenuItem>
                  <MenuItem value="daily">Daily</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Batch Size"
                defaultValue="1000"
                type="number"
                sx={{ mb: 2 }}
              />
            </Grid>
          </Grid>

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 4 }}>
            <Button variant="outlined">
              Reset to Defaults
            </Button>
            <Button variant="contained" startIcon={<Save />}>
              Save Configuration
            </Button>
          </Box>
        </Paper>
      )}

      {/* Model Configuration Dialog */}
      <Dialog
        open={modelConfigOpen}
        onClose={() => setModelConfigOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Configure Model - {selectedModel?.name}
        </DialogTitle>
        <DialogContent>
          {selectedModel && (
            <Box sx={{ pt: 2 }}>
              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Feature Weights
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Behavioral Analysis ({(selectedModel.weights.behavioral * 100).toFixed(0)}%)
                  </Typography>
                  <Slider
                    defaultValue={selectedModel.weights.behavioral * 100}
                    min={0}
                    max={100}
                    sx={{ mb: 3 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Geographical Analysis ({(selectedModel.weights.geographical * 100).toFixed(0)}%)
                  </Typography>
                  <Slider
                    defaultValue={selectedModel.weights.geographical * 100}
                    min={0}
                    max={100}
                    sx={{ mb: 3 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Transactional Analysis ({(selectedModel.weights.transactional * 100).toFixed(0)}%)
                  </Typography>
                  <Slider
                    defaultValue={selectedModel.weights.transactional * 100}
                    min={0}
                    max={100}
                    sx={{ mb: 3 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Network Analysis ({(selectedModel.weights.network * 100).toFixed(0)}%)
                  </Typography>
                  <Slider
                    defaultValue={selectedModel.weights.network * 100}
                    min={0}
                    max={100}
                    sx={{ mb: 3 }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Temporal Analysis ({(selectedModel.weights.temporal * 100).toFixed(0)}%)
                  </Typography>
                  <Slider
                    defaultValue={selectedModel.weights.temporal * 100}
                    min={0}
                    max={100}
                    sx={{ mb: 3 }}
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                Model Parameters
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Risk Threshold"
                    defaultValue={selectedModel.threshold}
                    type="number"
                    inputProps={{ min: 0, max: 100 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Model Status</InputLabel>
                    <Select defaultValue={selectedModel.status} label="Model Status">
                      <MenuItem value="active">Active</MenuItem>
                      <MenuItem value="inactive">Inactive</MenuItem>
                      <MenuItem value="training">Training</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModelConfigOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setModelConfigOpen(false)}>
            Save Configuration
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default RiskScoring;