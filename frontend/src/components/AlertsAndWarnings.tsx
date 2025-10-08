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
  Alert,
  Badge,
  Tooltip,
  Tabs,
  Tab,
  Divider,
  LinearProgress
} from '@mui/material';
import {
  Warning,
  Error,
  Info,
  CheckCircle,
  Visibility,
  Assignment,
  Close,
  FilterList,
  Search,
  Refresh,
  Download,
  NotificationsActive,
  Schedule,
  Person,
  LocationOn,
  Phone,
  Security,
  TrendingUp,
  PlayArrow,
  Pause
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, LineChart, Line } from 'recharts';

interface FraudAlert {
  id: string;
  type: 'critical' | 'high' | 'medium' | 'low';
  category: 'transaction' | 'behavior' | 'location' | 'network' | 'system';
  title: string;
  description: string;
  riskScore: number;
  timestamp: Date;
  status: 'new' | 'investigating' | 'resolved' | 'false_positive';
  assignedTo?: string;
  source: string;
  affectedUser?: string;
  affectedNumber?: string;
  location?: string;
  transactionAmount?: number;
  evidenceCount: number;
  relatedAlerts: number;
}

interface AlertStats {
  total: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  resolved: number;
  investigating: number;
  new: number;
}

const AlertsAndWarnings: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedAlert, setSelectedAlert] = useState<FraudAlert | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [assignDialogOpen, setAssignDialogOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Mock data
  const [alerts, setAlerts] = useState<FraudAlert[]>([
    {
      id: 'ALT-001',
      type: 'critical',
      category: 'transaction',
      title: 'Suspicious High-Value Transaction Pattern',
      description: 'Multiple high-value transactions detected from user account within short time frame',
      riskScore: 95,
      timestamp: new Date(Date.now() - 300000),
      status: 'new',
      source: 'Transaction ML Engine',
      affectedUser: 'user_12345',
      affectedNumber: '+1-555-0123',
      transactionAmount: 50000,
      evidenceCount: 7,
      relatedAlerts: 3
    },
    {
      id: 'ALT-002',
      type: 'high',
      category: 'behavior',
      title: 'Abnormal Call Pattern Detected',
      description: 'User exhibiting unusual calling behavior with significant deviation from baseline',
      riskScore: 87,
      timestamp: new Date(Date.now() - 600000),
      status: 'investigating',
      assignedTo: 'John Smith',
      source: 'Behavioral Analytics',
      affectedUser: 'user_67890',
      affectedNumber: '+1-555-0456',
      evidenceCount: 12,
      relatedAlerts: 2
    },
    {
      id: 'ALT-003',
      type: 'medium',
      category: 'location',
      title: 'Geographic Anomaly Alert',
      description: 'User location shows impossible geographic movement pattern',
      riskScore: 72,
      timestamp: new Date(Date.now() - 900000),
      status: 'new',
      source: 'Location Analytics Engine',
      affectedUser: 'user_24680',
      affectedNumber: '+1-555-0789',
      location: 'Chicago, IL → Tokyo, JP',
      evidenceCount: 5,
      relatedAlerts: 1
    },
    {
      id: 'ALT-004',
      type: 'high',
      category: 'network',
      title: 'Potential SIM Box Detected',
      description: 'Network traffic patterns suggest presence of SIM box operation',
      riskScore: 89,
      timestamp: new Date(Date.now() - 1200000),
      status: 'investigating',
      assignedTo: 'Sarah Johnson',
      source: 'Network Traffic Analyzer',
      location: 'Miami, FL',
      evidenceCount: 15,
      relatedAlerts: 8
    },
    {
      id: 'ALT-005',
      type: 'low',
      category: 'system',
      title: 'Minor Performance Degradation',
      description: 'Slight decrease in system performance metrics detected',
      riskScore: 45,
      timestamp: new Date(Date.now() - 1800000),
      status: 'resolved',
      assignedTo: 'Mike Wilson',
      source: 'System Monitor',
      evidenceCount: 3,
      relatedAlerts: 0
    }
  ]);

  const alertStats: AlertStats = {
    total: alerts.length,
    critical: alerts.filter(a => a.type === 'critical').length,
    high: alerts.filter(a => a.type === 'high').length,
    medium: alerts.filter(a => a.type === 'medium').length,
    low: alerts.filter(a => a.type === 'low').length,
    resolved: alerts.filter(a => a.status === 'resolved').length,
    investigating: alerts.filter(a => a.status === 'investigating').length,
    new: alerts.filter(a => a.status === 'new').length,
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'critical': return '#d32f2f';
      case 'high': return '#f57c00';
      case 'medium': return '#fbc02d';
      case 'low': return '#388e3c';
      default: return '#757575';
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical': return <Error sx={{ color: '#d32f2f' }} />;
      case 'high': return <Warning sx={{ color: '#f57c00' }} />;
      case 'medium': return <Info sx={{ color: '#fbc02d' }} />;
      case 'low': return <CheckCircle sx={{ color: '#388e3c' }} />;
      default: return <Info sx={{ color: '#757575' }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return 'error';
      case 'investigating': return 'warning';
      case 'resolved': return 'success';
      case 'false_positive': return 'info';
      default: return 'default';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    const matchesSearch = alert.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         alert.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         alert.affectedUser?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = filterType === 'all' || alert.type === filterType;
    const matchesStatus = filterStatus === 'all' || alert.status === filterStatus;
    
    return matchesSearch && matchesType && matchesStatus;
  });

  const handleViewDetails = (alert: FraudAlert) => {
    setSelectedAlert(alert);
    setDetailsOpen(true);
  };

  const handleAssignAlert = (alert: FraudAlert) => {
    setSelectedAlert(alert);
    setAssignDialogOpen(true);
  };

  const handleStatusChange = (alertId: string, newStatus: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, status: newStatus as any } : alert
    ));
  };

  const pieData = [
    { name: 'Critical', value: alertStats.critical, color: '#d32f2f' },
    { name: 'High', value: alertStats.high, color: '#f57c00' },
    { name: 'Medium', value: alertStats.medium, color: '#fbc02d' },
    { name: 'Low', value: alertStats.low, color: '#388e3c' }
  ];

  const trendData = [
    { time: '00:00', alerts: 12 },
    { time: '04:00', alerts: 8 },
    { time: '08:00', alerts: 25 },
    { time: '12:00', alerts: 45 },
    { time: '16:00', alerts: 38 },
    { time: '20:00', alerts: 22 },
  ];

  return (
    <Box sx={{ p: 3, minHeight: '100vh', bgcolor: '#f5f5f5' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          Alerts & Warnings
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={autoRefresh ? <Pause /> : <PlayArrow />}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? 'Pause' : 'Start'} Auto-refresh
          </Button>
          <Button variant="contained" startIcon={<Refresh />}>
            Refresh
          </Button>
          <Button variant="contained" startIcon={<Download />} color="secondary">
            Export
          </Button>
        </Box>
      </Box>

      {/* Alert Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {alertStats.total}
              </Typography>
              <Typography variant="body2">Total Alerts</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #d32f2f 0%, #f44336 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {alertStats.critical}
              </Typography>
              <Typography variant="body2">Critical</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #f57c00 0%, #ff9800 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {alertStats.high}
              </Typography>
              <Typography variant="body2">High</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #fbc02d 0%, #ffeb3b 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {alertStats.medium}
              </Typography>
              <Typography variant="body2">Medium</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #388e3c 0%, #4caf50 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {alertStats.new}
              </Typography>
              <Typography variant="body2">New</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #7b1fa2 0%, #9c27b0 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {alertStats.investigating}
              </Typography>
              <Typography variant="body2">Investigating</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 300 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Alert Distribution
            </Typography>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
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
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: 300 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Alert Trend (Last 24 Hours)
            </Typography>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <RechartsTooltip />
                <Line type="monotone" dataKey="alerts" stroke="#1976d2" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Filters and Search */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search Alerts"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />,
              }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Alert Type</InputLabel>
              <Select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                label="Alert Type"
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                label="Status"
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="new">New</MenuItem>
                <MenuItem value="investigating">Investigating</MenuItem>
                <MenuItem value="resolved">Resolved</MenuItem>
                <MenuItem value="false_positive">False Positive</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<FilterList />}
              onClick={() => {
                setSearchTerm('');
                setFilterType('all');
                setFilterStatus('all');
              }}
            >
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Alerts Table */}
      <Paper sx={{ p: 3 }}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Alert</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Risk Score</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Assigned To</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredAlerts
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((alert) => (
                <TableRow key={alert.id} hover>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getAlertIcon(alert.type)}
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                          {alert.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {alert.id} • {alert.source}
                        </Typography>
                      </Box>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={alert.type.toUpperCase()}
                      size="small"
                      sx={{
                        backgroundColor: getAlertColor(alert.type),
                        color: 'white',
                        fontWeight: 'bold'
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                        {alert.riskScore}
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={alert.riskScore}
                        sx={{ width: 60, height: 6 }}
                        color={alert.riskScore > 80 ? 'error' : alert.riskScore > 60 ? 'warning' : 'success'}
                      />
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={alert.status.replace('_', ' ').toUpperCase()}
                      size="small"
                      color={getStatusColor(alert.status) as any}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    {alert.assignedTo ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Person sx={{ fontSize: 16 }} />
                        {alert.assignedTo}
                      </Box>
                    ) : (
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleAssignAlert(alert)}
                      >
                        Assign
                      </Button>
                    )}
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {alert.timestamp.toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewDetails(alert)}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Assign">
                        <IconButton
                          size="small"
                          onClick={() => handleAssignAlert(alert)}
                        >
                          <Assignment />
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
          count={filteredAlerts.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={(_, newPage) => setPage(newPage)}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
        />
      </Paper>

      {/* Alert Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {selectedAlert && getAlertIcon(selectedAlert.type)}
            Alert Details - {selectedAlert?.id}
            <Chip
              label={selectedAlert?.type.toUpperCase()}
              size="small"
              sx={{
                backgroundColor: selectedAlert ? getAlertColor(selectedAlert.type) : '#757575',
                color: 'white',
                fontWeight: 'bold'
              }}
            />
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedAlert && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" sx={{ mb: 1 }}>
                  {selectedAlert.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                  {selectedAlert.description}
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                  Alert Information
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Typography variant="body2">
                    <strong>Risk Score:</strong> {selectedAlert.riskScore}/100
                  </Typography>
                  <Typography variant="body2">
                    <strong>Category:</strong> {selectedAlert.category}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Source:</strong> {selectedAlert.source}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Timestamp:</strong> {selectedAlert.timestamp.toLocaleString()}
                  </Typography>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                  Affected Details
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {selectedAlert.affectedUser && (
                    <Typography variant="body2">
                      <strong>User:</strong> {selectedAlert.affectedUser}
                    </Typography>
                  )}
                  {selectedAlert.affectedNumber && (
                    <Typography variant="body2">
                      <strong>Phone:</strong> {selectedAlert.affectedNumber}
                    </Typography>
                  )}
                  {selectedAlert.location && (
                    <Typography variant="body2">
                      <strong>Location:</strong> {selectedAlert.location}
                    </Typography>
                  )}
                  {selectedAlert.transactionAmount && (
                    <Typography variant="body2">
                      <strong>Amount:</strong> ${selectedAlert.transactionAmount.toLocaleString()}
                    </Typography>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                  Investigation Status
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <Typography variant="body2">
                    <strong>Evidence Items:</strong> {selectedAlert.evidenceCount}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Related Alerts:</strong> {selectedAlert.relatedAlerts}
                  </Typography>
                </Box>
                
                <FormControl sx={{ minWidth: 200 }}>
                  <InputLabel>Change Status</InputLabel>
                  <Select
                    value={selectedAlert.status}
                    onChange={(e) => handleStatusChange(selectedAlert.id, e.target.value)}
                    label="Change Status"
                  >
                    <MenuItem value="new">New</MenuItem>
                    <MenuItem value="investigating">Investigating</MenuItem>
                    <MenuItem value="resolved">Resolved</MenuItem>
                    <MenuItem value="false_positive">False Positive</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          <Button variant="contained" onClick={() => handleAssignAlert(selectedAlert!)}>
            Assign Alert
          </Button>
        </DialogActions>
      </Dialog>

      {/* Assign Dialog */}
      <Dialog
        open={assignDialogOpen}
        onClose={() => setAssignDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Assign Alert</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Assign To</InputLabel>
              <Select
                defaultValue=""
                label="Assign To"
              >
                <MenuItem value="john.smith">John Smith</MenuItem>
                <MenuItem value="sarah.johnson">Sarah Johnson</MenuItem>
                <MenuItem value="mike.wilson">Mike Wilson</MenuItem>
                <MenuItem value="alice.brown">Alice Brown</MenuItem>
              </Select>
            </FormControl>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="Assignment Notes"
              placeholder="Add any notes for the assignee..."
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAssignDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => {
              setAssignDialogOpen(false);
              // Handle assignment logic here
            }}
          >
            Assign
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AlertsAndWarnings;