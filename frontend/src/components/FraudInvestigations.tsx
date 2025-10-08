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
  Tabs,
  Tab,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Badge,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
} from '@mui/lab';
import {
  Visibility,
  Edit,
  Add,
  Search,
  FilterList,
  Download,
  Assignment,
  Person,
  Phone,
  LocationOn,
  AttachMoney,
  Schedule,
  Security,
  Warning,
  CheckCircle,
  Description,
  Link,
  Timeline as TimelineIcon,
  Assessment,
  Flag,
  Close,
  Save,
  Send,
  AttachFile
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, LineChart, Line } from 'recharts';

interface Investigation {
  id: string;
  caseNumber: string;
  title: string;
  description: string;
  status: 'open' | 'investigating' | 'pending_review' | 'closed' | 'escalated';
  priority: 'low' | 'medium' | 'high' | 'critical';
  investigator: string;
  createdDate: Date;
  lastUpdated: Date;
  estimatedLoss: number;
  affectedUsers: number;
  evidenceCount: number;
  relatedAlerts: string[];
  category: 'fraud' | 'abuse' | 'security' | 'compliance';
  riskScore: number;
  tags: string[];
}

interface Evidence {
  id: string;
  type: 'document' | 'screenshot' | 'log' | 'transaction' | 'communication' | 'witness';
  title: string;
  description: string;
  uploadedBy: string;
  uploadDate: Date;
  size: string;
  relevanceScore: number;
}

interface TimelineEvent {
  id: string;
  timestamp: Date;
  event: string;
  description: string;
  user: string;
  type: 'creation' | 'update' | 'assignment' | 'evidence' | 'communication' | 'status_change';
}

const FraudInvestigations: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedInvestigation, setSelectedInvestigation] = useState<Investigation | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [newCaseOpen, setNewCaseOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterPriority, setFilterPriority] = useState('all');
  const [activeStep, setActiveStep] = useState(0);

  // Mock data
  const [investigations, setInvestigations] = useState<Investigation[]>([
    {
      id: 'INV-001',
      caseNumber: 'CASE-2025-001',
      title: 'Multi-Account Transaction Fraud Ring',
      description: 'Investigation into coordinated fraudulent transactions across multiple user accounts',
      status: 'investigating',
      priority: 'critical',
      investigator: 'John Smith',
      createdDate: new Date(Date.now() - 86400000 * 3),
      lastUpdated: new Date(Date.now() - 3600000),
      estimatedLoss: 125000,
      affectedUsers: 8,
      evidenceCount: 15,
      relatedAlerts: ['ALT-001', 'ALT-007', 'ALT-012'],
      category: 'fraud',
      riskScore: 95,
      tags: ['high-value', 'coordinated', 'urgent']
    },
    {
      id: 'INV-002',
      caseNumber: 'CASE-2025-002',
      title: 'SIM Swapping Attack Investigation',
      description: 'Investigation into SIM swapping attacks targeting high-value accounts',
      status: 'open',
      priority: 'high',
      investigator: 'Sarah Johnson',
      createdDate: new Date(Date.now() - 86400000 * 2),
      lastUpdated: new Date(Date.now() - 7200000),
      estimatedLoss: 85000,
      affectedUsers: 3,
      evidenceCount: 9,
      relatedAlerts: ['ALT-003', 'ALT-008'],
      category: 'security',
      riskScore: 87,
      tags: ['sim-swap', 'identity-theft']
    },
    {
      id: 'INV-003',
      caseNumber: 'CASE-2025-003',
      title: 'Unusual Call Pattern Analysis',
      description: 'Investigation into abnormal calling patterns suggesting potential fraud',
      status: 'pending_review',
      priority: 'medium',
      investigator: 'Mike Wilson',
      createdDate: new Date(Date.now() - 86400000 * 5),
      lastUpdated: new Date(Date.now() - 14400000),
      estimatedLoss: 25000,
      affectedUsers: 12,
      evidenceCount: 22,
      relatedAlerts: ['ALT-004', 'ALT-009', 'ALT-015'],
      category: 'abuse',
      riskScore: 72,
      tags: ['pattern-analysis', 'behavioral']
    }
  ]);

  const [evidenceItems] = useState<Evidence[]>([
    {
      id: 'EV-001',
      type: 'transaction',
      title: 'Suspicious Transaction Log',
      description: 'Complete transaction history showing coordinated fraudulent activities',
      uploadedBy: 'John Smith',
      uploadDate: new Date(Date.now() - 86400000),
      size: '2.3 MB',
      relevanceScore: 95
    },
    {
      id: 'EV-002',
      type: 'screenshot',
      title: 'User Account Screenshots',
      description: 'Screenshots of compromised user accounts showing unauthorized changes',
      uploadedBy: 'Sarah Johnson',
      uploadDate: new Date(Date.now() - 43200000),
      size: '1.8 MB',
      relevanceScore: 88
    },
    {
      id: 'EV-003',
      type: 'log',
      title: 'System Access Logs',
      description: 'Server logs showing suspicious access patterns and login attempts',
      uploadedBy: 'Mike Wilson',
      uploadDate: new Date(Date.now() - 21600000),
      size: '4.1 MB',
      relevanceScore: 92
    }
  ]);

  const [timelineEvents] = useState<TimelineEvent[]>([
    {
      id: 'TL-001',
      timestamp: new Date(Date.now() - 3600000),
      event: 'Evidence Added',
      description: 'New transaction logs uploaded to case',
      user: 'John Smith',
      type: 'evidence'
    },
    {
      id: 'TL-002',
      timestamp: new Date(Date.now() - 7200000),
      event: 'Status Updated',
      description: 'Case status changed to "Investigating"',
      user: 'Sarah Johnson',
      type: 'status_change'
    },
    {
      id: 'TL-003',
      timestamp: new Date(Date.now() - 14400000),
      event: 'Case Assigned',
      description: 'Case assigned to investigation team',
      user: 'System',
      type: 'assignment'
    }
  ]);

  const investigationSteps = [
    'Initial Assessment',
    'Evidence Collection',
    'Analysis & Investigation',
    'Documentation',
    'Review & Approval',
    'Case Closure'
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'info';
      case 'investigating': return 'warning';
      case 'pending_review': return 'secondary';
      case 'closed': return 'success';
      case 'escalated': return 'error';
      default: return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return '#d32f2f';
      case 'high': return '#f57c00';
      case 'medium': return '#fbc02d';
      case 'low': return '#388e3c';
      default: return '#757575';
    }
  };

  const getEvidenceIcon = (type: string) => {
    switch (type) {
      case 'document': return <Description />;
      case 'screenshot': return <AttachFile />;
      case 'log': return <Assessment />;
      case 'transaction': return <AttachMoney />;
      case 'communication': return <Phone />;
      case 'witness': return <Person />;
      default: return <Description />;
    }
  };

  const getTimelineIcon = (type: string) => {
    switch (type) {
      case 'creation': return <Add />;
      case 'update': return <Edit />;
      case 'assignment': return <Assignment />;
      case 'evidence': return <Evidence />;
      case 'communication': return <Phone />;
      case 'status_change': return <Flag />;
      default: return <TimelineIcon />;
    }
  };

  const filteredInvestigations = investigations.filter(inv => {
    const matchesSearch = inv.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         inv.caseNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         inv.investigator.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || inv.status === filterStatus;
    const matchesPriority = filterPriority === 'all' || inv.priority === filterPriority;
    
    return matchesSearch && matchesStatus && matchesPriority;
  });

  const investigationStats = {
    total: investigations.length,
    open: investigations.filter(i => i.status === 'open').length,
    investigating: investigations.filter(i => i.status === 'investigating').length,
    pending: investigations.filter(i => i.status === 'pending_review').length,
    closed: investigations.filter(i => i.status === 'closed').length,
    critical: investigations.filter(i => i.priority === 'critical').length,
    totalLoss: investigations.reduce((sum, i) => sum + i.estimatedLoss, 0)
  };

  const statusData = [
    { name: 'Open', value: investigationStats.open, color: '#2196f3' },
    { name: 'Investigating', value: investigationStats.investigating, color: '#ff9800' },
    { name: 'Pending Review', value: investigationStats.pending, color: '#9c27b0' },
    { name: 'Closed', value: investigationStats.closed, color: '#4caf50' }
  ];

  const handleViewDetails = (investigation: Investigation) => {
    setSelectedInvestigation(investigation);
    setDetailsOpen(true);
  };

  return (
    <Box sx={{ p: 3, minHeight: '100vh', bgcolor: '#f5f5f5' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
          Fraud Investigations
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setNewCaseOpen(true)}
          >
            New Investigation
          </Button>
          <Button variant="outlined" startIcon={<Download />}>
            Export Cases
          </Button>
        </Box>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {investigationStats.total}
              </Typography>
              <Typography variant="body2">Total Cases</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #2196f3 0%, #64b5f6 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {investigationStats.open}
              </Typography>
              <Typography variant="body2">Open</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #ff9800 0%, #ffb74d 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {investigationStats.investigating}
              </Typography>
              <Typography variant="body2">Investigating</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #d32f2f 0%, #f44336 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {investigationStats.critical}
              </Typography>
              <Typography variant="body2">Critical</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #4caf50 0%, #81c784 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                {investigationStats.closed}
              </Typography>
              <Typography variant="body2">Closed</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <Card sx={{ background: 'linear-gradient(135deg, #f57c00 0%, #ffb74d 100%)', color: 'white' }}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
                ${(investigationStats.totalLoss / 1000).toFixed(0)}K
              </Typography>
              <Typography variant="body2">Total Loss</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 300 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
              Case Status Distribution
            </Typography>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {statusData.map((entry, index) => (
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
              Investigation Timeline
            </Typography>
            <Box sx={{ height: '85%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                Investigation timeline chart would be displayed here
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Filters */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search Investigations"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />,
              }}
            />
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
                <MenuItem value="open">Open</MenuItem>
                <MenuItem value="investigating">Investigating</MenuItem>
                <MenuItem value="pending_review">Pending Review</MenuItem>
                <MenuItem value="closed">Closed</MenuItem>
                <MenuItem value="escalated">Escalated</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Priority</InputLabel>
              <Select
                value={filterPriority}
                onChange={(e) => setFilterPriority(e.target.value)}
                label="Priority"
              >
                <MenuItem value="all">All Priorities</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="low">Low</MenuItem>
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
                setFilterStatus('all');
                setFilterPriority('all');
              }}
            >
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Investigations Table */}
      <Paper sx={{ p: 3 }}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Case</TableCell>
                <TableCell>Priority</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Investigator</TableCell>
                <TableCell>Estimated Loss</TableCell>
                <TableCell>Evidence</TableCell>
                <TableCell>Last Updated</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredInvestigations
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((investigation) => (
                <TableRow key={investigation.id} hover>
                  <TableCell>
                    <Box>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                        {investigation.title}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {investigation.caseNumber} • {investigation.category}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={investigation.priority.toUpperCase()}
                      size="small"
                      sx={{
                        backgroundColor: getPriorityColor(investigation.priority),
                        color: 'white',
                        fontWeight: 'bold'
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={investigation.status.replace('_', ' ').toUpperCase()}
                      size="small"
                      color={getStatusColor(investigation.status) as any}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Person sx={{ fontSize: 16 }} />
                      {investigation.investigator}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#d32f2f' }}>
                      ${investigation.estimatedLoss.toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Badge badgeContent={investigation.evidenceCount} color="primary">
                      <AttachFile />
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {investigation.lastUpdated.toLocaleDateString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewDetails(investigation)}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit">
                        <IconButton size="small">
                          <Edit />
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
          count={filteredInvestigations.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={(_, newPage) => setPage(newPage)}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
        />
      </Paper>

      {/* Investigation Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="h6">
                Investigation Details - {selectedInvestigation?.caseNumber}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {selectedInvestigation?.title}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                label={selectedInvestigation?.priority.toUpperCase()}
                size="small"
                sx={{
                  backgroundColor: selectedInvestigation ? getPriorityColor(selectedInvestigation.priority) : '#757575',
                  color: 'white',
                  fontWeight: 'bold'
                }}
              />
              <Chip
                label={selectedInvestigation?.status.replace('_', ' ').toUpperCase()}
                size="small"
                color={selectedInvestigation ? getStatusColor(selectedInvestigation.status) as any : 'default'}
                variant="outlined"
              />
            </Box>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
            <Tab label="Overview" />
            <Tab label="Evidence" />
            <Tab label="Timeline" />
            <Tab label="Progress" />
          </Tabs>

          <Box sx={{ mt: 3 }}>
            {selectedTab === 0 && selectedInvestigation && (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Case Information
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography variant="body1">
                      <strong>Description:</strong> {selectedInvestigation.description}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Investigator:</strong> {selectedInvestigation.investigator}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Created:</strong> {selectedInvestigation.createdDate.toLocaleDateString()}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Last Updated:</strong> {selectedInvestigation.lastUpdated.toLocaleDateString()}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Category:</strong> {selectedInvestigation.category}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Impact Assessment
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography variant="body2">
                      <strong>Estimated Loss:</strong> ${selectedInvestigation.estimatedLoss.toLocaleString()}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Affected Users:</strong> {selectedInvestigation.affectedUsers}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Risk Score:</strong> {selectedInvestigation.riskScore}/100
                    </Typography>
                    <Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Tags:</strong>
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {selectedInvestigation.tags.map((tag, index) => (
                          <Chip key={index} label={tag} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  </Box>
                </Grid>
              </Grid>
            )}

            {selectedTab === 1 && (
              <Box>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
                  Evidence Items ({evidenceItems.length})
                </Typography>
                <List>
                  {evidenceItems.map((evidence) => (
                    <React.Fragment key={evidence.id}>
                      <ListItem>
                        <ListItemIcon>
                          {getEvidenceIcon(evidence.type)}
                        </ListItemIcon>
                        <ListItemText
                          primary={evidence.title}
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                {evidence.description}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Uploaded by {evidence.uploadedBy} • {evidence.uploadDate.toLocaleDateString()} • {evidence.size} • Relevance: {evidence.relevanceScore}%
                              </Typography>
                            </Box>
                          }
                        />
                        <Button size="small" variant="outlined">
                          View
                        </Button>
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                </List>
                <Button
                  variant="contained"
                  startIcon={<Add />}
                  sx={{ mt: 2 }}
                >
                  Add Evidence
                </Button>
              </Box>
            )}

            {selectedTab === 2 && (
              <Box>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
                  Investigation Timeline
                </Typography>
                <Timeline>
                  {timelineEvents.map((event) => (
                    <TimelineItem key={event.id}>
                      <TimelineOppositeContent sx={{ m: 'auto 0' }} variant="body2" color="text.secondary">
                        {event.timestamp.toLocaleString()}
                      </TimelineOppositeContent>
                      <TimelineSeparator>
                        <TimelineDot color="primary">
                          {getTimelineIcon(event.type)}
                        </TimelineDot>
                        <TimelineConnector />
                      </TimelineSeparator>
                      <TimelineContent sx={{ py: '12px', px: 2 }}>
                        <Typography variant="h6" component="span">
                          {event.event}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {event.description}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          by {event.user}
                        </Typography>
                      </TimelineContent>
                    </TimelineItem>
                  ))}
                </Timeline>
              </Box>
            )}

            {selectedTab === 3 && (
              <Box>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold' }}>
                  Investigation Progress
                </Typography>
                <Stepper activeStep={2} orientation="vertical">
                  {investigationSteps.map((step, index) => (
                    <Step key={step}>
                      <StepLabel>
                        {step}
                      </StepLabel>
                      <StepContent>
                        <Typography variant="body2" color="text.secondary">
                          {index <= 2 ? 'Completed' : 'Pending'}
                        </Typography>
                      </StepContent>
                    </Step>
                  ))}
                </Stepper>
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          <Button variant="contained">Update Case</Button>
        </DialogActions>
      </Dialog>

      {/* New Case Dialog */}
      <Dialog
        open={newCaseOpen}
        onClose={() => setNewCaseOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Investigation</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Investigation Title"
                placeholder="Enter investigation title..."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Description"
                placeholder="Describe the investigation details..."
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select defaultValue="medium" label="Priority">
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="critical">Critical</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select defaultValue="fraud" label="Category">
                  <MenuItem value="fraud">Fraud</MenuItem>
                  <MenuItem value="abuse">Abuse</MenuItem>
                  <MenuItem value="security">Security</MenuItem>
                  <MenuItem value="compliance">Compliance</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Assign To</InputLabel>
                <Select defaultValue="" label="Assign To">
                  <MenuItem value="john.smith">John Smith</MenuItem>
                  <MenuItem value="sarah.johnson">Sarah Johnson</MenuItem>
                  <MenuItem value="mike.wilson">Mike Wilson</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Estimated Loss ($)"
                placeholder="0"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewCaseOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => {
              setNewCaseOpen(false);
              // Handle case creation logic here
            }}
          >
            Create Investigation
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FraudInvestigations;