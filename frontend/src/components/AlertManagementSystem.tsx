/**
 * Alert Management System
 * Professional alert workflow management with assign, priority, notes, resolve, and escalate
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
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Avatar,
  Divider,
  FormControlLabel,
  Checkbox,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Badge,
  styled,
} from '@mui/material';
import {
  Close,
  Notifications,
  Person,
  PriorityHigh,
  Note,
  CheckCircle,
  ArrowUpward,
  Edit,
  Visibility,
  Schedule,
  Warning,
  Error,
  Info,
  Assignment,
  Comment,
  History,
  FilterList,
  Sort,
  MoreVert,
  Flag,
  Check,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const AlertCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  cursor: 'pointer',
  '&:hover': {
    borderColor: customColors.primary[300],
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    transform: 'translateY(-1px)',
  },
}));

const PriorityIndicator = styled(Box)<{ priority: 'low' | 'medium' | 'high' | 'critical' }>(({ priority }) => ({
  width: 4,
  height: '100%',
  position: 'absolute',
  left: 0,
  top: 0,
  borderRadius: '4px 0 0 4px',
  backgroundColor: 
    priority === 'critical' ? '#d32f2f' :
    priority === 'high' ? customColors.warning[500] :
    priority === 'medium' ? customColors.primary[500] :
    customColors.neutral[400],
}));

interface AlertManagementSystemProps {
  open: boolean;
  onClose: () => void;
}

interface Alert {
  id: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'new' | 'assigned' | 'in-progress' | 'resolved' | 'escalated';
  type: 'fraud' | 'security' | 'system' | 'compliance';
  assignedTo?: string;
  reporter: string;
  createdAt: Date;
  updatedAt: Date;
  dueDate?: Date;
  notes: string[];
  tags: string[];
  riskScore: number;
}

const AlertManagementSystem: React.FC<AlertManagementSystemProps> = ({
  open,
  onClose
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterPriority, setFilterPriority] = useState('all');
  const [filterType, setFilterType] = useState('all');

  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: 'ALT-001',
      title: 'Suspicious Transaction Pattern Detected',
      description: 'Multiple high-value transactions from the same IP address within 30 minutes',
      priority: 'high',
      status: 'new',
      type: 'fraud',
      reporter: 'System Alert',
      createdAt: new Date('2024-01-15T14:30:00'),
      updatedAt: new Date('2024-01-15T14:30:00'),
      dueDate: new Date('2024-01-16T14:30:00'),
      notes: [],
      tags: ['transaction', 'pattern', 'ip-address'],
      riskScore: 8.7
    },
    {
      id: 'ALT-002',
      title: 'Account Takeover Attempt',
      description: 'Failed login attempts from multiple locations for user ID 12345',
      priority: 'critical',
      status: 'assigned',
      type: 'security',
      assignedTo: 'Sarah Johnson',
      reporter: 'Security System',
      createdAt: new Date('2024-01-15T13:45:00'),
      updatedAt: new Date('2024-01-15T14:00:00'),
      dueDate: new Date('2024-01-15T18:00:00'),
      notes: ['Initial investigation started', 'User contacted via secure channel'],
      tags: ['account-takeover', 'authentication', 'high-risk'],
      riskScore: 9.2
    },
    {
      id: 'ALT-003',
      title: 'System Performance Degradation',
      description: 'Transaction processing time increased by 200% in the last hour',
      priority: 'medium',
      status: 'in-progress',
      type: 'system',
      assignedTo: 'Mike Chen',
      reporter: 'Monitoring System',
      createdAt: new Date('2024-01-15T13:00:00'),
      updatedAt: new Date('2024-01-15T14:15:00'),
      notes: ['Database performance issues identified', 'Scaling additional resources'],
      tags: ['performance', 'database', 'latency'],
      riskScore: 6.1
    },
    {
      id: 'ALT-004',
      title: 'Compliance Violation Detected',
      description: 'Transaction exceeds daily limit without proper authorization',
      priority: 'high',
      status: 'escalated',
      type: 'compliance',
      assignedTo: 'Legal Team',
      reporter: 'Compliance System',
      createdAt: new Date('2024-01-15T12:30:00'),
      updatedAt: new Date('2024-01-15T14:10:00'),
      dueDate: new Date('2024-01-17T12:30:00'),
      notes: ['Escalated to legal department', 'Customer documentation required'],
      tags: ['compliance', 'authorization', 'limits'],
      riskScore: 7.8
    },
    {
      id: 'ALT-005',
      title: 'Unusual Geographic Activity',
      description: 'Credit card used in 3 different countries within 2 hours',
      priority: 'high',
      status: 'resolved',
      type: 'fraud',
      assignedTo: 'John Smith',
      reporter: 'Fraud Detection AI',
      createdAt: new Date('2024-01-15T11:00:00'),
      updatedAt: new Date('2024-01-15T13:30:00'),
      notes: ['Customer confirmed travel itinerary', 'False positive - resolved'],
      tags: ['geographic', 'travel', 'credit-card'],
      riskScore: 8.1
    }
  ]);

  const [assignmentData, setAssignmentData] = useState({
    analyst: '',
    priority: 'medium',
    dueDate: '',
    notes: ''
  });

  const handleAssignAlert = () => {
    if (!selectedAlert || !assignmentData.analyst) return;

    setAlerts(prev => prev.map(alert =>
      alert.id === selectedAlert.id
        ? {
            ...alert,
            assignedTo: assignmentData.analyst,
            priority: assignmentData.priority as any,
            status: 'assigned',
            dueDate: assignmentData.dueDate ? new Date(assignmentData.dueDate) : alert.dueDate,
            notes: assignmentData.notes 
              ? [...alert.notes, `Assigned to ${assignmentData.analyst}: ${assignmentData.notes}`]
              : [...alert.notes, `Assigned to ${assignmentData.analyst}`],
            updatedAt: new Date()
          }
        : alert
    ));

    alert(`Alert ${selectedAlert.id} assigned to ${assignmentData.analyst} successfully!`);
    setSelectedAlert(null);
  };

  const handleChangePriority = (alertId: string, newPriority: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? {
            ...alert,
            priority: newPriority as any,
            notes: [...alert.notes, `Priority changed to ${newPriority.toUpperCase()}`],
            updatedAt: new Date()
          }
        : alert
    ));
  };

  const handleAddNote = (alertId: string, note: string) => {
    if (!note.trim()) return;

    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? {
            ...alert,
            notes: [...alert.notes, note],
            updatedAt: new Date()
          }
        : alert
    ));
  };

  const handleResolveAlert = (alertId: string, resolution: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? {
            ...alert,
            status: 'resolved',
            notes: [...alert.notes, `RESOLVED: ${resolution}`],
            updatedAt: new Date()
          }
        : alert
    ));
  };

  const handleEscalateAlert = (alertId: string, escalationReason: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? {
            ...alert,
            status: 'escalated',
            priority: alert.priority === 'critical' ? 'critical' : 'high',
            notes: [...alert.notes, `ESCALATED: ${escalationReason}`],
            updatedAt: new Date()
          }
        : alert
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return customColors.primary[500];
      case 'assigned': return customColors.warning[500];
      case 'in-progress': return '#1976d2';
      case 'resolved': return customColors.success[500];
      case 'escalated': return '#d32f2f';
      default: return customColors.neutral[500];
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'fraud': return <Warning />;
      case 'security': return <Error />;
      case 'system': return <Info />;
      case 'compliance': return <Assignment />;
      default: return <Notifications />;
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filterStatus !== 'all' && alert.status !== filterStatus) return false;
    if (filterPriority !== 'all' && alert.priority !== filterPriority) return false;
    if (filterType !== 'all' && alert.type !== filterType) return false;
    return true;
  });

  const renderAlertsList = () => (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Alert Management Dashboard
        </Typography>
        <Box display="flex" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="new">New</MenuItem>
              <MenuItem value="assigned">Assigned</MenuItem>
              <MenuItem value="in-progress">In Progress</MenuItem>
              <MenuItem value="resolved">Resolved</MenuItem>
              <MenuItem value="escalated">Escalated</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Priority</InputLabel>
            <Select
              value={filterPriority}
              onChange={(e) => setFilterPriority(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>
          <Button startIcon={<FilterList />} variant="outlined" size="small">
            More Filters
          </Button>
        </Box>
      </Box>

      <Grid container spacing={2}>
        {filteredAlerts.map((alert) => (
          <Grid item xs={12} key={alert.id}>
            <AlertCard onClick={() => setSelectedAlert(alert)}>
              <PriorityIndicator priority={alert.priority} />
              <CardContent sx={{ position: 'relative', ml: 1 }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} md={6}>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      {getTypeIcon(alert.type)}
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {alert.title}
                      </Typography>
                      <Chip 
                        label={alert.id} 
                        size="small" 
                        variant="outlined"
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {alert.description}
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={0.5}>
                      {alert.tags.map((tag, index) => (
                        <Chip key={index} label={tag} size="small" variant="filled" />
                      ))}
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Box mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Status:</strong>
                      </Typography>
                      <Chip
                        label={alert.status.replace('-', ' ').toUpperCase()}
                        size="small"
                        sx={{ 
                          bgcolor: getStatusColor(alert.status),
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    </Box>
                    <Box mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Priority:</strong>
                      </Typography>
                      <Chip
                        label={alert.priority.toUpperCase()}
                        size="small"
                        color={alert.priority === 'critical' ? 'error' : 
                               alert.priority === 'high' ? 'warning' : 'primary'}
                      />
                    </Box>
                    {alert.assignedTo && (
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Assigned to:</strong>
                        </Typography>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Avatar sx={{ width: 24, height: 24, fontSize: '0.75rem' }}>
                            {alert.assignedTo.charAt(0)}
                          </Avatar>
                          <Typography variant="body2">
                            {alert.assignedTo}
                          </Typography>
                        </Box>
                      </Box>
                    )}
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Risk Score: {alert.riskScore}/10
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Created: {alert.createdAt.toLocaleDateString()}
                        </Typography>
                      </Box>
                      <Box display="flex" gap={1}>
                        <Tooltip title="View Details">
                          <IconButton size="small">
                            <Visibility />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
                          <IconButton size="small">
                            <Edit />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Actions">
                          <IconButton size="small">
                            <MoreVert />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </AlertCard>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const renderAlertDetails = () => {
    if (!selectedAlert) return null;

    return (
      <Box p={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Alert Details - {selectedAlert.id}
          </Typography>
          <Button onClick={() => setSelectedAlert(null)}>
            Back to List
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  {selectedAlert.title}
                </Typography>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  {selectedAlert.description}
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Type:</strong> {selectedAlert.type}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Risk Score:</strong> {selectedAlert.riskScore}/10
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Reporter:</strong> {selectedAlert.reporter}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Created:</strong> {selectedAlert.createdAt.toLocaleString()}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Notes & Timeline
                </Typography>
                <List>
                  {selectedAlert.notes.map((note, index) => (
                    <ListItem key={index} divider>
                      <ListItemIcon>
                        <Comment />
                      </ListItemIcon>
                      <ListItemText
                        primary={note}
                        secondary={`Added on ${selectedAlert.updatedAt.toLocaleString()}`}
                      />
                    </ListItem>
                  ))}
                  {selectedAlert.notes.length === 0 && (
                    <Typography variant="body2" color="text.secondary">
                      No notes added yet.
                    </Typography>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Quick Actions
                </Typography>
                
                <Box display="flex" flexDirection="column" gap={1}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Person />}
                    onClick={() => setCurrentTab(2)}
                  >
                    Assign Analyst
                  </Button>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<PriorityHigh />}
                    onClick={() => {
                      const newPriority = prompt('Enter new priority (low, medium, high, critical):');
                      if (newPriority) handleChangePriority(selectedAlert.id, newPriority);
                    }}
                  >
                    Change Priority
                  </Button>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Note />}
                    onClick={() => {
                      const note = prompt('Add a note:');
                      if (note) handleAddNote(selectedAlert.id, note);
                    }}
                  >
                    Add Note
                  </Button>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<CheckCircle />}
                    color="success"
                    onClick={() => {
                      const resolution = prompt('Enter resolution details:');
                      if (resolution) handleResolveAlert(selectedAlert.id, resolution);
                    }}
                  >
                    Mark Resolved
                  </Button>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<ArrowUpward />}
                    color="error"
                    onClick={() => {
                      const reason = prompt('Enter escalation reason:');
                      if (reason) handleEscalateAlert(selectedAlert.id, reason);
                    }}
                  >
                    Escalate Alert
                  </Button>
                </Box>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Alert Information
                </Typography>
                <Box mb={2}>
                  <Typography variant="body2" color="text.secondary">Status</Typography>
                  <Chip
                    label={selectedAlert.status.replace('-', ' ').toUpperCase()}
                    sx={{ 
                      bgcolor: getStatusColor(selectedAlert.status),
                      color: 'white',
                      fontWeight: 600
                    }}
                  />
                </Box>
                <Box mb={2}>
                  <Typography variant="body2" color="text.secondary">Priority</Typography>
                  <Chip
                    label={selectedAlert.priority.toUpperCase()}
                    color={selectedAlert.priority === 'critical' ? 'error' : 
                           selectedAlert.priority === 'high' ? 'warning' : 'primary'}
                  />
                </Box>
                {selectedAlert.assignedTo && (
                  <Box mb={2}>
                    <Typography variant="body2" color="text.secondary">Assigned To</Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Avatar sx={{ width: 32, height: 32 }}>
                        {selectedAlert.assignedTo.charAt(0)}
                      </Avatar>
                      <Typography variant="body2">
                        {selectedAlert.assignedTo}
                      </Typography>
                    </Box>
                  </Box>
                )}
                {selectedAlert.dueDate && (
                  <Box>
                    <Typography variant="body2" color="text.secondary">Due Date</Typography>
                    <Typography variant="body2">
                      {selectedAlert.dueDate.toLocaleDateString()}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const renderAssignAnalyst = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Assign Analyst
      </Typography>
      
      {selectedAlert && (
        <>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                Alert: {selectedAlert.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {selectedAlert.description}
              </Typography>
            </CardContent>
          </Card>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Assign to Analyst</InputLabel>
                <Select
                  value={assignmentData.analyst}
                  onChange={(e) => setAssignmentData(prev => ({ ...prev, analyst: e.target.value }))}
                >
                  <MenuItem value="Sarah Johnson">Sarah Johnson</MenuItem>
                  <MenuItem value="Mike Chen">Mike Chen</MenuItem>
                  <MenuItem value="John Smith">John Smith</MenuItem>
                  <MenuItem value="Emma Davis">Emma Davis</MenuItem>
                  <MenuItem value="Legal Team">Legal Team</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={assignmentData.priority}
                  onChange={(e) => setAssignmentData(prev => ({ ...prev, priority: e.target.value }))}
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="critical">Critical</MenuItem>
                </Select>
              </FormControl>
              
              <TextField
                fullWidth
                label="Due Date"
                type="datetime-local"
                value={assignmentData.dueDate}
                onChange={(e) => setAssignmentData(prev => ({ ...prev, dueDate: e.target.value }))}
                InputLabelProps={{ shrink: true }}
                sx={{ mb: 2 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Assignment Notes"
                multiline
                rows={8}
                value={assignmentData.notes}
                onChange={(e) => setAssignmentData(prev => ({ ...prev, notes: e.target.value }))}
                placeholder="Add any specific instructions or context for the assigned analyst..."
              />
            </Grid>
          </Grid>
          
          <Box display="flex" gap={2} mt={3}>
            <Button
              variant="contained"
              startIcon={<Assignment />}
              onClick={handleAssignAlert}
              disabled={!assignmentData.analyst}
            >
              Assign Alert
            </Button>
            <Button variant="outlined" onClick={() => setCurrentTab(0)}>
              Cancel
            </Button>
          </Box>
        </>
      )}
    </Box>
  );

  const tabLabels = ['All Alerts', 'Alert Details', 'Assign Analyst'];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '85vh',
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
          <Badge badgeContent={alerts.filter(a => a.status === 'new').length} color="error">
            <Notifications color="primary" />
          </Badge>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Alert Management System
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

      <DialogContent sx={{ p: 0, height: '65vh', overflow: 'auto' }}>
        {currentTab === 0 && renderAlertsList()}
        {currentTab === 1 && renderAlertDetails()}
        {currentTab === 2 && renderAssignAnalyst()}
      </DialogContent>

      <DialogActions sx={{ 
        p: 2, 
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
          Total alerts: {alerts.length} | New: {alerts.filter(a => a.status === 'new').length} | 
          In Progress: {alerts.filter(a => a.status === 'in-progress').length} | 
          Resolved: {alerts.filter(a => a.status === 'resolved').length}
        </Typography>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AlertManagementSystem;