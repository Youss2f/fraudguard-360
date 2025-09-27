/**
 * Investigation Views
 * Comprehensive investigation panels for alert analysis, all alerts view, and user investigation features
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
  Chip,
  Avatar,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  styled,
} from '@mui/material';
import {
  Close,
  Search,
  Timeline,
  Person,
  Warning,
  TrendingUp,
  Assessment,
  Security,
  NetworkCheck,
  AccountBalance,
  LocationOn,
  Schedule,
  Money,
  CreditCard,
  Phone,
  Email,
  Computer,
  ExpandMore,
  Visibility,
  Flag,
  Link,
  ShowChart,
  Analytics,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const InvestigationCard = styled(Card)(({ theme }) => ({
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

const MetricCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  textAlign: 'center',
}));

const RiskIndicator = styled(Box)<{ risk: 'low' | 'medium' | 'high' | 'critical' }>(({ risk }) => ({
  width: 12,
  height: 12,
  borderRadius: '50%',
  backgroundColor: 
    risk === 'critical' ? '#d32f2f' :
    risk === 'high' ? customColors.warning[500] :
    risk === 'medium' ? customColors.primary[500] :
    customColors.success[500],
  marginRight: 8,
}));

interface InvestigationViewsProps {
  open: boolean;
  onClose: () => void;
}

interface Investigation {
  id: string;
  title: string;
  description: string;
  status: 'active' | 'completed' | 'on-hold' | 'escalated';
  priority: 'low' | 'medium' | 'high' | 'critical';
  investigator: string;
  createdAt: Date;
  updatedAt: Date;
  alertsCount: number;
  usersInvolved: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  category: 'fraud' | 'security' | 'compliance';
}

interface UserProfile {
  id: string;
  name: string;
  email: string;
  phone: string;
  accountNumber: string;
  registrationDate: Date;
  lastActivity: Date;
  riskScore: number;
  accountStatus: 'active' | 'suspended' | 'closed';
  flaggedTransactions: number;
  totalTransactions: number;
  location: string;
}

const InvestigationViews: React.FC<InvestigationViewsProps> = ({
  open,
  onClose
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedInvestigation, setSelectedInvestigation] = useState<Investigation | null>(null);
  const [selectedUser, setSelectedUser] = useState<UserProfile | null>(null);

  const [investigations] = useState<Investigation[]>([
    {
      id: 'INV-001',
      title: 'Multi-Account Transaction Ring Investigation',
      description: 'Complex fraud pattern involving 15 linked accounts conducting coordinated transactions',
      status: 'active',
      priority: 'high',
      investigator: 'Sarah Johnson',
      createdAt: new Date('2024-01-10T09:00:00'),
      updatedAt: new Date('2024-01-15T14:30:00'),
      alertsCount: 23,
      usersInvolved: 15,
      riskLevel: 'high',
      category: 'fraud'
    },
    {
      id: 'INV-002',
      title: 'Identity Theft Investigation',
      description: 'Suspected identity theft with multiple account access from different geographic locations',
      status: 'active',
      priority: 'critical',
      investigator: 'Mike Chen',
      createdAt: new Date('2024-01-12T11:30:00'),
      updatedAt: new Date('2024-01-15T16:00:00'),
      alertsCount: 8,
      usersInvolved: 3,
      riskLevel: 'critical',
      category: 'security'
    },
    {
      id: 'INV-003',
      title: 'Money Laundering Pattern Analysis',
      description: 'Structured transactions designed to avoid reporting thresholds',
      status: 'completed',
      priority: 'high',
      investigator: 'Emma Davis',
      createdAt: new Date('2024-01-05T14:00:00'),
      updatedAt: new Date('2024-01-14T10:00:00'),
      alertsCount: 45,
      usersInvolved: 7,
      riskLevel: 'high',
      category: 'compliance'
    }
  ]);

  const [userProfiles] = useState<UserProfile[]>([
    {
      id: 'USR-001',
      name: 'John Doe',
      email: 'john.doe@email.com',
      phone: '+1-555-0123',
      accountNumber: 'ACC-789123456',
      registrationDate: new Date('2023-06-15T10:00:00'),
      lastActivity: new Date('2024-01-15T14:25:00'),
      riskScore: 8.7,
      accountStatus: 'active',
      flaggedTransactions: 12,
      totalTransactions: 156,
      location: 'New York, NY'
    },
    {
      id: 'USR-002',
      name: 'Maria Rodriguez',
      email: 'maria.r@email.com',
      phone: '+1-555-0456',
      accountNumber: 'ACC-456789123',
      registrationDate: new Date('2023-03-22T14:30:00'),
      lastActivity: new Date('2024-01-15T12:15:00'),
      riskScore: 3.2,
      accountStatus: 'active',
      flaggedTransactions: 2,
      totalTransactions: 89,
      location: 'Los Angeles, CA'
    },
    {
      id: 'USR-003',
      name: 'Robert Smith',
      email: 'r.smith@email.com',
      phone: '+1-555-0789',
      accountNumber: 'ACC-123456789',
      registrationDate: new Date('2022-11-10T09:15:00'),
      lastActivity: new Date('2024-01-14T16:45:00'),
      riskScore: 9.1,
      accountStatus: 'suspended',
      flaggedTransactions: 28,
      totalTransactions: 234,
      location: 'Chicago, IL'
    }
  ]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return customColors.primary[500];
      case 'completed': return customColors.success[500];
      case 'on-hold': return customColors.warning[500];
      case 'escalated': return '#d32f2f';
      default: return customColors.neutral[500];
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return '#d32f2f';
      case 'high': return customColors.warning[500];
      case 'medium': return customColors.primary[500];
      case 'low': return customColors.success[500];
      default: return customColors.neutral[500];
    }
  };

  const renderInvestigationsList = () => (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Active Investigations
        </Typography>
        <Button variant="contained" startIcon={<Search />}>
          New Investigation
        </Button>
      </Box>

      <Grid container spacing={3}>
        {investigations.map((investigation) => (
          <Grid item xs={12} key={investigation.id}>
            <InvestigationCard onClick={() => setSelectedInvestigation(investigation)}>
              <CardContent>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} md={6}>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Assessment />
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {investigation.title}
                      </Typography>
                      <Chip label={investigation.id} size="small" variant="outlined" />
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {investigation.description}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <RiskIndicator risk={investigation.riskLevel} />
                      <Typography variant="body2">
                        Risk Level: {investigation.riskLevel.toUpperCase()}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Box mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Status:</strong>
                      </Typography>
                      <Chip
                        label={investigation.status.replace('-', ' ').toUpperCase()}
                        size="small"
                        sx={{
                          bgcolor: getStatusColor(investigation.status),
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
                        label={investigation.priority.toUpperCase()}
                        size="small"
                        sx={{
                          bgcolor: getPriorityColor(investigation.priority),
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    </Box>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Avatar sx={{ width: 24, height: 24, fontSize: '0.75rem' }}>
                        {investigation.investigator.charAt(0)}
                      </Avatar>
                      <Typography variant="body2">
                        {investigation.investigator}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={3}>
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <MetricCard>
                          <Typography variant="h6" sx={{ fontWeight: 600, color: customColors.primary[600] }}>
                            {investigation.alertsCount}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Alerts
                          </Typography>
                        </MetricCard>
                      </Grid>
                      <Grid item xs={6}>
                        <MetricCard>
                          <Typography variant="h6" sx={{ fontWeight: 600, color: customColors.warning[600] }}>
                            {investigation.usersInvolved}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Users
                          </Typography>
                        </MetricCard>
                      </Grid>
                    </Grid>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Updated: {investigation.updatedAt.toLocaleDateString()}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </InvestigationCard>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const renderInvestigationDetails = () => {
    if (!selectedInvestigation) return null;

    return (
      <Box p={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Investigation Details - {selectedInvestigation.id}
          </Typography>
          <Button onClick={() => setSelectedInvestigation(null)}>
            Back to List
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  {selectedInvestigation.title}
                </Typography>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  {selectedInvestigation.description}
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Category:</strong> {selectedInvestigation.category}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Created:</strong> {selectedInvestigation.createdAt.toLocaleDateString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Investigator:</strong> {selectedInvestigation.investigator}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Last Updated:</strong> {selectedInvestigation.updatedAt.toLocaleDateString()}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Investigation Timeline
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List>
                  <ListItem>
                    <ListItemIcon><Timeline /></ListItemIcon>
                    <ListItemText
                      primary="Investigation Initiated"
                      secondary={`${selectedInvestigation.createdAt.toLocaleString()} - Investigation opened by ${selectedInvestigation.investigator}`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Warning /></ListItemIcon>
                    <ListItemText
                      primary="Initial Alerts Processed"
                      secondary="Multiple fraud indicators detected and consolidated"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Assessment /></ListItemIcon>
                    <ListItemText
                      primary="Pattern Analysis Completed"
                      secondary="Advanced analytics identified suspicious transaction patterns"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Security /></ListItemIcon>
                    <ListItemText
                      primary="Risk Assessment Updated"
                      secondary={`Risk level elevated to ${selectedInvestigation.riskLevel.toUpperCase()}`}
                    />
                  </ListItem>
                </List>
              </AccordionDetails>
            </Accordion>

            <Accordion sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Related Entities
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Entity Type</TableCell>
                        <TableCell>Identifier</TableCell>
                        <TableCell>Risk Score</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>User Account</TableCell>
                        <TableCell>ACC-789123456</TableCell>
                        <TableCell>8.7/10</TableCell>
                        <TableCell>
                          <Chip label="Active" color="primary" size="small" />
                        </TableCell>
                        <TableCell align="right">
                          <IconButton size="small">
                            <Visibility />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Payment Method</TableCell>
                        <TableCell>****-****-****-1234</TableCell>
                        <TableCell>6.2/10</TableCell>
                        <TableCell>
                          <Chip label="Flagged" color="warning" size="small" />
                        </TableCell>
                        <TableCell align="right">
                          <IconButton size="small">
                            <Visibility />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>IP Address</TableCell>
                        <TableCell>192.168.1.***</TableCell>
                        <TableCell>9.1/10</TableCell>
                        <TableCell>
                          <Chip label="Blocked" color="error" size="small" />
                        </TableCell>
                        <TableCell align="right">
                          <IconButton size="small">
                            <Visibility />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </AccordionDetails>
            </Accordion>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Investigation Metrics
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <MetricCard>
                      <Typography variant="h4" sx={{ fontWeight: 600, color: customColors.primary[600] }}>
                        {selectedInvestigation.alertsCount}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Alerts
                      </Typography>
                    </MetricCard>
                  </Grid>
                  <Grid item xs={6}>
                    <MetricCard>
                      <Typography variant="h4" sx={{ fontWeight: 600, color: customColors.warning[600] }}>
                        {selectedInvestigation.usersInvolved}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Users Involved
                      </Typography>
                    </MetricCard>
                  </Grid>
                </Grid>
                
                <Box mt={2}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Investigation Progress
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={selectedInvestigation.status === 'completed' ? 100 : 
                           selectedInvestigation.status === 'active' ? 65 : 30} 
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {selectedInvestigation.status === 'completed' ? '100%' : 
                     selectedInvestigation.status === 'active' ? '65%' : '30%'} Complete
                  </Typography>
                </Box>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Quick Actions
                </Typography>
                <Box display="flex" flexDirection="column" gap={1}>
                  <Button fullWidth variant="outlined" startIcon={<Analytics />}>
                    Generate Report
                  </Button>
                  <Button fullWidth variant="outlined" startIcon={<Timeline />}>
                    View Timeline
                  </Button>
                  <Button fullWidth variant="outlined" startIcon={<ShowChart />}>
                    Network Analysis
                  </Button>
                  <Button fullWidth variant="outlined" startIcon={<Flag />}>
                    Add Flag
                  </Button>
                  <Button fullWidth variant="contained" startIcon={<Security />} color="error">
                    Escalate Investigation
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const renderUserInvestigation = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        User Investigation Dashboard
      </Typography>

      <Grid container spacing={3}>
        {userProfiles.map((user) => (
          <Grid item xs={12} md={6} lg={4} key={user.id}>
            <InvestigationCard onClick={() => setSelectedUser(user)}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                  <Avatar sx={{ width: 40, height: 40 }}>
                    {user.name.charAt(0)}
                  </Avatar>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {user.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {user.accountNumber}
                    </Typography>
                  </Box>
                </Box>

                <Box mb={2}>
                  <Box display="flex" justifyContent="space-between" mb={1}>
                    <Typography variant="body2">Risk Score</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {user.riskScore}/10
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={user.riskScore * 10}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: user.riskScore >= 7 ? '#d32f2f' : 
                                        user.riskScore >= 5 ? customColors.warning[500] : 
                                        customColors.success[500]
                      }
                    }}
                  />
                </Box>

                <Grid container spacing={1} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Status:</strong>
                    </Typography>
                    <Chip
                      label={user.accountStatus.toUpperCase()}
                      size="small"
                      color={user.accountStatus === 'active' ? 'success' : 
                             user.accountStatus === 'suspended' ? 'warning' : 'error'}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Flagged:</strong>
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {user.flaggedTransactions}/{user.totalTransactions}
                    </Typography>
                  </Grid>
                </Grid>

                <Divider sx={{ my: 1 }} />

                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Email fontSize="small" />
                  <Typography variant="body2">{user.email}</Typography>
                </Box>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Phone fontSize="small" />
                  <Typography variant="body2">{user.phone}</Typography>
                </Box>
                <Box display="flex" alignItems="center" gap={1}>
                  <LocationOn fontSize="small" />
                  <Typography variant="body2">{user.location}</Typography>
                </Box>
              </CardContent>
            </InvestigationCard>
          </Grid>
        ))}
      </Grid>

      {selectedUser && (
        <Dialog open={!!selectedUser} onClose={() => setSelectedUser(null)} maxWidth="md" fullWidth>
          <DialogTitle>
            User Investigation - {selectedUser.name}
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Account Information
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Account Number:</strong> {selectedUser.accountNumber}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Registration:</strong> {selectedUser.registrationDate.toLocaleDateString()}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Last Activity:</strong> {selectedUser.lastActivity.toLocaleString()}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Risk Score:</strong> {selectedUser.riskScore}/10
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Transaction Summary
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Total Transactions:</strong> {selectedUser.totalTransactions}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Flagged Transactions:</strong> {selectedUser.flaggedTransactions}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Flag Rate:</strong> {((selectedUser.flaggedTransactions / selectedUser.totalTransactions) * 100).toFixed(1)}%
                </Typography>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSelectedUser(null)}>Close</Button>
            <Button variant="contained">Full Investigation</Button>
          </DialogActions>
        </Dialog>
      )}
    </Box>
  );

  const tabLabels = ['Active Investigations', 'Investigation Details', 'User Investigation'];

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
          <Search color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Investigation Views
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
        {currentTab === 0 && renderInvestigationsList()}
        {currentTab === 1 && renderInvestigationDetails()}
        {currentTab === 2 && renderUserInvestigation()}
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

export default InvestigationViews;