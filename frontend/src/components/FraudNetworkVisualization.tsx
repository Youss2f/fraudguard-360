/**
 * Professional Fraud Investigation Interface
 * Excel 2010-style network visualization with enterprise features
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Grid,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  SelectChangeEvent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Avatar,
  Tabs,
  Tab,
  Alert,
  styled,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  Info,
  Search,
  AccountCircle,
  Phone,
  LocationOn,
  DeviceHub,
  Timeline,
  Warning,
  CheckCircle,
  Block,
  Flag,
  Person,
  Assignment,
  Share,
  Print,
  GetApp,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { customColors } from '../theme/enterpriseTheme';

// Professional styled components
const InvestigationCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  border: `1px solid ${customColors.neutral[200]}`,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
}));

const CaseHeader = styled(Box)(({ theme }) => ({
  backgroundColor: customColors.background.ribbon,
  padding: '16px 24px',
  borderBottom: `2px solid ${customColors.primary[500]}`,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
}));

const ActionButton = styled(Button)(({ theme }) => ({
  minWidth: '100px',
  height: '36px',
  margin: '0 4px',
  textTransform: 'none',
  fontWeight: 600,
  borderRadius: '4px',
}));

const ProfessionalInvestigation: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [currentTab, setCurrentTab] = useState(0);
  const [caseData, setCaseData] = useState<any>(null);
  const [legendOpen, setLegendOpen] = useState<boolean>(false);
  const [assignDialogOpen, setAssignDialogOpen] = useState<boolean>(false);

  // Mock case data
  const mockCaseData = {
    id: 'CASE_FR2025001',
    alertId: 'ALERT_FR2025001',
    riskScore: 0.89,
    status: 'In Investigation',
    priority: 'High',
    assignedAnalyst: 'Sarah Johnson',
    createdDate: new Date('2025-09-26T08:30:00'),
    suspect: {
      id: 'user_5678',
      name: 'John Suspicious',
      phone: '+1-555-0123',
      email: 'john.suspicious@email.com',
      accountCreated: new Date('2024-01-15'),
      verification: 'Failed',
      lastActivity: new Date('2025-09-26T07:45:00'),
      location: 'New York, NY',
    },
    fraudIndicators: [
      'High-frequency transactions',
      'Unusual location pattern',
      'Device fingerprint mismatch',
      'Velocity anomaly detected',
      'Known fraudulent network connection',
    ],
    estimatedImpact: 25000,
    transactionCount: 47,
    networkSize: 23,
  };

  useEffect(() => {
    setCaseData(mockCaseData);
  }, []);

  const handleCaseAction = (action: string) => {
    console.log(`Case action: ${action}`);
    switch (action) {
      case 'assign':
        setAssignDialogOpen(true);
        break;
      case 'escalate':
        window.alert('Case escalated to senior analyst');
        break;
      case 'close':
        if (window.confirm('Are you sure you want to close this case?')) {
          window.alert('Case closed successfully');
        }
        break;
      case 'export':
        window.alert('Exporting case data...');
        break;
    }
  };

  const renderCaseHeader = () => (
    <CaseHeader>
      <Box>
        <Typography variant="h5" sx={{ fontWeight: 700, color: customColors.neutral[900] }}>
          {caseData?.id} - Fraud Investigation
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Created: {caseData && format(caseData.createdDate, 'MMM dd, yyyy HH:mm')} | 
          Analyst: {caseData?.assignedAnalyst}
        </Typography>
      </Box>
      
      <Box display="flex" gap={1}>
        <ActionButton
          variant="contained"
          color="primary"
          startIcon={<Assignment />}
          onClick={() => handleCaseAction('assign')}
        >
          Assign
        </ActionButton>
        <ActionButton
          variant="outlined"
          color="warning"
          startIcon={<Warning />}
          onClick={() => handleCaseAction('escalate')}
        >
          Escalate
        </ActionButton>
        <ActionButton
          variant="outlined"
          color="success"
          startIcon={<CheckCircle />}
          onClick={() => handleCaseAction('close')}
        >
          Close Case
        </ActionButton>
        <ActionButton
          variant="outlined"
          startIcon={<GetApp />}
          onClick={() => handleCaseAction('export')}
        >
          Export
        </ActionButton>
      </Box>
    </CaseHeader>
  );

  const renderSuspectProfile = () => (
    <InvestigationCard>
      <CardHeader
        title="Suspect Profile"
        avatar={<Avatar sx={{ bgcolor: customColors.error[500] }}><Person /></Avatar>}
        action={
          <Chip
            label={`Risk: ${(caseData?.riskScore * 100).toFixed(0)}%`}
            color="error"
            variant="filled"
          />
        }
      />
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">Name</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>{caseData?.suspect.name}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">User ID</Typography>
            <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>{caseData?.suspect.id}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">Phone</Typography>
            <Typography variant="body1">{caseData?.suspect.phone}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">Email</Typography>
            <Typography variant="body1">{caseData?.suspect.email}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">Location</Typography>
            <Typography variant="body1">{caseData?.suspect.location}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">Verification</Typography>
            <Chip 
              label={caseData?.suspect.verification} 
              color="error" 
              size="small"
            />
          </Grid>
        </Grid>
      </CardContent>
    </InvestigationCard>
  );

  return (
    <Box sx={{ backgroundColor: customColors.background.default, minHeight: '100vh' }}>
      {renderCaseHeader()}
      
      <Box sx={{ p: 3 }}>
        {/* Alert Banner */}
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            High-Risk Fraud Case - Immediate Investigation Required
          </Typography>
          <Typography variant="body2">
            Estimated financial impact: ${caseData?.estimatedImpact.toLocaleString()} | 
            Network size: {caseData?.networkSize} entities | 
            Transaction count: {caseData?.transactionCount}
          </Typography>
        </Alert>

        <Grid container spacing={3}>
          {/* Main Investigation View */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ height: '600px', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Box textAlign="center">
                <Typography variant="h6" color="textSecondary" gutterBottom>
                  Network Visualization
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Interactive fraud network graph will be displayed here
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                  Features: Node expansion, relationship mapping, risk scoring, timeline analysis
                </Typography>
              </Box>
            </Paper>
          </Grid>

          {/* Investigation Panel */}
          <Grid item xs={12} lg={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '600px' }}>
              {/* Tabs */}
              <Paper>
                <Tabs
                  value={currentTab}
                  onChange={(_, newValue) => setCurrentTab(newValue)}
                  variant="fullWidth"
                >
                  <Tab label="Case Details" />
                  <Tab label="Evidence" />
                  <Tab label="Actions" />
                </Tabs>
              </Paper>

              {/* Tab Content */}
              <Box sx={{ flex: 1, overflow: 'auto' }}>
                {currentTab === 0 && (
                  <Box display="flex" flexDirection="column" gap={2}>
                    {renderSuspectProfile()}
                    <InvestigationCard>
                      <CardHeader
                        title="Fraud Indicators"
                        avatar={<Avatar sx={{ bgcolor: customColors.warning[500] }}><Warning /></Avatar>}
                      />
                      <CardContent>
                        <Box display="flex" flexWrap="wrap" gap={1}>
                          {caseData?.fraudIndicators.map((indicator: string, index: number) => (
                            <Chip
                              key={index}
                              label={indicator}
                              color="warning"
                              variant="outlined"
                              size="small"
                              icon={<Flag />}
                            />
                          ))}
                        </Box>
                      </CardContent>
                    </InvestigationCard>
                  </Box>
                )}
                
                {currentTab === 1 && (
                  <InvestigationCard>
                    <CardHeader
                      title="Related Transactions"
                      avatar={<Avatar sx={{ bgcolor: customColors.primary[500] }}><Timeline /></Avatar>}
                    />
                    <CardContent>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>ID</TableCell>
                              <TableCell>Amount</TableCell>
                              <TableCell>Risk</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Array.from({ length: 5 }, (_, i) => (
                              <TableRow key={i} hover>
                                <TableCell sx={{ fontFamily: 'monospace' }}>TXN_{String(i + 1).padStart(3, '0')}</TableCell>
                                <TableCell>${(Math.random() * 5000).toFixed(2)}</TableCell>
                                <TableCell>
                                  <Chip
                                    label={Math.random() > 0.5 ? 'High' : 'Low'}
                                    color={Math.random() > 0.5 ? 'error' : 'success'}
                                    size="small"
                                  />
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </InvestigationCard>
                )}
                
                {currentTab === 2 && (
                  <InvestigationCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>Investigation Actions</Typography>
                      <Box display="flex" flexDirection="column" gap={1}>
                        <Button variant="outlined" startIcon={<Block />}>
                          Block User
                        </Button>
                        <Button variant="outlined" startIcon={<Flag />}>
                          Flag as Fraudulent
                        </Button>
                        <Button variant="outlined" startIcon={<CheckCircle />}>
                          Mark as False Positive
                        </Button>
                        <Button variant="outlined" startIcon={<Share />}>
                          Share Case
                        </Button>
                        <Button variant="outlined" startIcon={<Print />}>
                          Generate Report
                        </Button>
                      </Box>
                    </CardContent>
                  </InvestigationCard>
                )}
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Assignment Dialog */}
      <Dialog open={assignDialogOpen} onClose={() => setAssignDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Assign Case</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Analyst</InputLabel>
            <Select>
              <MenuItem value="sarah">Sarah Johnson</MenuItem>
              <MenuItem value="mike">Mike Chen</MenuItem>
              <MenuItem value="anna">Anna Rodriguez</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAssignDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => setAssignDialogOpen(false)} variant="contained">Assign</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProfessionalInvestigation;