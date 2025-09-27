/**
 * Comprehensive Case Management Panel
 * Professional case management with escalate, close, export, and share functionality
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
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Divider,
  Avatar,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Checkbox,
  RadioGroup,
  Radio,
  Slider,
  styled,
} from '@mui/material';
import {
  Close,
  Assignment,
  ArrowUpward,
  CheckCircle,
  GetApp,
  Share,
  Person,
  Schedule,
  PriorityHigh,
  Security,
  Description,
  Attachment,
  Comment,
  History,
  Email,
  Phone,
  LocationOn,
  Warning,
  Error,
  Info,
  ExpandMore,
  Save,
  Send,
  Print,
  FileCopy,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const CaseCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  '&:hover': {
    borderColor: customColors.primary[300],
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
  },
}));

const PriorityChip = styled(Chip)<{ priority: 'low' | 'medium' | 'high' | 'critical' }>(({ priority }) => ({
  backgroundColor: priority === 'critical' 
    ? '#ffebee' 
    : priority === 'high' 
    ? customColors.warning[50]
    : priority === 'medium'
    ? customColors.primary[50]
    : customColors.neutral[50],
  color: priority === 'critical' 
    ? '#d32f2f' 
    : priority === 'high' 
    ? customColors.warning[700]
    : priority === 'medium'
    ? customColors.primary[700]
    : customColors.neutral[700],
}));

interface CaseManagementPanelProps {
  open: boolean;
  onClose: () => void;
  caseId?: string;
}

interface CaseData {
  id: string;
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'open' | 'in-progress' | 'escalated' | 'resolved' | 'closed';
  assignedTo: string;
  reporter: string;
  createdDate: Date;
  lastUpdated: Date;
  dueDate: Date;
  tags: string[];
  attachments: any[];
  comments: any[];
  timeline: any[];
}

const CaseManagementPanel: React.FC<CaseManagementPanelProps> = ({
  open,
  onClose,
  caseId = 'CASE-001'
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [caseData, setCaseData] = useState<CaseData>({
    id: caseId,
    title: 'Suspicious Transaction Pattern',
    description: 'Multiple high-value transactions detected from same IP address within short timeframe',
    priority: 'high',
    status: 'in-progress',
    assignedTo: 'Sarah Johnson',
    reporter: 'System Alert',
    createdDate: new Date('2024-01-15T10:30:00'),
    lastUpdated: new Date(),
    dueDate: new Date('2024-01-18T17:00:00'),
    tags: ['fraud', 'transaction', 'pattern-analysis'],
    attachments: [],
    comments: [],
    timeline: []
  });

  const [escalationData, setEscalationData] = useState({
    reason: '',
    urgency: 'medium',
    department: 'security',
    additionalInfo: ''
  });

  const [closeData, setCloseData] = useState({
    resolution: '',
    outcome: 'resolved',
    followUpRequired: false,
    notes: ''
  });

  const [shareData, setShareData] = useState({
    recipients: '',
    accessLevel: 'view',
    expirationDays: 7,
    includeAttachments: true,
    message: ''
  });

  const handleEscalateCase = () => {
    setCaseData(prev => ({
      ...prev,
      status: 'escalated',
      priority: prev.priority === 'critical' ? 'critical' : 'high',
      lastUpdated: new Date()
    }));
    
    alert(`Case ${caseData.id} escalated successfully!\n\nEscalation Details:\n• Reason: ${escalationData.reason}\n• Department: ${escalationData.department}\n• Urgency: ${escalationData.urgency}`);
  };

  const handleCloseCase = () => {
    setCaseData(prev => ({
      ...prev,
      status: 'closed',
      lastUpdated: new Date()
    }));
    
    alert(`Case ${caseData.id} closed successfully!\n\nResolution: ${closeData.resolution}\nOutcome: ${closeData.outcome}`);
  };

  const handleExportCase = (format: 'pdf' | 'excel' | 'json') => {
    const exportData = {
      case: caseData,
      exportDate: new Date().toISOString(),
      format: format.toUpperCase()
    };
    
    // In a real app, this would generate the actual file
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `case-${caseData.id}-export.${format === 'excel' ? 'xlsx' : format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    alert(`Case ${caseData.id} exported as ${format.toUpperCase()} successfully!`);
  };

  const handleShareCase = () => {
    // In a real app, this would create secure sharing links
    const shareLink = `https://fraudguard360.com/cases/${caseData.id}/shared/${Date.now()}`;
    
    navigator.clipboard.writeText(shareLink).then(() => {
      alert(`Case ${caseData.id} shared successfully!\n\nShare link copied to clipboard:\n${shareLink}\n\nAccess Level: ${shareData.accessLevel}\nExpires: ${shareData.expirationDays} days`);
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return customColors.neutral[600];
      case 'in-progress': return customColors.primary[600];
      case 'escalated': return customColors.warning[600];
      case 'resolved': return customColors.success[600];
      case 'closed': return customColors.neutral[500];
      default: return customColors.neutral[600];
    }
  };

  const renderCaseOverview = () => (
    <Box p={3}>
      <Grid container spacing={3}>
        {/* Case Header */}
        <Grid item xs={12}>
          <CaseCard>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
                    {caseData.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Case ID: {caseData.id}
                  </Typography>
                </Box>
                <Box display="flex" gap={1}>
                  <PriorityChip 
                    label={caseData.priority.toUpperCase()} 
                    priority={caseData.priority}
                    size="small"
                  />
                  <Chip 
                    label={caseData.status.replace('-', ' ').toUpperCase()}
                    sx={{ bgcolor: getStatusColor(caseData.status), color: 'white' }}
                    size="small"
                  />
                </Box>
              </Box>
              
              <Typography variant="body1" sx={{ mb: 2 }}>
                {caseData.description}
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Person fontSize="small" />
                    <Typography variant="body2">
                      <strong>Assigned:</strong> {caseData.assignedTo}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Schedule fontSize="small" />
                    <Typography variant="body2">
                      <strong>Created:</strong> {caseData.createdDate.toLocaleDateString()}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1}>
                    <PriorityHigh fontSize="small" />
                    <Typography variant="body2">
                      <strong>Due:</strong> {caseData.dueDate.toLocaleDateString()}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Security fontSize="small" />
                    <Typography variant="body2">
                      <strong>Reporter:</strong> {caseData.reporter}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <History fontSize="small" />
                    <Typography variant="body2">
                      <strong>Updated:</strong> {caseData.lastUpdated.toLocaleString()}
                    </Typography>
                  </Box>
                  <Box display="flex" flexWrap="wrap" gap={0.5}>
                    {caseData.tags.map((tag, index) => (
                      <Chip key={index} label={tag} size="small" variant="outlined" />
                    ))}
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </CaseCard>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <CaseCard>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Case Actions
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<ArrowUpward />}
                    onClick={() => setCurrentTab(1)}
                    disabled={caseData.status === 'closed'}
                  >
                    Escalate
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<CheckCircle />}
                    onClick={() => setCurrentTab(2)}
                    color="success"
                    disabled={caseData.status === 'closed'}
                  >
                    Close Case
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<GetApp />}
                    onClick={() => setCurrentTab(3)}
                  >
                    Export
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Share />}
                    onClick={() => setCurrentTab(4)}
                  >
                    Share Case
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </CaseCard>
        </Grid>
      </Grid>
    </Box>
  );

  const renderEscalation = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Escalate Case
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Escalation Reason"
            multiline
            rows={4}
            value={escalationData.reason}
            onChange={(e) => setEscalationData(prev => ({ ...prev, reason: e.target.value }))}
            placeholder="Explain why this case needs to be escalated..."
            sx={{ mb: 2 }}
          />
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Urgency Level</InputLabel>
            <Select
              value={escalationData.urgency}
              onChange={(e) => setEscalationData(prev => ({ ...prev, urgency: e.target.value }))}
            >
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Department</InputLabel>
            <Select
              value={escalationData.department}
              onChange={(e) => setEscalationData(prev => ({ ...prev, department: e.target.value }))}
            >
              <MenuItem value="security">Security Team</MenuItem>
              <MenuItem value="management">Management</MenuItem>
              <MenuItem value="legal">Legal Department</MenuItem>
              <MenuItem value="compliance">Compliance</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Additional Information"
            multiline
            rows={8}
            value={escalationData.additionalInfo}
            onChange={(e) => setEscalationData(prev => ({ ...prev, additionalInfo: e.target.value }))}
            placeholder="Any additional context or information..."
          />
        </Grid>
      </Grid>
      
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          startIcon={<ArrowUpward />}
          onClick={handleEscalateCase}
          disabled={!escalationData.reason}
        >
          Escalate Case
        </Button>
        <Button variant="outlined" onClick={() => setCurrentTab(0)}>
          Cancel
        </Button>
      </Box>
    </Box>
  );

  const renderCloseCase = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Close Case
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Resolution Summary"
            multiline
            rows={4}
            value={closeData.resolution}
            onChange={(e) => setCloseData(prev => ({ ...prev, resolution: e.target.value }))}
            placeholder="Describe how the case was resolved..."
            sx={{ mb: 2 }}
          />
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Outcome</InputLabel>
            <Select
              value={closeData.outcome}
              onChange={(e) => setCloseData(prev => ({ ...prev, outcome: e.target.value }))}
            >
              <MenuItem value="resolved">Resolved</MenuItem>
              <MenuItem value="false-positive">False Positive</MenuItem>
              <MenuItem value="no-action-needed">No Action Needed</MenuItem>
              <MenuItem value="referred">Referred to Another Team</MenuItem>
            </Select>
          </FormControl>
          
          <FormControlLabel
            control={
              <Checkbox
                checked={closeData.followUpRequired}
                onChange={(e) => setCloseData(prev => ({ ...prev, followUpRequired: e.target.checked }))}
              />
            }
            label="Follow-up Required"
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Closing Notes"
            multiline
            rows={8}
            value={closeData.notes}
            onChange={(e) => setCloseData(prev => ({ ...prev, notes: e.target.value }))}
            placeholder="Any additional notes or recommendations..."
          />
        </Grid>
      </Grid>
      
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          startIcon={<CheckCircle />}
          onClick={handleCloseCase}
          color="success"
          disabled={!closeData.resolution}
        >
          Close Case
        </Button>
        <Button variant="outlined" onClick={() => setCurrentTab(0)}>
          Cancel
        </Button>
      </Box>
    </Box>
  );

  const renderExport = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Export Case Data
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<Print />}
            onClick={() => handleExportCase('pdf')}
            sx={{ mb: 2, height: '120px', flexDirection: 'column' }}
          >
            <Typography variant="h6">PDF Report</Typography>
            <Typography variant="body2" color="text.secondary">
              Professional case report with formatting
            </Typography>
          </Button>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<FileCopy />}
            onClick={() => handleExportCase('excel')}
            sx={{ mb: 2, height: '120px', flexDirection: 'column' }}
          >
            <Typography variant="h6">Excel Spreadsheet</Typography>
            <Typography variant="body2" color="text.secondary">
              Structured data for analysis
            </Typography>
          </Button>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<GetApp />}
            onClick={() => handleExportCase('json')}
            sx={{ mb: 2, height: '120px', flexDirection: 'column' }}
          >
            <Typography variant="h6">JSON Data</Typography>
            <Typography variant="body2" color="text.secondary">
              Raw data for system integration
            </Typography>
          </Button>
        </Grid>
      </Grid>
    </Box>
  );

  const renderShare = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Share Case
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Recipients (Email Addresses)"
            multiline
            rows={3}
            value={shareData.recipients}
            onChange={(e) => setShareData(prev => ({ ...prev, recipients: e.target.value }))}
            placeholder="Enter email addresses separated by commas..."
            sx={{ mb: 2 }}
          />
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Access Level</InputLabel>
            <Select
              value={shareData.accessLevel}
              onChange={(e) => setShareData(prev => ({ ...prev, accessLevel: e.target.value }))}
            >
              <MenuItem value="view">View Only</MenuItem>
              <MenuItem value="comment">View & Comment</MenuItem>
              <MenuItem value="edit">Full Edit Access</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            fullWidth
            label="Expiration (Days)"
            type="number"
            value={shareData.expirationDays}
            onChange={(e) => setShareData(prev => ({ ...prev, expirationDays: parseInt(e.target.value) }))}
            sx={{ mb: 2 }}
          />
          
          <FormControlLabel
            control={
              <Checkbox
                checked={shareData.includeAttachments}
                onChange={(e) => setShareData(prev => ({ ...prev, includeAttachments: e.target.checked }))}
              />
            }
            label="Include Attachments"
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Message to Recipients"
            multiline
            rows={8}
            value={shareData.message}
            onChange={(e) => setShareData(prev => ({ ...prev, message: e.target.value }))}
            placeholder="Optional message to include with the shared case..."
          />
        </Grid>
      </Grid>
      
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          startIcon={<Share />}
          onClick={handleShareCase}
          disabled={!shareData.recipients}
        >
          Share Case
        </Button>
        <Button variant="outlined" onClick={() => setCurrentTab(0)}>
          Cancel
        </Button>
      </Box>
    </Box>
  );

  const tabLabels = ['Overview', 'Escalate', 'Close Case', 'Export', 'Share'];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '80vh',
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
          <Assignment color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Case Management - {caseData.id}
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

      <DialogContent sx={{ p: 0, height: '60vh', overflow: 'auto' }}>
        {currentTab === 0 && renderCaseOverview()}
        {currentTab === 1 && renderEscalation()}
        {currentTab === 2 && renderCloseCase()}
        {currentTab === 3 && renderExport()}
        {currentTab === 4 && renderShare()}
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

export default CaseManagementPanel;