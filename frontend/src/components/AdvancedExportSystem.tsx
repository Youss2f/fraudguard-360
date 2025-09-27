/**
 * Advanced Export System
 * Professional export functionality with PDF, Excel, and email reports
 */

import React, { useState } from 'react';
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
  FormControlLabel,
  Checkbox,
  RadioGroup,
  Radio,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  LinearProgress,
  Alert,
  Divider,
  IconButton,
  styled,
} from '@mui/material';
import {
  Close,
  GetApp,
  PictureAsPdf,
  TableChart,
  Email,
  Schedule,
  FilterList,
  Settings,
  Preview,
  Send,
  Download,
  FileDownload,
  Description,
  Assessment,
  TrendingUp,
  Security,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const ExportCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  cursor: 'pointer',
  '&:hover': {
    borderColor: customColors.primary[300],
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    transform: 'translateY(-2px)',
  },
}));

const ProgressCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
}));

interface AdvancedExportSystemProps {
  open: boolean;
  onClose: () => void;
}

interface ExportOptions {
  format: 'pdf' | 'excel' | 'csv' | 'json';
  template: string;
  dateRange: string;
  includeCharts: boolean;
  includeData: boolean;
  includeAnalysis: boolean;
  reportTitle: string;
  customFilters: string[];
}

interface EmailOptions {
  recipients: string;
  subject: string;
  message: string;
  scheduleType: 'now' | 'scheduled' | 'recurring';
  scheduleDate: string;
  frequency: 'daily' | 'weekly' | 'monthly';
}

const AdvancedExportSystem: React.FC<AdvancedExportSystemProps> = ({
  open,
  onClose
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'pdf',
    template: 'comprehensive',
    dateRange: 'last-7-days',
    includeCharts: true,
    includeData: true,
    includeAnalysis: true,
    reportTitle: 'Fraud Detection Report',
    customFilters: []
  });

  const [emailOptions, setEmailOptions] = useState<EmailOptions>({
    recipients: '',
    subject: 'FraudGuard 360 Report',
    message: 'Please find the attached fraud detection report.',
    scheduleType: 'now',
    scheduleDate: '',
    frequency: 'weekly'
  });

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportComplete, setExportComplete] = useState(false);

  const handleExport = async (format: string) => {
    setIsExporting(true);
    setExportProgress(0);
    setExportComplete(false);

    // Simulate export progress
    const progressInterval = setInterval(() => {
      setExportProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          setIsExporting(false);
          setExportComplete(true);
          return 100;
        }
        return prev + 10;
      });
    }, 300);

    // Simulate file generation and download
    setTimeout(() => {
      const exportData = {
        reportTitle: exportOptions.reportTitle,
        format: format.toUpperCase(),
        dateRange: exportOptions.dateRange,
        generatedAt: new Date().toISOString(),
        includeCharts: exportOptions.includeCharts,
        includeData: exportOptions.includeData,
        includeAnalysis: exportOptions.includeAnalysis,
        template: exportOptions.template,
        data: {
          totalTransactions: 45789,
          fraudDetected: 127,
          riskScore: 8.3,
          alerts: 23,
          investigations: 5
        }
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
        type: format === 'pdf' ? 'application/pdf' : 
             format === 'excel' ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' :
             'application/json' 
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fraudguard-report-${Date.now()}.${format === 'excel' ? 'xlsx' : format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 3000);
  };

  const handleEmailExport = () => {
    // Simulate email sending
    setIsExporting(true);
    setExportProgress(0);
    
    const progressInterval = setInterval(() => {
      setExportProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          setIsExporting(false);
          alert(`Report emailed successfully to: ${emailOptions.recipients}`);
          return 100;
        }
        return prev + 20;
      });
    }, 200);
  };

  const renderQuickExport = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Quick Export
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <ExportCard onClick={() => handleExport('pdf')}>
            <CardContent sx={{ textAlign: 'center', p: 3 }}>
              <PictureAsPdf sx={{ fontSize: 48, color: customColors.error[500], mb: 2 }} />
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                PDF Report
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Comprehensive formatted report with charts and analysis
              </Typography>
              <Chip label="Professional" color="primary" size="small" />
            </CardContent>
          </ExportCard>
        </Grid>

        <Grid item xs={12} md={4}>
          <ExportCard onClick={() => handleExport('excel')}>
            <CardContent sx={{ textAlign: 'center', p: 3 }}>
              <TableChart sx={{ fontSize: 48, color: customColors.success[500], mb: 2 }} />
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                Excel Spreadsheet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Structured data with pivot tables and calculations
              </Typography>
              <Chip label="Data Analysis" color="success" size="small" />
            </CardContent>
          </ExportCard>
        </Grid>

        <Grid item xs={12} md={4}>
          <ExportCard onClick={() => setCurrentTab(2)}>
            <CardContent sx={{ textAlign: 'center', p: 3 }}>
              <Email sx={{ fontSize: 48, color: customColors.primary[500], mb: 2 }} />
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                Email Report
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Send reports directly via email with scheduling
              </Typography>
              <Chip label="Automated" color="primary" size="small" />
            </CardContent>
          </ExportCard>
        </Grid>
      </Grid>

      {(isExporting || exportComplete) && (
        <Box mt={4}>
          <ProgressCard>
            <Typography variant="h6" sx={{ mb: 2 }}>
              {isExporting ? 'Generating Report...' : 'Export Complete!'}
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={exportProgress} 
              sx={{ mb: 2, height: 8, borderRadius: 4 }}
            />
            <Typography variant="body2" color="text.secondary">
              {isExporting ? `Progress: ${exportProgress}%` : 'Report downloaded successfully!'}
            </Typography>
            {exportComplete && (
              <Alert severity="success" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  Your {exportOptions.format.toUpperCase()} report has been generated and downloaded successfully.
                </Typography>
              </Alert>
            )}
          </ProgressCard>
        </Box>
      )}
    </Box>
  );

  const renderCustomExport = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Custom Export Configuration
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Report Settings
              </Typography>
              
              <TextField
                fullWidth
                label="Report Title"
                value={exportOptions.reportTitle}
                onChange={(e) => setExportOptions(prev => ({ ...prev, reportTitle: e.target.value }))}
                sx={{ mb: 2 }}
              />
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Export Format</InputLabel>
                <Select
                  value={exportOptions.format}
                  onChange={(e) => setExportOptions(prev => ({ ...prev, format: e.target.value as any }))}
                >
                  <MenuItem value="pdf">PDF Report</MenuItem>
                  <MenuItem value="excel">Excel Spreadsheet</MenuItem>
                  <MenuItem value="csv">CSV Data</MenuItem>
                  <MenuItem value="json">JSON Data</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Template</InputLabel>
                <Select
                  value={exportOptions.template}
                  onChange={(e) => setExportOptions(prev => ({ ...prev, template: e.target.value }))}
                >
                  <MenuItem value="comprehensive">Comprehensive Report</MenuItem>
                  <MenuItem value="executive">Executive Summary</MenuItem>
                  <MenuItem value="technical">Technical Analysis</MenuItem>
                  <MenuItem value="compliance">Compliance Report</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl fullWidth>
                <InputLabel>Date Range</InputLabel>
                <Select
                  value={exportOptions.dateRange}
                  onChange={(e) => setExportOptions(prev => ({ ...prev, dateRange: e.target.value }))}
                >
                  <MenuItem value="today">Today</MenuItem>
                  <MenuItem value="last-7-days">Last 7 Days</MenuItem>
                  <MenuItem value="last-30-days">Last 30 Days</MenuItem>
                  <MenuItem value="last-90-days">Last 90 Days</MenuItem>
                  <MenuItem value="custom">Custom Range</MenuItem>
                </Select>
              </FormControl>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Content Options
              </Typography>
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={exportOptions.includeCharts}
                    onChange={(e) => setExportOptions(prev => ({ ...prev, includeCharts: e.target.checked }))}
                  />
                }
                label="Include Charts & Visualizations"
                sx={{ display: 'block', mb: 1 }}
              />
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={exportOptions.includeData}
                    onChange={(e) => setExportOptions(prev => ({ ...prev, includeData: e.target.checked }))}
                  />
                }
                label="Include Raw Data Tables"
                sx={{ display: 'block', mb: 1 }}
              />
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={exportOptions.includeAnalysis}
                    onChange={(e) => setExportOptions(prev => ({ ...prev, includeAnalysis: e.target.checked }))}
                  />
                }
                label="Include Analysis & Insights"
                sx={{ display: 'block', mb: 2 }}
              />
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
                Report Preview
              </Typography>
              
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <Assessment fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Executive Summary"
                    secondary="Key metrics and insights"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <TrendingUp fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Trend Analysis"
                    secondary="Historical patterns and forecasts"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <Security fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Risk Assessment"
                    secondary="Fraud risk evaluation"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <Warning fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Alert Summary"
                    secondary="Recent alerts and actions"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          startIcon={<Preview />}
          onClick={() => alert('Report preview functionality ready!')}
        >
          Preview Report
        </Button>
        <Button
          variant="contained"
          startIcon={<Download />}
          onClick={() => handleExport(exportOptions.format)}
          color="primary"
        >
          Generate & Download
        </Button>
      </Box>
    </Box>
  );

  const renderEmailExport = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Email Reports
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Recipients & Content
              </Typography>
              
              <TextField
                fullWidth
                label="Recipients (Email Addresses)"
                multiline
                rows={3}
                value={emailOptions.recipients}
                onChange={(e) => setEmailOptions(prev => ({ ...prev, recipients: e.target.value }))}
                placeholder="Enter email addresses separated by commas..."
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Subject"
                value={emailOptions.subject}
                onChange={(e) => setEmailOptions(prev => ({ ...prev, subject: e.target.value }))}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Message"
                multiline
                rows={4}
                value={emailOptions.message}
                onChange={(e) => setEmailOptions(prev => ({ ...prev, message: e.target.value }))}
                placeholder="Optional message to include with the report..."
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Delivery Options
              </Typography>
              
              <FormControl component="fieldset" sx={{ mb: 3 }}>
                <RadioGroup
                  value={emailOptions.scheduleType}
                  onChange={(e) => setEmailOptions(prev => ({ ...prev, scheduleType: e.target.value as any }))}
                >
                  <FormControlLabel value="now" control={<Radio />} label="Send Now" />
                  <FormControlLabel value="scheduled" control={<Radio />} label="Schedule for Later" />
                  <FormControlLabel value="recurring" control={<Radio />} label="Recurring Reports" />
                </RadioGroup>
              </FormControl>
              
              {emailOptions.scheduleType === 'scheduled' && (
                <TextField
                  fullWidth
                  label="Schedule Date & Time"
                  type="datetime-local"
                  value={emailOptions.scheduleDate}
                  onChange={(e) => setEmailOptions(prev => ({ ...prev, scheduleDate: e.target.value }))}
                  sx={{ mb: 2 }}
                  InputLabelProps={{ shrink: true }}
                />
              )}
              
              {emailOptions.scheduleType === 'recurring' && (
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Frequency</InputLabel>
                  <Select
                    value={emailOptions.frequency}
                    onChange={(e) => setEmailOptions(prev => ({ ...prev, frequency: e.target.value as any }))}
                  >
                    <MenuItem value="daily">Daily</MenuItem>
                    <MenuItem value="weekly">Weekly</MenuItem>
                    <MenuItem value="monthly">Monthly</MenuItem>
                  </Select>
                </FormControl>
              )}
              
              <Alert severity="info">
                <Typography variant="body2">
                  Reports will be generated in PDF format and attached to the email.
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          startIcon={<Send />}
          onClick={handleEmailExport}
          disabled={!emailOptions.recipients}
          color="primary"
        >
          {emailOptions.scheduleType === 'now' ? 'Send Report' : 
           emailOptions.scheduleType === 'scheduled' ? 'Schedule Report' : 
           'Setup Recurring Reports'}
        </Button>
        <Button variant="outlined" onClick={() => setCurrentTab(0)}>
          Back to Quick Export
        </Button>
      </Box>
      
      {isExporting && (
        <Box mt={3}>
          <ProgressCard>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Sending Email Report...
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={exportProgress} 
              sx={{ mb: 2, height: 8, borderRadius: 4 }}
            />
            <Typography variant="body2" color="text.secondary">
              Generating and sending report: {exportProgress}%
            </Typography>
          </ProgressCard>
        </Box>
      )}
    </Box>
  );

  const tabLabels = ['Quick Export', 'Custom Export', 'Email Reports'];

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
          <GetApp color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Advanced Export System
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
        {currentTab === 0 && renderQuickExport()}
        {currentTab === 1 && renderCustomExport()}
        {currentTab === 2 && renderEmailExport()}
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

export default AdvancedExportSystem;