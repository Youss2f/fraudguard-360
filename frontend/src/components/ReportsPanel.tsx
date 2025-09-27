/**
 * Professional Reports Panel Component
 * Report generation and management interface
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
  CardActions,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  IconButton,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Checkbox,
  FormControlLabel,
  FormGroup,
  RadioGroup,
  Radio,
  LinearProgress,
  Divider,
  styled,
} from '@mui/material';
import {
  Assessment,
  Close,
  PictureAsPdf,
  InsertChart,
  TableChart,
  BarChart,
  TrendingUp,
  Download,
  Share,
  Schedule,
  PlayArrow,
  Refresh,
  Visibility,
  Edit,
  Delete,
  Add,
  DateRange,
  FilterList,
  Settings,
  Email,
} from '@mui/icons-material';
// Date picker imports removed - using standard inputs for dates
import { customColors } from '../theme/enterpriseTheme';

const ReportCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  cursor: 'pointer',
  '&:hover': {
    borderColor: customColors.primary[500],
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)',
  },
}));

interface ReportsPanelProps {
  open: boolean;
  onClose: () => void;
}

interface Report {
  id: string;
  name: string;
  type: 'fraud_summary' | 'risk_analysis' | 'compliance' | 'performance' | 'custom';
  status: 'completed' | 'running' | 'scheduled' | 'failed';
  createdDate: Date;
  lastRun: Date;
  nextRun?: Date;
  format: 'pdf' | 'excel' | 'csv' | 'json';
  recipients: string[];
  description: string;
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  category: string;
  estimatedTime: string;
}

const ReportsPanel: React.FC<ReportsPanelProps> = ({ 
  open, 
  onClose 
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [reports, setReports] = useState<Report[]>([]);
  const [reportTemplates, setReportTemplates] = useState<ReportTemplate[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [reportConfig, setReportConfig] = useState({
    name: '',
    description: '',
    dateRange: 'last_30_days',
    customStartDate: new Date(),
    customEndDate: new Date(),
    format: 'pdf',
    includeCharts: true,
    includeDetails: true,
    includeSummary: true,
    schedule: 'none',
    recipients: [] as string[],
  });
  const [generationProgress, setGenerationProgress] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    // Generate mock reports
    const mockReports: Report[] = [
      {
        id: 'rpt-001',
        name: 'Monthly Fraud Summary',
        type: 'fraud_summary',
        status: 'completed',
        createdDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
        lastRun: new Date(Date.now() - 2 * 60 * 60 * 1000),
        nextRun: new Date(Date.now() + 28 * 24 * 60 * 60 * 1000),
        format: 'pdf',
        recipients: ['sarah.johnson@company.com', 'mike.chen@company.com'],
        description: 'Comprehensive monthly fraud detection summary with key metrics and trends'
      },
      {
        id: 'rpt-002',
        name: 'Risk Analysis Report',
        type: 'risk_analysis',
        status: 'running',
        createdDate: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
        lastRun: new Date(Date.now() - 15 * 60 * 1000),
        format: 'excel',
        recipients: ['emma.davis@company.com'],
        description: 'Detailed risk assessment analysis for high-risk accounts'
      },
      {
        id: 'rpt-003',
        name: 'Compliance Audit',
        type: 'compliance',
        status: 'scheduled',
        createdDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        lastRun: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        nextRun: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        format: 'pdf',
        recipients: ['john.smith@company.com'],
        description: 'Weekly compliance audit report for regulatory requirements'
      }
    ];
    setReports(mockReports);

    // Generate report templates
    const mockTemplates: ReportTemplate[] = [
      {
        id: 'tpl-fraud-summary',
        name: 'Fraud Detection Summary',
        description: 'Overview of fraud detection activities, alerts, and resolution status',
        icon: <Assessment />,
        category: 'Fraud Analysis',
        estimatedTime: '5-10 minutes'
      },
      {
        id: 'tpl-risk-analysis',
        name: 'Risk Analysis Report',
        description: 'Comprehensive risk assessment with customer profiles and scoring',
        icon: <TrendingUp />,
        category: 'Risk Management',
        estimatedTime: '10-15 minutes'
      },
      {
        id: 'tpl-transaction-analysis',
        name: 'Transaction Analysis',
        description: 'Detailed analysis of transaction patterns and anomalies',
        icon: <BarChart />,
        category: 'Transaction Monitoring',
        estimatedTime: '3-5 minutes'
      },
      {
        id: 'tpl-compliance',
        name: 'Compliance Report',
        description: 'Regulatory compliance status and audit trail documentation',
        icon: <TableChart />,
        category: 'Compliance',
        estimatedTime: '8-12 minutes'
      },
      {
        id: 'tpl-performance',
        name: 'System Performance',
        description: 'System performance metrics and operational statistics',
        icon: <InsertChart />,
        category: 'Operations',
        estimatedTime: '2-3 minutes'
      },
      {
        id: 'tpl-investigation',
        name: 'Investigation Summary',
        description: 'Summary of ongoing investigations and case status updates',
        icon: <Visibility />,
        category: 'Investigations',
        estimatedTime: '5-8 minutes'
      }
    ];
    setReportTemplates(mockTemplates);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return customColors.success[500];
      case 'running': return customColors.primary[500];
      case 'scheduled': return customColors.warning[500];
      case 'failed': return customColors.error[500];
      default: return customColors.neutral[500];
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓';
      case 'running': return '⟳';
      case 'scheduled': return '⏰';
      case 'failed': return '✗';
      default: return '?';
    }
  };

  const handleGenerateReport = () => {
    setIsGenerating(true);
    setGenerationProgress(0);
    
    const interval = setInterval(() => {
      setGenerationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsGenerating(false);
          alert('Report generated successfully!');
          return 100;
        }
        return prev + 10;
      });
    }, 400);
  };

  const renderReportGeneration = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Generate New Report
      </Typography>
      
      {/* Template Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Select Report Template
          </Typography>
          
          <Grid container spacing={2}>
            {reportTemplates.map((template) => (
              <Grid item xs={12} sm={6} md={4} key={template.id}>
                <ReportCard 
                  onClick={() => setSelectedTemplate(template.id)}
                  sx={{ 
                    border: selectedTemplate === template.id 
                      ? `2px solid ${customColors.primary[500]}` 
                      : `1px solid ${customColors.neutral[200]}`
                  }}
                >
                  <CardContent>
                    <Box display="flex" alignItems="center" gap={2} mb={2}>
                      <Box sx={{ color: customColors.primary[500] }}>
                        {template.icon}
                      </Box>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          {template.name}
                        </Typography>
                        <Chip 
                          size="small" 
                          label={template.category}
                          variant="outlined"
                        />
                      </Box>
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {template.description}
                    </Typography>
                    
                    <Typography variant="caption" color="text.secondary">
                      Estimated time: {template.estimatedTime}
                    </Typography>
                  </CardContent>
                </ReportCard>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Report Configuration */}
      {selectedTemplate && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Report Configuration
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Report Name"
                  value={reportConfig.name}
                  onChange={(e) => setReportConfig(prev => ({ ...prev, name: e.target.value }))}
                  sx={{ mb: 2 }}
                />
                
                <TextField
                  fullWidth
                  label="Description"
                  multiline
                  rows={3}
                  value={reportConfig.description}
                  onChange={(e) => setReportConfig(prev => ({ ...prev, description: e.target.value }))}
                  sx={{ mb: 2 }}
                />
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Date Range</InputLabel>
                  <Select
                    value={reportConfig.dateRange}
                    onChange={(e) => setReportConfig(prev => ({ ...prev, dateRange: e.target.value }))}
                  >
                    <MenuItem value="last_7_days">Last 7 Days</MenuItem>
                    <MenuItem value="last_30_days">Last 30 Days</MenuItem>
                    <MenuItem value="last_90_days">Last 90 Days</MenuItem>
                    <MenuItem value="custom">Custom Range</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>Output Format</InputLabel>
                  <Select
                    value={reportConfig.format}
                    onChange={(e) => setReportConfig(prev => ({ ...prev, format: e.target.value }))}
                  >
                    <MenuItem value="pdf">PDF Document</MenuItem>
                    <MenuItem value="excel">Excel Spreadsheet</MenuItem>
                    <MenuItem value="csv">CSV Data</MenuItem>
                    <MenuItem value="json">JSON Data</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  Include Sections
                </Typography>
                
                <FormGroup sx={{ mb: 2 }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={reportConfig.includeSummary}
                        onChange={(e) => setReportConfig(prev => ({ ...prev, includeSummary: e.target.checked }))}
                      />
                    }
                    label="Executive Summary"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={reportConfig.includeCharts}
                        onChange={(e) => setReportConfig(prev => ({ ...prev, includeCharts: e.target.checked }))}
                      />
                    }
                    label="Charts and Visualizations"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={reportConfig.includeDetails}
                        onChange={(e) => setReportConfig(prev => ({ ...prev, includeDetails: e.target.checked }))}
                      />
                    }
                    label="Detailed Analysis"
                  />
                </FormGroup>
                
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  Delivery Options
                </Typography>
                
                <RadioGroup
                  value={reportConfig.schedule}
                  onChange={(e) => setReportConfig(prev => ({ ...prev, schedule: e.target.value }))}
                >
                  <FormControlLabel value="none" control={<Radio />} label="Generate Once" />
                  <FormControlLabel value="daily" control={<Radio />} label="Daily Schedule" />
                  <FormControlLabel value="weekly" control={<Radio />} label="Weekly Schedule" />
                  <FormControlLabel value="monthly" control={<Radio />} label="Monthly Schedule" />
                </RadioGroup>
              </Grid>
            </Grid>
          </CardContent>
          
          <CardActions>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={handleGenerateReport}
              disabled={isGenerating}
            >
              Generate Report
            </Button>
            <Button startIcon={<Schedule />}>
              Schedule Report
            </Button>
            <Button startIcon={<Settings />}>
              Advanced Options
            </Button>
          </CardActions>
        </Card>
      )}

      {/* Generation Progress */}
      {(isGenerating || generationProgress > 0) && (
        <Card>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
              Generating Report...
            </Typography>
            
            <LinearProgress 
              variant="determinate" 
              value={generationProgress}
              sx={{ mb: 1 }}
            />
            <Typography variant="body2" color="text.secondary">
              {generationProgress}% complete - Processing data and generating visualizations...
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );

  const renderReportHistory = () => (
    <Box p={3}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Report History
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => alert('Reports refreshed')}
        >
          Refresh
        </Button>
      </Box>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Report Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Last Run</TableCell>
              <TableCell>Next Run</TableCell>
              <TableCell>Format</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {reports.map((report) => (
              <TableRow key={report.id}>
                <TableCell>
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {report.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {report.description}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip size="small" label={report.type.replace('_', ' ')} variant="outlined" />
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography 
                      variant="body2" 
                      sx={{ color: getStatusColor(report.status) }}
                    >
                      {getStatusIcon(report.status)}
                    </Typography>
                    <Typography variant="body2">
                      {report.status}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {report.lastRun.toLocaleString()}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {report.nextRun ? report.nextRun.toLocaleString() : 'Not scheduled'}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip size="small" label={report.format.toUpperCase()} />
                </TableCell>
                <TableCell>
                  <IconButton size="small" title="View Report">
                    <Visibility />
                  </IconButton>
                  <IconButton size="small" title="Download">
                    <Download />
                  </IconButton>
                  <IconButton size="small" title="Share">
                    <Share />
                  </IconButton>
                  <IconButton size="small" title="Edit">
                    <Edit />
                  </IconButton>
                  <IconButton size="small" title="Delete" color="error">
                    <Delete />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );

  const tabLabels = [
    'Generate Report',
    'Report History'
  ];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="lg"
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
            <Assessment color="primary" />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Reports Center
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
          {currentTab === 0 && renderReportGeneration()}
          {currentTab === 1 && renderReportHistory()}
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

export default ReportsPanel;