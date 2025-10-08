/**
 * Excel-Style Reports Dashboard
 * Professional reporting interface with export capabilities
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,

  styled,
  alpha,
  Divider,
} from '@mui/material';
import {
  Report,
  Download,
  Share,
  Schedule,
  Assessment,
  Security,
  TrendingUp,
  PictureAsPdf,
  TableChart,
  BarChart,
  Print,
  Email,
  MoreVert,
  Add,
  Visibility,
  Edit,
  Delete,
  CheckCircle,
  Warning,
  Error,
  Info,
} from '@mui/icons-material';
import { excelColors } from '../theme/excelTheme';

// Styled components
const ReportsCard = styled(Card)({
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  '&:hover': {
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
  },
  transition: 'box-shadow 0.2s ease',
});

const ReportTemplateCard = styled(Paper)({
  padding: '20px',
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  '&:hover': {
    borderColor: excelColors.primary.main,
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
  },
});

// Mock data
const reportTemplates = [
  {
    id: 1,
    name: 'Executive Summary',
    description: 'High-level fraud metrics and KPIs for executive review',
    icon: Assessment,
    category: 'Executive',
    frequency: 'Monthly',
    format: ['PDF', 'Excel'],
    lastGenerated: '2024-09-28',
    status: 'active',
  },
  {
    id: 2,
    name: 'Fraud Detection Report',
    description: 'Detailed analysis of detected fraud cases and patterns',
    icon: Security,
    category: 'Operations',
    frequency: 'Weekly',
    format: ['PDF', 'CSV'],
    lastGenerated: '2024-09-27',
    status: 'active',
  },
  {
    id: 3,
    name: 'Risk Assessment',
    description: 'Comprehensive risk analysis and scoring trends',
    icon: Warning,
    category: 'Risk',
    frequency: 'Daily',
    format: ['Excel', 'JSON'],
    lastGenerated: '2024-09-28',
    status: 'active',
  },
  {
    id: 4,
    name: 'Investigation Report',
    description: 'Case-by-case investigation details and outcomes',
    icon: Report,
    category: 'Investigation',
    frequency: 'On-demand',
    format: ['PDF', 'Word'],
    lastGenerated: '2024-09-26',
    status: 'draft',
  },
];

const scheduledReports = [
  {
    id: 1,
    name: 'Daily Fraud Summary',
    schedule: 'Daily at 8:00 AM',
    recipients: ['admin@company.com', 'security@company.com'],
    format: 'PDF',
    nextRun: '2024-09-29 08:00',
    status: 'active',
  },
  {
    id: 2,
    name: 'Weekly Executive Report',
    schedule: 'Weekly on Monday at 9:00 AM',
    recipients: ['ceo@company.com', 'cfo@company.com'],
    format: 'PDF + Excel',
    nextRun: '2024-09-30 09:00',
    status: 'active',
  },
  {
    id: 3,
    name: 'Monthly Compliance Report',
    schedule: 'Monthly on 1st at 10:00 AM',
    recipients: ['compliance@company.com'],
    format: 'PDF',
    nextRun: '2024-10-01 10:00',
    status: 'paused',
  },
];

const recentReports = [
  {
    id: 1,
    name: 'September 2024 Executive Summary',
    type: 'Executive Summary',
    generatedBy: 'System Auto',
    generatedAt: '2024-09-28 08:30:00',
    fileSize: '2.4 MB',
    format: 'PDF',
    downloadCount: 15,
    status: 'completed',
  },
  {
    id: 2,
    name: 'Fraud Detection Analysis - Week 39',
    type: 'Fraud Detection Report',
    generatedBy: 'Sarah Johnson',
    generatedAt: '2024-09-27 14:22:00',
    fileSize: '1.8 MB',
    format: 'Excel',
    downloadCount: 8,
    status: 'completed',
  },
  {
    id: 3,
    name: 'Risk Assessment Dashboard',
    type: 'Risk Assessment',
    generatedBy: 'Mike Chen',
    generatedAt: '2024-09-26 16:45:00',
    fileSize: '956 KB',
    format: 'PDF',
    downloadCount: 12,
    status: 'completed',
  },
];

const ExcelReports: React.FC = () => {
  const [selectedTemplate, setSelectedTemplate] = useState<number | null>(null);
  const [reportName, setReportName] = useState('');
  const [reportFormat, setReportFormat] = useState('PDF');
  const [dateRange, setDateRange] = useState('last-30-days');

  const getStatusChip = (status: string) => {
    const configs = {
      active: { label: 'Active', color: excelColors.success.main, icon: CheckCircle },
      draft: { label: 'Draft', color: excelColors.warning.main, icon: Warning },
      paused: { label: 'Paused', color: excelColors.text.secondary, icon: Error },
      completed: { label: 'Completed', color: excelColors.success.main, icon: CheckCircle },
      processing: { label: 'Processing', color: excelColors.info.main, icon: Info },
    };
    
    const config = configs[status as keyof typeof configs] || configs.draft;
    
    return (
      <Chip
        label={config.label}
        size="small"
        icon={<config.icon sx={{ fontSize: '16px !important' }} />}
        sx={{
          backgroundColor: alpha(config.color, 0.1),
          color: config.color,
          border: `1px solid ${config.color}`,
          fontSize: '0.75rem',
        }}
      />
    );
  };

  const getFormatIcon = (format: string) => {
    switch (format.toLowerCase()) {
      case 'pdf':
        return <PictureAsPdf sx={{ color: excelColors.error.main }} />;
      case 'excel':
      case 'xlsx':
        return <TableChart sx={{ color: excelColors.success.main }} />;
      case 'csv':
        return <TableChart sx={{ color: excelColors.accent.blue }} />;
      case 'word':
        return <Report sx={{ color: excelColors.accent.blue }} />;
      default:
        return <Report sx={{ color: excelColors.text.secondary }} />;
    }
  };

  const handleGenerateReport = (templateId: number) => {
    console.log('Generating report for template:', templateId);
    // Implementation would trigger report generation
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
          Reports & Analytics
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            startIcon={<Add />}
            variant="contained"
            size="small"
            onClick={() => window.alert('Create New Report:\n- Custom Fraud Report\n- Transaction Analysis\n- Risk Assessment Report\n- Compliance Report\n- Executive Summary')}
            sx={{ backgroundColor: excelColors.primary.main }}
          >
            New Report
          </Button>
          <Button
            startIcon={<Schedule />}
            variant="outlined"
            size="small"
            onClick={() => window.alert('Schedule Reports:\n- Daily Fraud Summary\n- Weekly Risk Assessment\n- Monthly Executive Report\n- Automated Alerts\n- Custom Schedule Setup')}
            sx={{ borderColor: excelColors.background.border }}
          >
            Schedule
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
        {/* Report Templates */}
        <Box sx={{ flex: '2 1 600px', minWidth: 600 }}>
          <ReportsCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, color: excelColors.text.primary }}>
                Report Templates
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                {reportTemplates.map((template) => (
                  <Box key={template.id} sx={{ flex: '1 1 280px', minWidth: 280 }}>
                    <ReportTemplateCard
                      onClick={() => setSelectedTemplate(template.id)}
                      sx={{
                        borderColor: selectedTemplate === template.id ? excelColors.primary.main : excelColors.background.border,
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                        <template.icon sx={{ color: excelColors.primary.main, mr: 2, fontSize: 32 }} />
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                            {template.name}
                          </Typography>
                          <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                            {template.description}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            <Chip label={template.category} size="small" variant="outlined" />
                            <Chip label={template.frequency} size="small" variant="outlined" />
                            {getStatusChip(template.status)}
                          </Box>
                        </Box>
                      </Box>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {template.format.map((format) => (
                            <Box key={format} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              {getFormatIcon(format)}
                              <Typography variant="caption">{format}</Typography>
                            </Box>
                          ))}
                        </Box>
                        <Button
                          size="small"
                          variant="contained"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleGenerateReport(template.id);
                          }}
                          sx={{ backgroundColor: excelColors.primary.main }}
                        >
                          Generate
                        </Button>
                      </Box>
                    </ReportTemplateCard>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </ReportsCard>

          {/* Recent Reports */}
          <Box sx={{ mt: 3 }}>
            <ReportsCard>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
                    Recent Reports
                  </Typography>
                  <Button 
                    size="small" 
                    startIcon={<Visibility />}
                    onClick={() => window.alert('View All Reports:\n- Fraud Detection Reports\n- Risk Assessment Reports\n- Compliance Reports\n- Executive Summaries\n- Custom Reports\n- Archived Reports')}
                  >
                    View All
                  </Button>
                </Box>
                
                <List>
                  {recentReports.map((report, index) => (
                    <React.Fragment key={report.id}>
                      <ListItem>
                        <ListItemIcon>
                          {getFormatIcon(report.format)}
                        </ListItemIcon>
                        <ListItemText
                          primary={report.name}
                          secondary={
                            <Box>
                              <Typography variant="body2" color="textSecondary">
                                Generated by {report.generatedBy} • {new Date(report.generatedAt).toLocaleString()}
                              </Typography>
                              <Typography variant="caption" color="textSecondary">
                                {report.fileSize} • Downloaded {report.downloadCount} times
                              </Typography>
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getStatusChip(report.status)}
                            <IconButton 
                              size="small"
                              onClick={() => window.alert(`Downloading report: ${report.name}\nFormat: ${report.format}\nSize: ${report.fileSize}`)}
                            >
                              <Download fontSize="small" />
                            </IconButton>
                            <IconButton 
                              size="small"
                              onClick={() => window.alert(`Share report: ${report.name}\n- Email to team\n- Generate public link\n- Export to shared drive\n- Schedule automatic sharing`)}
                            >
                              <Share fontSize="small" />
                            </IconButton>
                            <IconButton 
                              size="small"
                              onClick={() => window.alert(`Report Actions for: ${report.name}\n- View Details\n- Duplicate Report\n- Edit Template\n- Delete Report\n- Archive Report`)}
                            >
                              <MoreVert fontSize="small" />
                            </IconButton>
                          </Box>
                        </ListItemSecondaryAction>
                      </ListItem>
                      {index < recentReports.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </ReportsCard>
          </Box>
        </Box>

        {/* Sidebar */}
        <Box sx={{ flex: '1 1 300px', minWidth: 300 }}>
          {/* Quick Actions */}
          <ReportsCard sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, color: excelColors.text.primary }}>
                Quick Actions
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<Assessment />}
                  onClick={() => window.alert('Executive Dashboard Report:\n- High-level KPIs\n- Fraud trend summary\n- Financial impact analysis\n- Risk assessment overview\n- Regulatory compliance status')}
                  sx={{ justifyContent: 'flex-start', borderColor: excelColors.background.border }}
                >
                  Executive Dashboard
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<Security />}
                  onClick={() => window.alert('Security Report:\n- Security incidents\n- Threat analysis\n- Vulnerability assessment\n- Compliance audit results\n- Security recommendations')}
                  sx={{ justifyContent: 'flex-start', borderColor: excelColors.background.border }}
                >
                  Security Report
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<TrendingUp />}
                  onClick={() => window.alert('Trend Analysis Report:\n- Fraud pattern analysis\n- Seasonal trends\n- Geographic distribution\n- Time-based analysis\n- Predictive insights')}
                  sx={{ justifyContent: 'flex-start', borderColor: excelColors.background.border }}
                >
                  Trend Analysis
                </Button>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<BarChart />}
                  onClick={() => window.alert('Custom Report Builder:\n- Select data sources\n- Choose visualization types\n- Set filters and parameters\n- Schedule delivery\n- Export options')}
                  sx={{ justifyContent: 'flex-start', borderColor: excelColors.background.border }}
                >
                  Custom Report
                </Button>
              </Box>
            </CardContent>
          </ReportsCard>

          {/* Scheduled Reports */}
          <ReportsCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, color: excelColors.text.primary }}>
                Scheduled Reports
              </Typography>
              
              <List dense>
                {scheduledReports.map((report, index) => (
                  <React.Fragment key={report.id}>
                    <ListItem>
                      <ListItemIcon>
                        <Schedule sx={{ color: excelColors.accent.blue }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={report.name}
                        secondary={
                          <Box>
                            <Typography variant="caption" color="textSecondary">
                              {report.schedule}
                            </Typography>
                            <br />
                            <Typography variant="caption" color="textSecondary">
                              Next: {new Date(report.nextRun).toLocaleString()}
                            </Typography>
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        {getStatusChip(report.status)}
                      </ListItemSecondaryAction>
                    </ListItem>
                    {index < scheduledReports.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
              
              <Box sx={{ mt: 2 }}>
                <Button
                  fullWidth
                  size="small"
                  startIcon={<Add />}
                  variant="outlined"
                  sx={{ borderColor: excelColors.background.border }}
                >
                  Add Schedule
                </Button>
              </Box>
            </CardContent>
          </ReportsCard>
        </Box>
      </Box>
    </Box>
  );
};

export default ExcelReports;