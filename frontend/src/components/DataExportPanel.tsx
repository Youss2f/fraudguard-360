/**
 * Professional Data Export Component
 * Multi-format data export with customizable options
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Checkbox,
  TextField,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  styled,
} from '@mui/material';
import {
  GetApp,
  Close,
  InsertDriveFile,
  TableChart,
  PictureAsPdf,
  Description,
  Code,
  Schedule,
  CheckCircle,
  Error,
  Warning,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const ExportCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  '&:hover': {
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    transform: 'translateY(-2px)',
  },
  '&.selected': {
    border: `2px solid ${customColors.primary[500]}`,
    backgroundColor: customColors.primary[50],
  },
}));

interface DataExportProps {
  open: boolean;
  onClose: () => void;
  data?: any;
  exportType?: 'dashboard' | 'alerts' | 'transactions' | 'users' | 'custom';
}

interface ExportState {
  format: 'csv' | 'json' | 'excel' | 'pdf' | 'xml';
  includeFields: string[];
  dateRange: {
    start: string;
    end: string;
  };
  filters: {
    status: string[];
    severity: string[];
    assigned: boolean;
  };
  options: {
    includeHeaders: boolean;
    includeMetadata: boolean;
    compressFile: boolean;
    encryptFile: boolean;
    password: string;
  };
  fileName: string;
}

const DataExportPanel: React.FC<DataExportProps> = ({ 
  open, 
  onClose,
  data,
  exportType = 'dashboard'
}) => {
  const [exportState, setExportState] = useState<ExportState>({
    format: 'csv',
    includeFields: ['id', 'timestamp', 'type', 'severity', 'status'],
    dateRange: {
      start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      end: new Date().toISOString().split('T')[0],
    },
    filters: {
      status: ['new', 'investigating'],
      severity: ['critical', 'high'],
      assigned: false,
    },
    options: {
      includeHeaders: true,
      includeMetadata: true,
      compressFile: false,
      encryptFile: false,
      password: '',
    },
    fileName: `fraudguard_${exportType}_${new Date().toISOString().split('T')[0]}`,
  });

  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportComplete, setExportComplete] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const formatOptions = [
    {
      id: 'csv',
      name: 'CSV',
      icon: <TableChart />,
      description: 'Comma-separated values for spreadsheets',
      size: 'Small',
      compatibility: 'Excel, Google Sheets, etc.'
    },
    {
      id: 'json',
      name: 'JSON',
      icon: <Code />,
      description: 'Structured data format for applications',
      size: 'Medium',
      compatibility: 'APIs, databases, etc.'
    },
    {
      id: 'excel',
      name: 'Excel',
      icon: <InsertDriveFile />,
      description: 'Microsoft Excel workbook format',
      size: 'Large',
      compatibility: 'Excel, LibreOffice Calc'
    },
    {
      id: 'pdf',
      name: 'PDF',
      icon: <PictureAsPdf />,
      description: 'Portable document format for reports',
      size: 'Large',
      compatibility: 'Universal'
    },
    {
      id: 'xml',
      name: 'XML',
      icon: <Description />,
      description: 'Extensible markup language',
      size: 'Medium',
      compatibility: 'Enterprise systems'
    }
  ];

  const availableFields = [
    { id: 'id', name: 'Alert ID', category: 'basic' },
    { id: 'timestamp', name: 'Timestamp', category: 'basic' },
    { id: 'type', name: 'Fraud Type', category: 'basic' },
    { id: 'severity', name: 'Severity', category: 'basic' },
    { id: 'status', name: 'Status', category: 'basic' },
    { id: 'risk_score', name: 'Risk Score', category: 'analysis' },
    { id: 'user_id', name: 'User ID', category: 'user' },
    { id: 'transaction_id', name: 'Transaction ID', category: 'transaction' },
    { id: 'amount', name: 'Amount', category: 'transaction' },
    { id: 'location', name: 'Location', category: 'geographic' },
    { id: 'ip_address', name: 'IP Address', category: 'technical' },
    { id: 'device_id', name: 'Device ID', category: 'technical' },
    { id: 'assigned_analyst', name: 'Assigned Analyst', category: 'workflow' },
    { id: 'investigation_notes', name: 'Investigation Notes', category: 'workflow' },
    { id: 'resolution', name: 'Resolution', category: 'workflow' },
  ];

  const handleFormatChange = (format: string) => {
    setExportState(prev => ({ ...prev, format: format as any }));
  };

  const handleFieldToggle = (fieldId: string) => {
    setExportState(prev => ({
      ...prev,
      includeFields: prev.includeFields.includes(fieldId)
        ? prev.includeFields.filter(f => f !== fieldId)
        : [...prev.includeFields, fieldId]
    }));
  };

  const handleExport = async () => {
    setExporting(true);
    setExportProgress(0);
    setExportError(null);
    setExportComplete(false);

    try {
      // Simulate export process
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setExportProgress(i);
      }

      // Generate mock file content based on format
      let content = '';
      let mimeType = '';
      let fileExtension = '';

      switch (exportState.format) {
        case 'csv':
          content = generateCSV();
          mimeType = 'text/csv';
          fileExtension = 'csv';
          break;
        case 'json':
          content = generateJSON();
          mimeType = 'application/json';
          fileExtension = 'json';
          break;
        case 'excel':
          content = 'Excel format would be generated here';
          mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
          fileExtension = 'xlsx';
          break;
        case 'pdf':
          content = 'PDF content would be generated here';
          mimeType = 'application/pdf';
          fileExtension = 'pdf';
          break;
        case 'xml':
          content = generateXML();
          mimeType = 'application/xml';
          fileExtension = 'xml';
          break;
      }

      // Create and download file
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${exportState.fileName}.${fileExtension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setExportComplete(true);
    } catch (error) {
      setExportError('Failed to export data. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  const generateCSV = () => {
    const headers = exportState.includeFields.join(',');
    const sampleData = [
      'ALT-12345,2025-09-26T10:00:00Z,Card Testing,Critical,New',
      'ALT-12346,2025-09-26T10:05:00Z,Account Takeover,High,Investigating',
      'ALT-12347,2025-09-26T10:10:00Z,Velocity Check Failed,Medium,Resolved'
    ];
    return [headers, ...sampleData].join('\n');
  };

  const generateJSON = () => {
    const sampleData = [
      {
        id: 'ALT-12345',
        timestamp: '2025-09-26T10:00:00Z',
        type: 'Card Testing',
        severity: 'Critical',
        status: 'New'
      },
      {
        id: 'ALT-12346',
        timestamp: '2025-09-26T10:05:00Z',  
        type: 'Account Takeover',
        severity: 'High',
        status: 'Investigating'
      }
    ];
    return JSON.stringify({ 
      metadata: { 
        export_date: new Date().toISOString(),
        record_count: sampleData.length,
        format_version: '1.0'
      },
      data: sampleData 
    }, null, 2);
  };

  const generateXML = () => {
    return `<?xml version="1.0" encoding="UTF-8"?>
<fraud_alerts>
  <metadata>
    <export_date>${new Date().toISOString()}</export_date>
    <record_count>2</record_count>
  </metadata>
  <alerts>
    <alert>
      <id>ALT-12345</id>
      <timestamp>2025-09-26T10:00:00Z</timestamp>
      <type>Card Testing</type>
      <severity>Critical</severity>
      <status>New</status>
    </alert>
    <alert>
      <id>ALT-12346</id>
      <timestamp>2025-09-26T10:05:00Z</timestamp>
      <type>Account Takeover</type>
      <severity>High</severity>
      <status>Investigating</status>
    </alert>
  </alerts>
</fraud_alerts>`;
  };

  const getEstimatedSize = () => {
    const baseSize = exportState.includeFields.length * 100; // Rough estimate
    const multiplier = { csv: 1, json: 1.5, excel: 2, pdf: 3, xml: 1.8 };
    return Math.round(baseSize * multiplier[exportState.format] / 1024) + ' KB';
  };

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
          <GetApp color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Export Data - {exportType.charAt(0).toUpperCase() + exportType.slice(1)}
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        {exportError && (
          <Alert severity="error" sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1}>
              <Error />
              {exportError}
            </Box>
          </Alert>
        )}

        {exportComplete && (
          <Alert severity="success" sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1}>
              <CheckCircle />
              Export completed successfully! Your file has been downloaded.
            </Box>
          </Alert>
        )}

        {exporting && (
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={2} mb={1}>
              <Schedule />
              <Typography>Exporting data...</Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={exportProgress}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" color="text.secondary">
              {exportProgress}% complete
            </Typography>
          </Box>
        )}

        <Grid container spacing={3}>
          {/* Format Selection */}
          <Grid item xs={12}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Export Format
            </Typography>
            <Grid container spacing={2}>
              {formatOptions.map((format) => (
                <Grid item xs={12} sm={6} md={4} key={format.id}>
                  <ExportCard 
                    className={exportState.format === format.id ? 'selected' : ''}
                    onClick={() => handleFormatChange(format.id)}
                  >
                    <CardContent>
                      <Box display="flex" alignItems="center" gap={2} mb={1}>
                        {format.icon}
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {format.name}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {format.description}
                      </Typography>
                      <Box display="flex" gap={1}>
                        <Chip size="small" label={`Size: ${format.size}`} />
                        <Chip size="small" label={format.compatibility} variant="outlined" />
                      </Box>
                    </CardContent>
                  </ExportCard>
                </Grid>
              ))}
            </Grid>
          </Grid>

          {/* Field Selection */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Include Fields
            </Typography>
            <Card>
              <CardContent>
                <List dense>
                  {availableFields.map((field) => (
                    <ListItem key={field.id} sx={{ px: 0 }}>
                      <ListItemIcon>
                        <Checkbox
                          checked={exportState.includeFields.includes(field.id)}
                          onChange={() => handleFieldToggle(field.id)}
                          size="small"
                        />
                      </ListItemIcon>
                      <ListItemText 
                        primary={field.name}
                        secondary={field.category}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Export Options */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Export Options
            </Typography>
            <Card>
              <CardContent>
                <Box mb={3}>
                  <TextField
                    label="File Name"
                    fullWidth
                    size="small"
                    value={exportState.fileName}
                    onChange={(e) => setExportState(prev => ({ ...prev, fileName: e.target.value }))}
                    sx={{ mb: 2 }}
                  />
                </Box>

                <Box mb={3}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    Date Range
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="Start Date"
                        type="date"
                        size="small"
                        fullWidth
                        value={exportState.dateRange.start}
                        onChange={(e) => setExportState(prev => ({
                          ...prev,
                          dateRange: { ...prev.dateRange, start: e.target.value }
                        }))}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="End Date"
                        type="date"
                        size="small"
                        fullWidth
                        value={exportState.dateRange.end}
                        onChange={(e) => setExportState(prev => ({
                          ...prev,
                          dateRange: { ...prev.dateRange, end: e.target.value }
                        }))}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                  </Grid>
                </Box>

                <Box mb={3}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    Additional Options
                  </Typography>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportState.options.includeHeaders}
                        onChange={(e) => setExportState(prev => ({
                          ...prev,
                          options: { ...prev.options, includeHeaders: e.target.checked }
                        }))}
                      />
                    }
                    label="Include headers"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportState.options.includeMetadata}
                        onChange={(e) => setExportState(prev => ({
                          ...prev,
                          options: { ...prev.options, includeMetadata: e.target.checked }
                        }))}
                      />
                    }
                    label="Include metadata"
                  />
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={exportState.options.compressFile}
                        onChange={(e) => setExportState(prev => ({
                          ...prev,
                          options: { ...prev.options, compressFile: e.target.checked }
                        }))}
                      />
                    }
                    label="Compress file"
                  />
                </Box>

                <Divider sx={{ my: 2 }} />

                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    Export Summary
                  </Typography>
                  <Box display="flex" gap={1} flexWrap="wrap">
                    <Chip size="small" label={`Format: ${exportState.format.toUpperCase()}`} />
                    <Chip size="small" label={`Fields: ${exportState.includeFields.length}`} />
                    <Chip size="small" label={`Est. Size: ${getEstimatedSize()}`} />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </DialogContent>

      <DialogActions sx={{ 
        p: 3, 
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Button onClick={onClose} disabled={exporting}>
          Cancel
        </Button>
        <Button
          startIcon={<GetApp />}
          onClick={handleExport}
          variant="contained"
          color="primary"
          disabled={exporting || exportState.includeFields.length === 0}
        >
          {exporting ? 'Exporting...' : 'Export Data'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DataExportPanel;