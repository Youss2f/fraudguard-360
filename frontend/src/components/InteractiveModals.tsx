import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Chip,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  RadioGroup,
  Radio,
  Checkbox,
  Tabs,
  Tab,
  LinearProgress,
  IconButton,
  Tooltip,
  Fade,
  Zoom,
  Slide,
  Grow,
} from '@mui/material';
import {
  FilterList,
  Visibility,
  Download,
  Share,
  Security,
  Assessment,
  TrendingUp,
  BarChart,
  Warning,
  Block,
  Search as Investigation,
  CheckCircle,
  ExpandMore,
  Close,
  Refresh,
  Settings,
  Fullscreen,
  Timeline,
  Analytics,
  BugReport,
  Shield,
  Speed,
  DataUsage,
  Memory,
  Storage,
  CloudDownload,
  Email,
  Link,
  Folder,
  Schedule,
  Edit,
  Delete,
  Archive,
  Star,
  BookmarkBorder,
  ThumbUp,
  Flag,
} from '@mui/icons-material';

const excelColors = {
  primary: { main: '#0078d4', dark: '#106ebe', light: '#40e0ff' },
  secondary: { main: '#107c10', dark: '#0b5e0b', light: '#6bb26b' },
  accent: { 
    orange: '#ff8c00', 
    red: '#d83b01', 
    blue: '#0078d4',
    green: '#107c10',
    purple: '#5c2e91',
    teal: '#008272'
  },
  background: {
    default: '#faf9f8',
    paper: '#ffffff',
    border: '#e1dfdd',
    hover: '#f3f2f1',
  },
  text: {
    primary: '#323130',
    secondary: '#605e5c',
  }
};

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  type: 'filter' | 'export' | 'share' | 'actions' | 'threats' | 'reports' | 'charts' | 'alerts';
  data?: any;
  onApplyFilter?: (filters: any) => void;
  onExportData?: (format: string, options: any) => void;
  onActionExecute?: (action: string, data: any) => void;
}

export const RevolutionaryModal: React.FC<ModalProps> = ({ 
  open, 
  onClose, 
  title, 
  type, 
  data, 
  onApplyFilter,
  onExportData,
  onActionExecute 
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [progress, setProgress] = useState(0);
  const [animateProgress, setAnimateProgress] = useState(false);
  const [filters, setFilters] = useState({
    riskLevel: 'all',
    dateRange: '7days',
    amount: [0, 100000],
    status: [] as string[],
    region: 'all'
  });
  const [exportFormat, setExportFormat] = useState('excel');
  const [exportOptions, setExportOptions] = useState({
    includeCharts: true,
    includeDetails: true,
    dateRange: '30days'
  });

  useEffect(() => {
    if (open && type === 'export') {
      setAnimateProgress(true);
      const timer = setInterval(() => {
        setProgress((oldProgress) => {
          if (oldProgress === 100) {
            clearInterval(timer);
            return 100;
          }
          const diff = Math.random() * 10;
          return Math.min(oldProgress + diff, 100);
        });
      }, 200);
      return () => clearInterval(timer);
    }
  }, [open, type]);

  const handleFilterChange = (key: string, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const handleApplyFilters = () => {
    if (onApplyFilter) {
      onApplyFilter(filters);
    }
    onClose();
  };

  const handleExportData = (format: string) => {
    setExportFormat(format);
    setAnimateProgress(true);
    const timer = setInterval(() => {
      setProgress((oldProgress) => {
        if (oldProgress === 100) {
          clearInterval(timer);
          if (onExportData) {
            onExportData(format, exportOptions);
          }
          setTimeout(() => {
            setProgress(0);
            setAnimateProgress(false);
            onClose();
          }, 500);
          return 100;
        }
        const diff = Math.random() * 10;
        return Math.min(oldProgress + diff, 100);
      });
    }, 200);
  };

  const handleActionExecute = (actionType: string) => {
    if (onActionExecute) {
      onActionExecute(actionType, data);
    }
    onClose();
  };

  const renderFilterModal = () => (
    <Box sx={{ minWidth: 600 }}>
      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tab label="Basic Filters" />
        <Tab label="Advanced" />
        <Tab label="Saved Filters" />
      </Tabs>
      
      {tabValue === 0 && (
        <Box sx={{ p: 3 }}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Risk Level</InputLabel>
                <Select
                  value={filters.riskLevel}
                  onChange={(e) => handleFilterChange('riskLevel', e.target.value)}
                >
                  <MenuItem value="all">All Levels</MenuItem>
                  <MenuItem value="high">High Risk</MenuItem>
                  <MenuItem value="medium">Medium Risk</MenuItem>
                  <MenuItem value="low">Low Risk</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Date Range</InputLabel>
                <Select
                  value={filters.dateRange}
                  onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                >
                  <MenuItem value="1day">Last 24 Hours</MenuItem>
                  <MenuItem value="7days">Last 7 Days</MenuItem>
                  <MenuItem value="30days">Last 30 Days</MenuItem>
                  <MenuItem value="custom">Custom Range</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Typography gutterBottom>Amount Range: ${filters.amount[0].toLocaleString()} - ${filters.amount[1].toLocaleString()}</Typography>
              <Slider
                value={filters.amount}
                onChange={(e, value) => handleFilterChange('amount', value)}
                valueLabelDisplay="auto"
                max={1000000}
                step={1000}
                marks={[
                  { value: 0, label: '$0' },
                  { value: 50000, label: '$50K' },
                  { value: 100000, label: '$100K' },
                  { value: 500000, label: '$500K' },
                  { value: 1000000, label: '$1M' }
                ]}
              />
            </Grid>
          </Grid>
        </Box>
      )}

      {tabValue === 1 && (
        <Box sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Advanced Fraud Detection Filters</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>Transaction Types</Typography>
              {['Credit Card', 'Wire Transfer', 'ACH', 'Cryptocurrency', 'Mobile Payment'].map(type => (
                <FormControlLabel
                  key={type}
                  control={<Checkbox />}
                  label={type}
                />
              ))}
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>Fraud Patterns</Typography>
              {['Velocity Fraud', 'Identity Theft', 'Account Takeover', 'Synthetic Identity', 'Money Laundering'].map(pattern => (
                <FormControlLabel
                  key={pattern}
                  control={<Checkbox />}
                  label={pattern}
                />
              ))}
            </Grid>
          </Grid>
        </Box>
      )}

      {tabValue === 2 && (
        <Box sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Saved Filter Sets</Typography>
          <List>
            {['High Risk Transactions', 'Recent Fraud Cases', 'International Transfers', 'Large Amount Alerts'].map((filter, index) => (
              <ListItem key={filter} sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}>
                <ListItemIcon>
                  <Star color="primary" />
                </ListItemIcon>
                <ListItemText primary={filter} secondary={`Applied ${index + 1} times this week`} />
                <Button size="small">Apply</Button>
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Box>
  );

  const renderExportModal = () => (
    <Box sx={{ minWidth: 500 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>Export Data</Typography>
          {animateProgress && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress variant="determinate" value={progress} />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Preparing export... {Math.round(progress)}%
              </Typography>
            </Box>
          )}
        </Grid>
        <Grid item xs={12} md={6}>
          <Card 
            sx={{ cursor: 'pointer', '&:hover': { bgcolor: excelColors.background.hover } }}
            onClick={() => handleExportData('excel')}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Assessment color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Excel Report</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Comprehensive fraud analysis with charts and pivot tables
              </Typography>
              <Chip label="Recommended" color="primary" size="small" sx={{ mt: 1 }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card 
            sx={{ cursor: 'pointer', '&:hover': { bgcolor: excelColors.background.hover } }}
            onClick={() => handleExportData('csv')}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <DataUsage color="secondary" sx={{ mr: 1 }} />
                <Typography variant="h6">CSV Data</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Raw data export for further analysis
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card sx={{ cursor: 'pointer', '&:hover': { bgcolor: excelColors.background.hover } }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <CloudDownload color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">PDF Report</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Executive summary for stakeholders
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card sx={{ cursor: 'pointer', '&:hover': { bgcolor: excelColors.background.hover } }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Analytics color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">PowerBI Dataset</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Interactive dashboards and visualizations
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderShareModal = () => (
    <Box sx={{ minWidth: 500 }}>
      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
        <Tab label="Email" />
        <Tab label="Link Sharing" />
        <Tab label="Export to Drive" />
      </Tabs>
      
      {tabValue === 0 && (
        <Box sx={{ p: 3 }}>
          <TextField
            fullWidth
            label="Recipients"
            placeholder="Enter email addresses separated by commas"
            multiline
            rows={3}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Subject"
            defaultValue="FraudGuard 360 - Fraud Detection Report"
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Message"
            multiline
            rows={4}
            defaultValue="Please find the attached fraud detection report from FraudGuard 360 system."
          />
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Include executive summary"
            sx={{ mt: 2 }}
          />
        </Box>
      )}

      {tabValue === 1 && (
        <Box sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Generate Secure Link</Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Access Level</InputLabel>
            <Select defaultValue="view">
              <MenuItem value="view">View Only</MenuItem>
              <MenuItem value="comment">View & Comment</MenuItem>
              <MenuItem value="edit">Full Access</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Expiration</InputLabel>
            <Select defaultValue="7days">
              <MenuItem value="1day">24 Hours</MenuItem>
              <MenuItem value="7days">7 Days</MenuItem>
              <MenuItem value="30days">30 Days</MenuItem>
              <MenuItem value="never">Never</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Generated Link"
            value="https://fraudguard360.com/shared/abc123xyz"
            InputProps={{ readOnly: true }}
            sx={{ mb: 2 }}
          />
          <Button variant="outlined" startIcon={<Link />} fullWidth>
            Copy Link to Clipboard
          </Button>
        </Box>
      )}

      {tabValue === 2 && (
        <Box sx={{ p: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Card sx={{ cursor: 'pointer', textAlign: 'center', p: 2 }}>
                <Folder color="primary" sx={{ fontSize: 48, mb: 1 }} />
                <Typography variant="h6">OneDrive</Typography>
                <Typography variant="body2" color="text.secondary">
                  Save to your OneDrive folder
                </Typography>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card sx={{ cursor: 'pointer', textAlign: 'center', p: 2 }}>
                <CloudDownload color="secondary" sx={{ fontSize: 48, mb: 1 }} />
                <Typography variant="h6">SharePoint</Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload to team SharePoint
                </Typography>
              </Card>
            </Grid>
          </Grid>
        </Box>
      )}
    </Box>
  );

  const renderActionsModal = () => (
    <Box sx={{ minWidth: 600 }}>
      <Typography variant="h6" gutterBottom>Available Actions</Typography>
      <Grid container spacing={2}>
        {[
          { icon: Investigation, title: 'Investigate Case', desc: 'Deep dive analysis and evidence collection', color: 'primary' },
          { icon: Block, title: 'Block Account', desc: 'Immediately suspend suspicious account', color: 'error' },
          { icon: CheckCircle, title: 'Mark Resolved', desc: 'Close case as resolved with notes', color: 'success' },
          { icon: Flag, title: 'Escalate to L2', desc: 'Forward to senior fraud analyst', color: 'warning' },
          { icon: Shield, title: 'Add to Whitelist', desc: 'Mark as trusted transaction', color: 'info' },
          { icon: BugReport, title: 'Report False Positive', desc: 'Improve detection algorithms', color: 'secondary' }
        ].map((action, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Grow in={open} timeout={300 + index * 100}>
              <Card sx={{ 
                cursor: 'pointer', 
                '&:hover': { 
                  transform: 'translateY(-2px)', 
                  boxShadow: 4,
                  bgcolor: excelColors.background.hover 
                },
                transition: 'all 0.3s ease'
              }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <action.icon color={action.color as any} sx={{ mr: 2 }} />
                    <Typography variant="h6">{action.title}</Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {action.desc}
                  </Typography>
                </CardContent>
              </Card>
            </Grow>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const renderContent = () => {
    switch (type) {
      case 'filter': return renderFilterModal();
      case 'export': return renderExportModal();
      case 'share': return renderShareModal();
      case 'actions': return renderActionsModal();
      default: return <Typography>Interactive content for {type}</Typography>;
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      TransitionComponent={Zoom}
      transitionDuration={300}
    >
      <DialogTitle sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        bgcolor: excelColors.primary.main,
        color: 'white'
      }}>
        <Typography variant="h6">{title}</Typography>
        <IconButton onClick={onClose} sx={{ color: 'white' }}>
          <Close />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0 }}>
        {renderContent()}
      </DialogContent>
      <DialogActions sx={{ p: 2, bgcolor: excelColors.background.hover }}>
        <Button onClick={onClose} variant="outlined">
          Cancel
        </Button>
        {type === 'filter' && (
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleApplyFilters}
            startIcon={<FilterList />}
          >
            Apply Filters
          </Button>
        )}
        {type === 'export' && (
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => handleExportData(exportFormat)}
            startIcon={<Download />}
            disabled={animateProgress}
          >
            {animateProgress ? `Exporting... ${Math.round(progress)}%` : 'Export Data'}
          </Button>
        )}
        {type === 'share' && (
          <Button 
            variant="contained" 
            color="primary" 
            onClick={onClose}
            startIcon={<Share />}
          >
            Share Now
          </Button>
        )}
        {type === 'actions' && (
          <Button 
            variant="contained" 
            color="primary" 
            onClick={onClose}
            startIcon={<Security />}
          >
            Execute Action
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default RevolutionaryModal;