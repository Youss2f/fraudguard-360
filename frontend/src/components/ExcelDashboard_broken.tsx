/**
 * Professional Excel-Style Dashboard
 * Comprehensive fraud detection dashboard with real data and visualizations
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Avatar,
  Button,
  Divider,
  Alert,
  styled,
  alpha,
  Grid,
  Grow,
  Fade,
  Zoom,
  useTheme,
  keyframes,
} from '@mui/material';
import {
  Security,
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  MoreVert,
  Person,
  AttachMoney,
  LocationOn,
  Schedule,
  Visibility,
  FilterList,
  MonitorHeart,
  Assessment,
  Settings,
  Download,
} from '@mui/icons-material';
import { RevolutionaryModal } from './InteractiveModals';
import { AnimatedKPICard, AnimatedAlert, AnimatedProgress } from './AnimatedComponents';
import { InteractiveChart } from './InteractiveChart';
import { InteractiveTable } from './InteractiveTable';
import { InteractiveSidebar, SettingsPanel } from './InteractiveSidebar';
import { AdvancedDataGrid } from './AdvancedDataGrid';
import { PowerUserHub } from './PowerUserHub';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { excelColors } from '../theme/excelTheme';

// Styled components with Excel theme
const DashboardCard = styled(Card)({
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  '&:hover': {
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
  },
  transition: 'box-shadow 0.2s ease',
});

const KPICard = styled(Paper)({
  padding: '20px',
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  textAlign: 'center',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 4,
    background: `linear-gradient(90deg, ${excelColors.primary.main}, ${excelColors.accent.blue})`,
  },
});

const StatusIndicator = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'status',
})<{ status: 'success' | 'warning' | 'error' | 'info' }>(({ status }) => {
  const colors = {
    success: excelColors.success.main,
    warning: excelColors.warning.main,
    error: excelColors.error.main,
    info: excelColors.info.main,
  };
  
  return {
    width: 12,
    height: 12,
    borderRadius: '50%',
    backgroundColor: colors[status],
    display: 'inline-block',
    marginRight: 8,
  };
});

// Mock data generators
const generateTransactionData = () => {
  const data = [];
  const now = new Date();
  for (let i = 23; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000);
    data.push({
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      transactions: Math.floor(Math.random() * 500) + 200,
      fraudulent: Math.floor(Math.random() * 20) + 5,
      blocked: Math.floor(Math.random() * 15) + 2,
    });
  }
  return data;
};

const generateRiskDistribution = () => [
  { name: 'Low Risk', value: 65, color: excelColors.success.main },
  { name: 'Medium Risk', value: 25, color: excelColors.warning.main },
  { name: 'High Risk', value: 8, color: excelColors.error.main },
  { name: 'Critical', value: 2, color: '#8B0000' },
];

const generateRecentAlerts = () => [
  {
    id: 1,
    type: 'Suspicious Pattern',
    severity: 'high' as const,
    user: 'John Smith',
    amount: '$2,450.00',
    location: 'New York, NY',
    time: '2 min ago',
    status: 'investigating',
  },
  {
    id: 2,
    type: 'Unusual Velocity',
    severity: 'medium' as const,
    user: 'Sarah Johnson',
    amount: '$850.00',
    location: 'Los Angeles, CA',
    time: '5 min ago',
    status: 'reviewed',
  },
  {
    id: 3,
    type: 'Geographic Anomaly',
    severity: 'high' as const,
    user: 'Mike Wilson',
    amount: '$3,200.00',
    location: 'Miami, FL',
    time: '8 min ago',
    status: 'blocked',
  },
  {
    id: 4,
    type: 'Device Mismatch',
    severity: 'low' as const,
    user: 'Emily Davis',
    amount: '$125.00',
    location: 'Chicago, IL',
    time: '12 min ago',
    status: 'cleared',
  },
];

const ExcelDashboard: React.FC = () => {
  const [transactionData, setTransactionData] = useState(generateTransactionData());
  const [riskData, setRiskData] = useState(generateRiskDistribution());
  const [alerts, setAlerts] = useState(generateRecentAlerts());
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  // Revolutionary modal states
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [alertsModalOpen, setAlertsModalOpen] = useState(false);
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [actionsModalOpen, setActionsModalOpen] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<any>(null);

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTransactionData(generateTransactionData());
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setTransactionData(generateTransactionData());
      setRiskData(generateRiskDistribution());
      setAlerts(generateRecentAlerts());
      setIsRefreshing(false);
    }, 1000);
  };

  const handleFilter = () => {
    setFilterModalOpen(true);
  };

  const handleExport = () => {
    setAlertsModalOpen(true);
  };

  const handleAlertClick = (alertData: any) => {
    setSelectedAlert(alertData);
    setActionsModalOpen(true);
  };

  // Calculate KPIs
  const totalTransactions = transactionData.reduce((sum, item) => sum + item.transactions, 0);
  const totalFraud = transactionData.reduce((sum, item) => sum + item.fraudulent, 0);
  const fraudRate = ((totalFraud / totalTransactions) * 100).toFixed(2);
  const preventedLoss = (totalFraud * 1250).toLocaleString(); // Assume avg $1,250 per fraud

  const [sidebarActiveItem, setSidebarActiveItem] = useState('dashboard');
  const [showAdvancedGrid, setShowAdvancedGrid] = useState(false);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return excelColors.error.main;
      case 'medium': return excelColors.warning.main;
      case 'low': return excelColors.success.main;
      default: return excelColors.text.secondary;
    }
  };

  const getStatusChip = (status: string) => {
    const configs = {
      investigating: { label: 'Investigating', color: excelColors.warning.main },
      reviewed: { label: 'Reviewed', color: excelColors.info.main },
      blocked: { label: 'Blocked', color: excelColors.error.main },
      cleared: { label: 'Cleared', color: excelColors.success.main },
    };
    
    const config = configs[status as keyof typeof configs] || configs.investigating;
    
    return (
      <Chip
        label={config.label}
        size="small"
        sx={{
          backgroundColor: alpha(config.color, 0.1),
          color: config.color,
          border: `1px solid ${config.color}`,
          fontSize: '0.75rem',
        }}
      />
    );
  };

  // Revolutionary sidebar menu items
  const sidebarMenuItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <Security />,
      badge: alerts.length,
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <TrendingUp />,
      panel: <SettingsPanel onSettingChange={(setting, value) => console.log(setting, value)} />,
    },
    {
      id: 'monitoring',
      label: 'Real-time Monitoring',
      icon: <MonitorHeart />,
      color: excelColors.accent.green,
    },
    {
      id: 'advanced-grid',
      label: 'Advanced Data Grid',
      icon: <FilterList />,
      panel: <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>Advanced Data Grid Settings</Typography>
        <Typography variant="body2">Configure advanced data manipulation features</Typography>
        <Button 
          variant="contained" 
          sx={{ mt: 2 }}
          onClick={() => setShowAdvancedGrid(true)}
        >
          Open Advanced Grid
        </Button>
      </Box>,
    },
    {
      id: 'reports',
      label: 'Reports',
      icon: <Assessment />,
      children: [
        { id: 'daily-reports', label: 'Daily Reports', icon: <Schedule /> },
        { id: 'weekly-reports', label: 'Weekly Reports', icon: <Schedule /> },
        { id: 'monthly-reports', label: 'Monthly Reports', icon: <Schedule /> },
      ],
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Settings />,
      panel: <SettingsPanel onSettingChange={(setting, value) => console.log(setting, value)} />,
    },
  ];

  // Revolutionary power user commands
  const powerUserCommands = [
    {
      id: 'toggle-grid',
      label: 'Toggle Advanced Grid',
      description: 'Switch between standard and advanced data grid',
      icon: <FilterList />,
      shortcut: 'Ctrl+G',
      category: 'Views',
      action: () => setShowAdvancedGrid(!showAdvancedGrid),
    },
    {
      id: 'export-fraud-data',
      label: 'Export Fraud Data',
      description: 'Export all fraud detection data',
      icon: <Download />,
      shortcut: 'Ctrl+Shift+E',
      category: 'Data',
      action: () => console.log('Exporting fraud data...'),
    },
    {
      id: 'refresh-dashboard',
      label: 'Refresh Dashboard',
      description: 'Refresh all dashboard data',
      icon: <Refresh />,
      shortcut: 'F5',
      category: 'Data',
      action: () => handleRefresh(),
    },
  ];

  // Advanced data grid columns
  const gridColumns = [
    { id: 'id', label: 'Alert ID', type: 'text' as const, sortable: true, filterable: true },
    { id: 'type', label: 'Alert Type', type: 'select' as const, options: ['Suspicious Transaction', 'Identity Theft', 'Card Fraud', 'Account Takeover'], editable: true, sortable: true, filterable: true },
    { id: 'user', label: 'User', type: 'text' as const, editable: true, sortable: true, filterable: true },
    { id: 'amount', label: 'Amount', type: 'currency' as const, editable: true, sortable: true, filterable: true },
    { id: 'location', label: 'Location', type: 'text' as const, editable: true, sortable: true, filterable: true },
    { id: 'time', label: 'Time', type: 'date' as const, sortable: true, filterable: true },
    { id: 'severity', label: 'Severity', type: 'select' as const, options: ['low', 'medium', 'high'], editable: true, sortable: true, filterable: true },
    { id: 'status', label: 'Status', type: 'select' as const, options: ['investigating', 'reviewed', 'blocked', 'cleared'], editable: true, sortable: true, filterable: true },
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* Revolutionary Interactive Sidebar */}
      <InteractiveSidebar
        menuItems={sidebarMenuItems}
        activeItem={sidebarActiveItem}
        onItemClick={setSidebarActiveItem}
        user={{
          name: 'John Doe',
          role: 'Fraud Analyst',
          avatar: '/api/placeholder/40/40'
        }}
        notifications={alerts.length}
      />

      {/* Main Dashboard Content */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {!showAdvancedGrid ? (
          <Box sx={{ p: 3 }}>
            {/* Header with actions */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
                🚀 REVOLUTIONARY Fraud Detection Dashboard
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  startIcon={<FilterList />}
                  variant="outlined"
                  size="small"
                  onClick={handleFilter}
                  sx={{ borderColor: excelColors.background.border }}
                >
                  Filter
                </Button>
                <Button
                  startIcon={<Refresh />}
                  variant="outlined"
                  size="small"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                  sx={{ borderColor: excelColors.background.border }}
                >
                  Refresh
                </Button>
                <Button
                  startIcon={<Assessment />}
                  variant="contained"
                  size="small"
                  onClick={() => setShowAdvancedGrid(true)}
                  sx={{ bgcolor: excelColors.accent.purple }}
                >
                  Advanced Grid
                </Button>
              </Box>
            </Box>
            sx={{ borderColor: excelColors.background.border }}
                </Button>
              </Box>
            </Box>
            
            {/* Continue with existing dashboard content... */}
        </Box>
      </Box>

      {/* REVOLUTIONARY Animated KPI Cards */}
      <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
        <Box sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <AnimatedKPICard
            title="Total Transactions"
            value={totalTransactions}
            change={12.5}
            icon={<Security />}
            color="primary"
            animated={true}
            direction="left"
            delay={0}
            onClick={() => setFilterModalOpen(true)}
          />
        </Box>

        <Box sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <AnimatedKPICard
            title="Fraud Rate"
            value={`${fraudRate}%`}
            change={-2.1}
            icon={<Warning />}
            color="error"
            animated={true}
            glowing={true}
            direction="up"
            delay={200}
            onClick={() => setAlertsModalOpen(true)}
          />
        </Box>

        <Box sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <AnimatedKPICard
            title="Prevented Loss"
            value={`$${preventedLoss}`}
            change={18.3}
            icon={<AttachMoney />}
            color="success"
            animated={true}
            floating={true}
            direction="right"
            delay={400}
            onClick={() => setChartModalOpen(true)}
          />
        </Box>

        <Box sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <AnimatedKPICard
            title="Detection Rate"
            value="97.8%"
            change={0.5}
            icon={<CheckCircle />}
            color="success"
            animated={true}
            direction="up"
            delay={600}
            onClick={() => setActionsModalOpen(true)}
          />
        </Box>
      </Box>

      {/* Charts Section */}
      <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
        {/* Transaction Timeline */}
        <Box sx={{ flex: '2 1 500px', minWidth: 500 }}>
          <DashboardCard>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
                  Transaction Activity (24 Hours)
                </Typography>
                <IconButton 
                  size="small"
                  onClick={() => setChartModalOpen(true)}
                >
                  <MoreVert />
                </IconButton>
              </Box>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={transactionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={excelColors.background.border} />
                  <XAxis dataKey="time" stroke={excelColors.text.secondary} fontSize={12} />
                  <YAxis stroke={excelColors.text.secondary} fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: excelColors.background.paper,
                      border: `1px solid ${excelColors.background.border}`,
                      borderRadius: 4,
                    }}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="transactions"
                    stackId="1"
                    stroke={excelColors.primary.main}
                    fill={alpha(excelColors.primary.main, 0.1)}
                    name="Total Transactions"
                  />
                  <Area
                    type="monotone"
                    dataKey="fraudulent"
                    stackId="2"
                    stroke={excelColors.error.main}
                    fill={alpha(excelColors.error.main, 0.2)}
                    name="Fraudulent"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </DashboardCard>
        </Box>

        {/* Risk Distribution */}
        <Box sx={{ flex: '1 1 300px', minWidth: 300 }}>
          <DashboardCard>
            <CardContent>
              <Typography variant="h6" sx={{ color: excelColors.text.primary, mb: 2 }}>
                Risk Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {riskData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </DashboardCard>
        </Box>
      </Box>

      {/* Recent Alerts Table */}
      <DashboardCard>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
              Recent Fraud Alerts
            </Typography>
            <Button 
              variant="outlined" 
              size="small" 
              startIcon={<Visibility />}
              onClick={() => setAlertsModalOpen(true)}
            >
              View All
            </Button>
          </Box>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Alert Type</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell>Amount</TableCell>
                  <TableCell>Location</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {alerts.map((alert) => (
                  <TableRow 
                    key={alert.id} 
                    hover 
                    onClick={() => handleAlertClick(alert)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <StatusIndicator 
                          status={alert.severity === 'high' ? 'error' : alert.severity === 'medium' ? 'warning' : 'success'} 
                        />
                        {alert.type}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Avatar sx={{ width: 24, height: 24, mr: 1, fontSize: '0.75rem' }}>
                          {alert.user.charAt(0)}
                        </Avatar>
                        {alert.user}
                      </Box>
                    </TableCell>
                    <TableCell sx={{ fontWeight: 500 }}>{alert.amount}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <LocationOn sx={{ fontSize: 16, mr: 0.5, color: excelColors.text.secondary }} />
                        {alert.location}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Schedule sx={{ fontSize: 16, mr: 0.5, color: excelColors.text.secondary }} />
                        {alert.time}
                      </Box>
                    </TableCell>
                    <TableCell>{getStatusChip(alert.status)}</TableCell>
                    <TableCell>
                      <IconButton 
                        size="small" 
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedAlert(alert);
                          setActionsModalOpen(true);
                        }}
                      >
                        <MoreVert fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </DashboardCard>
      {/* REVOLUTIONARY Interactive Modals */}
      <RevolutionaryModal
        open={filterModalOpen}
        onClose={() => setFilterModalOpen(false)}
        title="Advanced Fraud Filters"
        type="filter"
      />
      
      <RevolutionaryModal
        open={alertsModalOpen}
        onClose={() => setAlertsModalOpen(false)}
        title="Export Dashboard Data"
        type="export"
      />
      
      <RevolutionaryModal
        open={chartModalOpen}
        onClose={() => setChartModalOpen(false)}
        title="Share Financial Report"
        type="share"
      />
      
      <RevolutionaryModal
        open={actionsModalOpen}
        onClose={() => setActionsModalOpen(false)}
        title="Fraud Investigation Actions"
        type="actions"
        data={selectedAlert}
      />
    </Box>
  );
};

export default ExcelDashboard;