/**
 * Revolutionary Excel-Style Dashboard - FIXED VERSION
 * Comprehensive fraud detection dashboard with revolutionary interactive components
 */

import React, { useState, useEffect, memo } from 'react';
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

const StatusIndicator = styled(Box)<{ status: 'error' | 'warning' | 'success' | 'info' }>(({ status }) => ({
  width: 8,
  height: 8,
  borderRadius: '50%',
  marginRight: 8,
  backgroundColor: status === 'error' ? excelColors.error.main :
                  status === 'warning' ? excelColors.warning.main :
                  status === 'success' ? excelColors.success.main :
                  excelColors.info.main,
}));

// Sample data generators
const generateTransactionData = () => {
  const hours = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, '0');
    return {
      time: `${hour}:00`,
      transactions: Math.floor(Math.random() * 1000) + 500,
      fraudulent: Math.floor(Math.random() * 50) + 10,
    };
  });
  return hours;
};

const generateRiskDistribution = () => [
  { name: 'Low Risk', value: 65, color: excelColors.success.main },
  { name: 'Medium Risk', value: 25, color: excelColors.warning.main },
  { name: 'High Risk', value: 10, color: excelColors.error.main },
];

const generateRecentAlerts = () => [
  {
    id: 1,
    type: 'Suspicious Transaction',
    user: 'John Smith',
    amount: '$2,547.00',
    location: 'New York, NY',
    time: '2 min ago',
    severity: 'high',
    status: 'investigating',
  },
  {
    id: 2,
    type: 'Identity Theft',
    user: 'Sarah Johnson',
    amount: '$892.50',
    location: 'Los Angeles, CA',
    time: '5 min ago',
    severity: 'medium',
    status: 'reviewed',
  },
  {
    id: 3,
    type: 'Card Fraud',
    user: 'Michael Brown',
    amount: '$1,234.75',
    location: 'Chicago, IL',
    time: '8 min ago',
    severity: 'high',
    status: 'blocked',
  },
  {
    id: 4,
    type: 'Account Takeover',
    user: 'Emily Davis',
    amount: '$567.25',
    location: 'Houston, TX',
    time: '12 min ago',
    severity: 'low',
    status: 'cleared',
  },
];

const ExcelDashboard: React.FC = () => {
  const [transactionData, setTransactionData] = useState(generateTransactionData());
  const [riskData, setRiskData] = useState(generateRiskDistribution());
  const [alerts, setAlerts] = useState(generateRecentAlerts());
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  // Filter states
  const [appliedFilters, setAppliedFilters] = useState<any>({
    riskLevel: 'all',
    dateRange: '7days',
    amount: [0, 100000],
    status: [],
    region: 'all'
  });
  const [filteredAlerts, setFilteredAlerts] = useState(generateRecentAlerts());
  const [filteredTransactionData, setFilteredTransactionData] = useState(generateTransactionData());
  
  // Revolutionary component states
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [alertsModalOpen, setAlertsModalOpen] = useState(false);
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [actionsModalOpen, setActionsModalOpen] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<any>(null);
  const [sidebarActiveItem, setSidebarActiveItem] = useState('dashboard');
  const [showAdvancedGrid, setShowAdvancedGrid] = useState(false);

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
      const newTransactionData = generateTransactionData();
      const newAlerts = generateRecentAlerts();
      setTransactionData(newTransactionData);
      setRiskData(generateRiskDistribution());
      setAlerts(newAlerts);
      
      // Apply current filters to new data
      applyFiltersToData(newTransactionData, newAlerts, appliedFilters);
      setIsRefreshing(false);
    }, 1000);
  };

  const applyFiltersToData = (transData: any[], alertsData: any[], filters: any) => {
    let filtered = alertsData;
    
    // Filter by risk level
    if (filters.riskLevel !== 'all') {
      filtered = filtered.filter((alert: any) => alert.severity === filters.riskLevel);
    }
    
    // Filter by amount range
    if (filters.amount) {
      filtered = filtered.filter((alert: any) => {
        const amount = parseFloat(alert.amount?.replace(/[$,]/g, '') || '0');
        return amount >= filters.amount[0] && amount <= filters.amount[1];
      });
    }
    
    // Filter by status
    if (filters.status && filters.status.length > 0) {
      filtered = filtered.filter((alert: any) => filters.status.includes(alert.status));
    }
    
    setFilteredAlerts(filtered);
    
    // Filter transaction data based on date range
    let filteredTrans = transData;
    if (filters.dateRange !== 'all') {
      const days = filters.dateRange === '7days' ? 7 : filters.dateRange === '30days' ? 30 : 365;
      filteredTrans = transData.slice(-days);
    }
    setFilteredTransactionData(filteredTrans);
  };

  const handleFilterApply = (filters: any) => {
    setAppliedFilters(filters);
    applyFiltersToData(transactionData, alerts, filters);
  };

  const handleExport = (format: string, options: any) => {
    // Generate actual export data based on current filtered data
    const exportData = {
      alerts: filteredAlerts,
      transactions: filteredTransactionData,
      summary: {
        totalAlerts: filteredAlerts.length,
        highRiskAlerts: filteredAlerts.filter((a: any) => a.severity === 'high').length,
        totalTransactions: filteredTransactionData.reduce((sum: number, item: any) => sum + (item.transactions || 0), 0),
        fraudDetected: filteredAlerts.filter((a: any) => a.type === 'fraud').length
      },
      timestamp: new Date().toISOString(),
      format,
      options
    };
    
    // Simulate export download
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `fraudguard-export-${Date.now()}.${format === 'excel' ? 'xlsx' : format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleActionExecute = (actionType: string, alertData: any) => {
    // Simulate different fraud investigation actions
    switch (actionType) {
      case 'block':
        setAlerts(prev => prev.map(alert => 
          alert.id === alertData?.id 
            ? { ...alert, status: 'blocked', action: 'Account Blocked' }
            : alert
        ));
        break;
      case 'investigate':
        setAlerts(prev => prev.map(alert => 
          alert.id === alertData?.id 
            ? { ...alert, status: 'investigating', action: 'Under Investigation' }
            : alert
        ));
        break;
      case 'resolve':
        setAlerts(prev => prev.map(alert => 
          alert.id === alertData?.id 
            ? { ...alert, status: 'resolved', action: 'Case Resolved' }
            : alert
        ));
        break;
      case 'escalate':
        setAlerts(prev => prev.map(alert => 
          alert.id === alertData?.id 
            ? { ...alert, status: 'escalated', action: 'Escalated to L2' }
            : alert
        ));
        break;
      default:
        console.log('Action executed:', actionType, alertData);
    }
    
    // Re-apply filters to updated data
    applyFiltersToData(transactionData, alerts, appliedFilters);
  };

  const handleFilter = () => {
    setFilterModalOpen(true);
  };

  const handleExportAction = () => {
    setAlertsModalOpen(true);
  };

  const handleAlertClick = (alertData: any) => {
    setSelectedAlert(alertData);
    setActionsModalOpen(true);
  };

  // Calculate KPIs based on filtered data
  const totalTransactions = filteredTransactionData.reduce((sum, item) => sum + item.transactions, 0);
  const totalFraud = filteredTransactionData.reduce((sum, item) => sum + item.fraudulent, 0);
  const fraudRate = totalTransactions > 0 ? ((totalFraud / totalTransactions) * 100).toFixed(2) : '0';
  const preventedLoss = (totalFraud * 1250).toLocaleString(); // Assume avg $1,250 per fraud

  // Use filtered data for display
  const displayAlerts = filteredAlerts;
  const displayTransactionData = filteredTransactionData;

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
    },
    {
      id: 'advanced-grid',
      label: 'Advanced Data Grid',
      icon: <FilterList />,
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
                >
                  Filter
                </Button>
                <Button
                  startIcon={<Refresh />}
                  variant="outlined"
                  size="small"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  Refresh
                </Button>
                <Button
                  startIcon={<Assessment />}
                  variant="contained"
                  size="small"
                  onClick={() => setShowAdvancedGrid(true)}
                  sx={{ bgcolor: excelColors.primary.main }}
                >
                  Advanced Grid
                </Button>
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
                      <AreaChart data={displayTransactionData}>
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
                      {displayAlerts.map((alert) => (
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
          </Box>
        ) : (
          <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
                🚀 REVOLUTIONARY Advanced Data Grid
              </Typography>
              <Button
                variant="outlined"
                onClick={() => setShowAdvancedGrid(false)}
              >
                Back to Dashboard
              </Button>
            </Box>
            <AdvancedDataGrid
              data={alerts}
              columns={gridColumns}
              title="Fraud Alerts - Advanced Grid"
              enableInlineEdit={true}
              enableBulkEdit={true}
              enableFiltering={true}
              enableSorting={true}
              enableSelection={true}
              enableKeyboardShortcuts={true}
            />
          </Box>
        )}

        {/* REVOLUTIONARY Interactive Modals */}
        <RevolutionaryModal
          open={filterModalOpen}
          onClose={() => setFilterModalOpen(false)}
          title="Advanced Fraud Filters"
          type="filter"
          onApplyFilter={handleFilterApply}
        />
        
        <RevolutionaryModal
          open={alertsModalOpen}
          onClose={() => setAlertsModalOpen(false)}
          title="Export Dashboard Data"
          type="export"
          onExportData={handleExport}
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
          onActionExecute={handleActionExecute}
        />

        {/* Power User Hub */}
        <PowerUserHub />
      </Box>
    </Box>
  );
};

export default memo(ExcelDashboard);