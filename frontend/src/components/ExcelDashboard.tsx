/**
 * Revolutionary Excel-Style Dashboard - FIXED VERSION
 * Comprehensive fraud detection dashboard with revolutionary interactive components
 */

import React, { memo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Avatar,
  Button,
  styled,
  alpha,
} from '@mui/material';
import {
  Security,
  Warning,
  CheckCircle,
  AttachMoney,
  LocationOn,
  Schedule,
  Visibility,
  FilterList,
  Refresh,
  MoreVert,
  Assessment,
} from '@mui/icons-material';
import { AnimatedKPICard } from './AnimatedComponents';
import {
  AreaChart,
  Area,
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

const generateRiskDistribution = () => [
  { name: 'Low Risk', value: 65, color: excelColors.success.main },
  { name: 'Medium Risk', value: 25, color: excelColors.warning.main },
  { name: 'High Risk', value: 10, color: excelColors.error.main },
];

interface ExcelDashboardProps {
  transactionData: any[];
  alerts: any[];
  isRefreshing: boolean;
  onRefresh: () => void;
  onFilter: () => void;
  onExport: () => void;
  onAlertClick: (alert: any) => void;
  onShowAdvancedGrid: (show: boolean) => void;
}

const ExcelDashboard: React.FC<ExcelDashboardProps> = ({
  transactionData,
  alerts,
  isRefreshing,
  onRefresh,
  onFilter,
  onExport,
  onAlertClick,
  onShowAdvancedGrid,
}) => {
  const riskData = generateRiskDistribution();

  // Calculate KPIs based on passed data
  const totalTransactions = transactionData.reduce((sum, item) => sum + item.transactions, 0);
  const totalFraud = transactionData.reduce((sum, item) => sum + item.fraudulent, 0);
  const fraudRate = totalTransactions > 0 ? ((totalFraud / totalTransactions) * 100).toFixed(2) : '0';
  const preventedLoss = (totalFraud * 1250).toLocaleString(); // Assume avg $1,250 per fraud

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

  return (
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
            onClick={onFilter}
          >
            Filter
          </Button>
          <Button
            startIcon={<Refresh />}
            variant="outlined"
            size="small"
            onClick={onRefresh}
            disabled={isRefreshing}
          >
            Refresh
          </Button>
          <Button
            startIcon={<Assessment />}
            variant="contained"
            size="small"
            onClick={() => onShowAdvancedGrid(true)}
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
            onClick={onFilter}
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
            onClick={onExport}
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
            onClick={() => { alert('Chart modal integration pending.'); }}
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
            onClick={() => { alert('Actions modal integration pending.'); }}
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
                  onClick={() => { alert('Chart modal integration pending.'); }}
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
              onClick={onExport} // Changed to use onExport for viewing all alerts
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
                    onClick={() => onAlertClick(alert)}
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
                          onAlertClick(alert);
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
  );
};

export default memo(ExcelDashboard);