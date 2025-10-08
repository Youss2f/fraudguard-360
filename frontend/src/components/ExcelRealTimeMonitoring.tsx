/**
 * Excel-Style Real-Time Monitoring
 * Comprehensive real-time monitoring dashboard with live data feeds
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  IconButton,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Avatar,
  LinearProgress,
  Alert,
  styled,
  alpha,
  Paper,
  Divider,
} from '@mui/material';
import {
  MonitorHeart,
  Security,
  Warning,
  CheckCircle,
  Error,
  TrendingUp,
  TrendingDown,
  Refresh,
  FilterList,
  Visibility,
  MoreVert,
  PlayArrow,
  Pause,
  Stop,
  NetworkCheck,
  Storage,
  Memory,
  Speed,
  Timeline,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { excelColors } from '../theme/excelTheme';

// Styled components
const MonitoringCard = styled(Card)({
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  '&:hover': {
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
  },
});

const LiveIndicator = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'status',
})<{ status: 'active' | 'warning' | 'error' }>(({ status }) => {
  const colors = {
    active: excelColors.success.main,
    warning: excelColors.warning.main,
    error: excelColors.error.main,
  };
  
  return {
    width: 12,
    height: 12,
    borderRadius: '50%',
    backgroundColor: colors[status],
    animation: 'pulse 2s infinite',
    '@keyframes pulse': {
      '0%': { opacity: 1 },
      '50%': { opacity: 0.5 },
      '100%': { opacity: 1 },
    },
  };
});

const MetricCard = styled(Paper)({
  padding: '16px',
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  textAlign: 'center',
});

// Mock data generators
const generateRealTimeData = () => {
  const data = [];
  const now = Date.now();
  for (let i = 59; i >= 0; i--) {
    const time = new Date(now - i * 1000);
    data.push({
      time: time.toLocaleTimeString(),
      transactions: Math.floor(Math.random() * 100) + 50,
      threats: Math.floor(Math.random() * 10) + 1,
      blocked: Math.floor(Math.random() * 5) + 1,
      cpu: Math.floor(Math.random() * 20) + 60,
      memory: Math.floor(Math.random() * 15) + 70,
      network: Math.floor(Math.random() * 30) + 40,
    });
  }
  return data;
};

const generateActiveThreats = () => [
  {
    id: 1,
    type: 'Brute Force Attack',
    source: '192.168.1.45',
    target: 'Login System',
    severity: 'high' as const,
    status: 'active',
    firstSeen: '30 sec ago',
    attempts: 127,
  },
  {
    id: 2,
    type: 'Suspicious Pattern',
    source: 'User: mike.johnson@company.com',
    target: 'Transaction Processing',
    severity: 'medium' as const,
    status: 'investigating',
    firstSeen: '2 min ago',
    attempts: 15,
  },
  {
    id: 3,
    type: 'Anomalous Behavior',
    source: 'Device: Mobile-iPhone-12',
    target: 'Account Access',
    severity: 'low' as const,
    status: 'monitoring',
    firstSeen: '5 min ago',
    attempts: 3,
  },
];

const ExcelRealTimeMonitoring: React.FC = () => {
  const [realTimeData, setRealTimeData] = useState(generateRealTimeData());
  const [activeThreats, setActiveThreats] = useState(generateActiveThreats());
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Simulate real-time updates
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isMonitoring && autoRefresh) {
      interval = setInterval(() => {
        setRealTimeData(prev => {
          const newData = [...prev];
          newData.shift(); // Remove first item
          const now = new Date();
          newData.push({
            time: now.toLocaleTimeString(),
            transactions: Math.floor(Math.random() * 100) + 50,
            threats: Math.floor(Math.random() * 10) + 1,
            blocked: Math.floor(Math.random() * 5) + 1,
            cpu: Math.floor(Math.random() * 20) + 60,
            memory: Math.floor(Math.random() * 15) + 70,
            network: Math.floor(Math.random() * 30) + 40,
          });
          return newData;
        });
      }, 2000); // Update every 2 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isMonitoring, autoRefresh]);

  const handleToggleMonitoring = () => {
    setIsMonitoring(!isMonitoring);
  };

  const handleToggleAutoRefresh = () => {
    setAutoRefresh(!autoRefresh);
  };

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
      active: { label: 'Active', color: excelColors.error.main },
      investigating: { label: 'Investigating', color: excelColors.warning.main },
      monitoring: { label: 'Monitoring', color: excelColors.info.main },
      resolved: { label: 'Resolved', color: excelColors.success.main },
    };
    
    const config = configs[status as keyof typeof configs] || configs.monitoring;
    
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

  const currentData = realTimeData[realTimeData.length - 1] || {};
  const avgTransactions = Math.floor(realTimeData.reduce((sum, item) => sum + (item.transactions || 0), 0) / realTimeData.length);

  return (
    <Box sx={{ p: 3 }}>
      {/* Header with controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
          Real-Time Monitoring
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
            <LiveIndicator status={isMonitoring ? 'active' : 'error'} />
            <Typography variant="body2" sx={{ ml: 1, color: excelColors.text.secondary }}>
              {isMonitoring ? 'Live' : 'Paused'}
            </Typography>
          </Box>
          <Button
            startIcon={isMonitoring ? <Pause /> : <PlayArrow />}
            variant="outlined"
            size="small"
            onClick={handleToggleMonitoring}
            sx={{ borderColor: excelColors.background.border }}
          >
            {isMonitoring ? 'Pause' : 'Resume'}
          </Button>
          <Button
            startIcon={<Refresh />}
            variant="outlined"
            size="small"
            onClick={handleToggleAutoRefresh}
            color={autoRefresh ? 'primary' : 'inherit'}
            sx={{ borderColor: excelColors.background.border }}
          >
            Auto Refresh
          </Button>
          <Button
            startIcon={<FilterList />}
            variant="outlined"
            size="small"
            onClick={() => window.alert('Monitoring Filters:\n- By Threat Level (High, Medium, Low)\n- By Threat Type (Fraud, Intrusion, Policy Violation)\n- By Time Range\n- By Source\n- By Status (Active, Resolved, Investigating)')}
            sx={{ borderColor: excelColors.background.border }}
          >
            Filter
          </Button>
        </Box>
      </Box>

      {/* System Status Alert */}
      {activeThreats.filter(t => t.severity === 'high').length > 0 && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
          action={
            <Button 
              size="small" 
              color="inherit"
              onClick={() => window.alert('High Severity Threats Details:\n- Critical fraud attempts detected\n- Real-time response required\n- Escalation protocols activated\n- Security teams notified')}
            >
              View Details
            </Button>
          }
        >
          High severity threats detected! {activeThreats.filter(t => t.severity === 'high').length} active threat(s) require immediate attention.
        </Alert>
      )}

      {/* Real-time metrics */}
      <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
        <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
          <MetricCard>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
              <MonitorHeart sx={{ color: excelColors.primary.main, mr: 1 }} />
              <Typography variant="h6" sx={{ color: excelColors.text.secondary }}>
                Transactions/Min
              </Typography>
            </Box>
            <Typography variant="h3" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
              {currentData.transactions || 0}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 1 }}>
              <TrendingUp sx={{ color: excelColors.success.main, fontSize: 16, mr: 0.5 }} />
              <Typography variant="body2" sx={{ color: excelColors.success.main }}>
                Avg: {avgTransactions}
              </Typography>
            </Box>
          </MetricCard>
        </Box>

        <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
          <MetricCard>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
              <Security sx={{ color: excelColors.error.main, mr: 1 }} />
              <Typography variant="h6" sx={{ color: excelColors.text.secondary }}>
                Active Threats
              </Typography>
            </Box>
            <Typography variant="h3" sx={{ color: excelColors.error.main, fontWeight: 600 }}>
              {activeThreats.length}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 1 }}>
              <Warning sx={{ color: excelColors.warning.main, fontSize: 16, mr: 0.5 }} />
              <Typography variant="body2" sx={{ color: excelColors.warning.main }}>
                {activeThreats.filter(t => t.severity === 'high').length} High Priority
              </Typography>
            </Box>
          </MetricCard>
        </Box>

        <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
          <MetricCard>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
              <Memory sx={{ color: excelColors.accent.blue, mr: 1 }} />
              <Typography variant="h6" sx={{ color: excelColors.text.secondary }}>
                System Load
              </Typography>
            </Box>
            <Typography variant="h3" sx={{ color: excelColors.accent.blue, fontWeight: 600 }}>
              {currentData.cpu || 0}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={currentData.cpu || 0}
              sx={{
                mt: 1,
                height: 6,
                borderRadius: 3,
                backgroundColor: alpha(excelColors.accent.blue, 0.1),
                '& .MuiLinearProgress-bar': {
                  backgroundColor: excelColors.accent.blue,
                },
              }}
            />
          </MetricCard>
        </Box>

        <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
          <MetricCard>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
              <CheckCircle sx={{ color: excelColors.success.main, mr: 1 }} />
              <Typography variant="h6" sx={{ color: excelColors.text.secondary }}>
                Blocked Attacks
              </Typography>
            </Box>
            <Typography variant="h3" sx={{ color: excelColors.success.main, fontWeight: 600 }}>
              {currentData.blocked || 0}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 1 }}>
              <TrendingUp sx={{ color: excelColors.success.main, fontSize: 16, mr: 0.5 }} />
              <Typography variant="body2" sx={{ color: excelColors.success.main }}>
                +{Math.floor(Math.random() * 10)}% this hour
              </Typography>
            </Box>
          </MetricCard>
        </Box>
      </Box>

      {/* Charts Section */}
      <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
        {/* Real-time Activity Chart */}
        <Box sx={{ flex: '2 1 600px', minWidth: 600 }}>
          <MonitoringCard>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
                  Live Activity Feed (Last 60 seconds)
                </Typography>
                <IconButton 
                  size="small"
                  onClick={() => window.alert('Live Activity Options:\n- Full Screen View\n- Export Data\n- Configure Alerts\n- Historical View\n- Customization Settings')}
                >
                  <MoreVert />
                </IconButton>
              </Box>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={realTimeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={excelColors.background.border} />
                  <XAxis dataKey="time" stroke={excelColors.text.secondary} fontSize={10} />
                  <YAxis stroke={excelColors.text.secondary} fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: excelColors.background.paper,
                      border: `1px solid ${excelColors.background.border}`,
                      borderRadius: 4,
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="transactions"
                    stroke={excelColors.primary.main}
                    strokeWidth={2}
                    dot={false}
                    name="Transactions"
                  />
                  <Line
                    type="monotone"
                    dataKey="threats"
                    stroke={excelColors.error.main}
                    strokeWidth={2}
                    dot={false}
                    name="Threats"
                  />
                  <Line
                    type="monotone"
                    dataKey="blocked"
                    stroke={excelColors.success.main}
                    strokeWidth={2}
                    dot={false}
                    name="Blocked"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </MonitoringCard>
        </Box>

        {/* System Performance */}
        <Box sx={{ flex: '1 1 300px', minWidth: 300 }}>
          <MonitoringCard>
            <CardContent>
              <Typography variant="h6" sx={{ color: excelColors.text.primary, mb: 2 }}>
                System Performance
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={realTimeData.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke={excelColors.background.border} />
                  <XAxis dataKey="time" stroke={excelColors.text.secondary} fontSize={10} />
                  <YAxis stroke={excelColors.text.secondary} fontSize={12} />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="cpu"
                    stackId="1"
                    stroke={excelColors.accent.blue}
                    fill={alpha(excelColors.accent.blue, 0.3)}
                    name="CPU %"
                  />
                  <Area
                    type="monotone"
                    dataKey="memory"
                    stackId="2"
                    stroke={excelColors.accent.orange}
                    fill={alpha(excelColors.accent.orange, 0.3)}
                    name="Memory %"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </MonitoringCard>
        </Box>
      </Box>

      {/* Active Threats Table */}
      <MonitoringCard>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
              Active Security Threats
            </Typography>
            <Button 
              variant="outlined" 
              size="small" 
              startIcon={<Visibility />}
              onClick={() => window.alert('View All Threats:\n- Active threat investigation\n- Historical threat analysis\n- Threat pattern recognition\n- Response management\n- Escalation procedures')}
            >
              View All Threats
            </Button>
          </Box>
          
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Threat Type</TableCell>
                  <TableCell>Source</TableCell>
                  <TableCell>Target</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>First Seen</TableCell>
                  <TableCell>Attempts</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {activeThreats.map((threat) => (
                  <TableRow key={threat.id} hover>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <LiveIndicator 
                          status={threat.severity === 'high' ? 'error' : threat.severity === 'medium' ? 'warning' : 'active'} 
                        />
                        <Typography sx={{ ml: 1, fontWeight: 500 }}>
                          {threat.type}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {threat.source}
                      </Typography>
                    </TableCell>
                    <TableCell>{threat.target}</TableCell>
                    <TableCell>
                      <Chip
                        label={threat.severity.toUpperCase()}
                        size="small"
                        sx={{
                          backgroundColor: alpha(getSeverityColor(threat.severity), 0.1),
                          color: getSeverityColor(threat.severity),
                          border: `1px solid ${getSeverityColor(threat.severity)}`,
                          fontSize: '0.75rem',
                          fontWeight: 600,
                        }}
                      />
                    </TableCell>
                    <TableCell>{getStatusChip(threat.status)}</TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ color: excelColors.text.secondary }}>
                        {threat.firstSeen}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {threat.attempts}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <IconButton 
                        size="small"
                        onClick={() => window.alert(`Threat Actions for: ${threat.type}\n- Investigate Threat\n- Block Source\n- Create Incident\n- Escalate to Security Team\n- Mark as False Positive`)}
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
      </MonitoringCard>
    </Box>
  );
};

export default ExcelRealTimeMonitoring;