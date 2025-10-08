/**
 * Excel-Style Analytics Dashboard
 * Comprehensive analytics and fraud investigation tools
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Avatar,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  styled,
  alpha,
  Paper,
} from '@mui/material';
import {
  Assessment,
  Security,
  TrendingUp,
  Search,
  FilterList,
  Download,
  Visibility,
  MoreVert,
  Person,
  AttachMoney,
  LocationOn,
  Schedule,
  Warning,
  CheckCircle,
  Block,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
} from 'recharts';
import { excelColors } from '../theme/excelTheme';

// Styled components
const AnalyticsCard = styled(Card)({
  backgroundColor: excelColors.background.paper,
  border: `1px solid ${excelColors.background.border}`,
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
});

const TabPanel = (props: { children?: React.ReactNode; value: number; index: number }) => {
  const { children, value, index } = props;
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
};

// Mock data
const fraudTrendData = [
  { month: 'Jan', detected: 245, prevented: 198, loss: 47000 },
  { month: 'Feb', detected: 289, prevented: 231, loss: 58000 },
  { month: 'Mar', detected: 312, prevented: 267, loss: 45000 },
  { month: 'Apr', detected: 334, prevented: 289, loss: 67000 },
  { month: 'May', detected: 278, prevented: 245, loss: 33000 },
  { month: 'Jun', detected: 356, prevented: 312, loss: 44000 },
];

const riskProfileData = [
  { subject: 'Transaction Amount', A: 120, B: 110, fullMark: 150 },
  { subject: 'User Behavior', A: 98, B: 130, fullMark: 150 },
  { subject: 'Location Risk', A: 86, B: 130, fullMark: 150 },
  { subject: 'Device Trust', A: 99, B: 100, fullMark: 150 },
  { subject: 'Time Pattern', A: 85, B: 90, fullMark: 150 },
  { subject: 'Network Score', A: 65, B: 85, fullMark: 150 },
];

const fraudCaseData = [
  {
    id: 'FR-2024-001',
    type: 'Credit Card Fraud',
    amount: '$3,250.00',
    status: 'confirmed',
    investigator: 'Sarah Johnson',
    riskScore: 95,
    dateCreated: '2024-09-28',
    location: 'New York, NY',
    victim: 'john.doe@email.com',
  },
  {
    id: 'FR-2024-002',
    type: 'Identity Theft',
    amount: '$1,850.00',
    status: 'investigating',
    investigator: 'Mike Chen',
    riskScore: 87,
    dateCreated: '2024-09-27',
    location: 'Los Angeles, CA',
    victim: 'jane.smith@email.com',
  },
  {
    id: 'FR-2024-003',
    type: 'Account Takeover',
    amount: '$4,120.00',
    status: 'pending',
    investigator: 'Emma Wilson',
    riskScore: 92,
    dateCreated: '2024-09-26',
    location: 'Chicago, IL',
    victim: 'bob.johnson@email.com',
  },
];

const ExcelAnalytics: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [filterPeriod, setFilterPeriod] = useState('last-30-days');
  const [searchTerm, setSearchTerm] = useState('');

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleExport = () => {
    window.alert('Exporting Analytics Data - This would export current analytics data based on selected time period and filters to CSV/Excel format');
    console.log('Export analytics data triggered');
  };

  const handleCaseClick = (caseItem: any) => {
    window.alert(`Case Details:\nID: ${caseItem.id}\nType: ${caseItem.type}\nAmount: ${caseItem.amount}\nStatus: ${caseItem.status}\nRisk Score: ${caseItem.risk_score}`);
    console.log('Case clicked:', caseItem);
  };

  const handleCaseAction = (caseId: string, action: string) => {
    window.alert(`${action} action performed on Case ${caseId}`);
    console.log(`Case ${caseId} - ${action} action`);
  };

  const getStatusChip = (status: string) => {
    const configs = {
      confirmed: { label: 'Confirmed Fraud', color: excelColors.error.main },
      investigating: { label: 'Under Investigation', color: excelColors.warning.main },
      pending: { label: 'Pending Review', color: excelColors.info.main },
      resolved: { label: 'Resolved', color: excelColors.success.main },
      false_positive: { label: 'False Positive', color: excelColors.text.secondary },
    };
    
    const config = configs[status as keyof typeof configs] || configs.pending;
    
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

  const getRiskScoreColor = (score: number) => {
    if (score >= 90) return excelColors.error.main;
    if (score >= 70) return excelColors.warning.main;
    if (score >= 50) return excelColors.accent.yellow;
    return excelColors.success.main;
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ color: excelColors.text.primary, fontWeight: 600 }}>
          Fraud Analytics
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel>Time Period</InputLabel>
            <Select
              value={filterPeriod}
              label="Time Period"
              onChange={(e) => setFilterPeriod(e.target.value)}
            >
              <MenuItem value="last-7-days">Last 7 Days</MenuItem>
              <MenuItem value="last-30-days">Last 30 Days</MenuItem>
              <MenuItem value="last-90-days">Last 90 Days</MenuItem>
              <MenuItem value="last-year">Last Year</MenuItem>
            </Select>
          </FormControl>
          <Button
            startIcon={<Download />}
            variant="outlined"
            size="small"
            onClick={handleExport}
            sx={{ borderColor: excelColors.background.border }}
          >
            Export
          </Button>
        </Box>
      </Box>

      {/* Navigation Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Overview" />
          <Tab label="Fraud Trends" />
          <Tab label="Risk Analysis" />
          <Tab label="Case Management" />
        </Tabs>
      </Box>

      {/* Overview Tab */}
      <TabPanel value={tabValue} index={0}>
        <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
          {/* Summary Cards */}
          <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Assessment sx={{ color: excelColors.primary.main, fontSize: 40, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 600, color: excelColors.text.primary }}>
                1,247
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Cases Analyzed
              </Typography>
            </Paper>
          </Box>
          <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Security sx={{ color: excelColors.error.main, fontSize: 40, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 600, color: excelColors.error.main }}>
                89
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Confirmed Fraud
              </Typography>
            </Paper>
          </Box>
          <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <AttachMoney sx={{ color: excelColors.success.main, fontSize: 40, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 600, color: excelColors.success.main }}>
                $2.3M
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Loss Prevented
              </Typography>
            </Paper>
          </Box>
          <Box sx={{ flex: '1 1 200px', minWidth: 200 }}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <TrendingUp sx={{ color: excelColors.accent.blue, fontSize: 40, mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 600, color: excelColors.accent.blue }}>
                94.2%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Detection Rate
              </Typography>
            </Paper>
          </Box>
        </Box>

        {/* Charts */}
        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          <Box sx={{ flex: '1 1 400px', minWidth: 400 }}>
            <AnalyticsCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Fraud Detection Trends
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={fraudTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="detected" stroke={excelColors.primary.main} name="Detected" />
                    <Line type="monotone" dataKey="prevented" stroke={excelColors.success.main} name="Prevented" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </AnalyticsCard>
          </Box>

          <Box sx={{ flex: '1 1 300px', minWidth: 300 }}>
            <AnalyticsCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Risk Profile Analysis
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={riskProfileData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis />
                    <Radar name="Current Period" dataKey="A" stroke={excelColors.primary.main} fill={alpha(excelColors.primary.main, 0.2)} />
                    <Radar name="Previous Period" dataKey="B" stroke={excelColors.accent.orange} fill={alpha(excelColors.accent.orange, 0.2)} />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </AnalyticsCard>
          </Box>
        </Box>
      </TabPanel>

      {/* Fraud Trends Tab */}
      <TabPanel value={tabValue} index={1}>
        <AnalyticsCard>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3 }}>
              Monthly Fraud Analysis
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={fraudTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="detected" fill={excelColors.primary.main} name="Cases Detected" />
                <Bar yAxisId="left" dataKey="prevented" fill={excelColors.success.main} name="Cases Prevented" />
                <Line type="monotone" dataKey="loss" stroke={excelColors.error.main} yAxisId="right" name="Financial Loss ($)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </AnalyticsCard>
      </TabPanel>

      {/* Risk Analysis Tab */}
      <TabPanel value={tabValue} index={2}>
        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          <Box sx={{ flex: '1 1 400px', minWidth: 400 }}>
            <AnalyticsCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Risk Score Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Low Risk (0-49)', value: 65, color: excelColors.success.main },
                        { name: 'Medium Risk (50-69)', value: 20, color: excelColors.accent.yellow },
                        { name: 'High Risk (70-89)', value: 12, color: excelColors.warning.main },
                        { name: 'Critical Risk (90-100)', value: 3, color: excelColors.error.main },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      dataKey="value"
                    >
                      {[
                        { name: 'Low Risk (0-49)', value: 65, color: excelColors.success.main },
                        { name: 'Medium Risk (50-69)', value: 20, color: excelColors.accent.yellow },
                        { name: 'High Risk (70-89)', value: 12, color: excelColors.warning.main },
                        { name: 'Critical Risk (90-100)', value: 3, color: excelColors.error.main },
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </AnalyticsCard>
          </Box>

          <Box sx={{ flex: '1 1 400px', minWidth: 400 }}>
            <AnalyticsCard>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Risk Factors Analysis
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={riskProfileData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis angle={90} domain={[0, 150]} />
                    <Radar name="Risk Profile" dataKey="A" stroke={excelColors.primary.main} fill={alpha(excelColors.primary.main, 0.3)} />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </AnalyticsCard>
          </Box>
        </Box>
      </TabPanel>

      {/* Case Management Tab */}
      <TabPanel value={tabValue} index={3}>
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              size="small"
              placeholder="Search cases..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: excelColors.text.secondary }} />,
              }}
              sx={{ flexGrow: 1, maxWidth: 400 }}
            />
            <Button
              startIcon={<FilterList />}
              variant="outlined"
              size="small"
              onClick={() => window.alert('Case Filters:\n- By Status (New, Under Investigation, Resolved)\n- By Risk Level (High, Medium, Low)\n- By Date Range\n- By Amount Range\n- By Case Type')}
              sx={{ borderColor: excelColors.background.border }}
            >
              Filter
            </Button>
          </Box>
        </Box>

        <AnalyticsCard>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Active Fraud Cases
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Case ID</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Amount</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Investigator</TableCell>
                    <TableCell>Risk Score</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {fraudCaseData.map((case_) => (
                    <TableRow 
                      key={case_.id} 
                      hover 
                      onClick={() => handleCaseClick(case_)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                          {case_.id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Security sx={{ mr: 1, fontSize: 16, color: excelColors.error.main }} />
                          {case_.type}
                        </Box>
                      </TableCell>
                      <TableCell sx={{ fontWeight: 500 }}>{case_.amount}</TableCell>
                      <TableCell>{getStatusChip(case_.status)}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Avatar sx={{ width: 24, height: 24, mr: 1, fontSize: '0.75rem' }}>
                            {case_.investigator.charAt(0)}
                          </Avatar>
                          {case_.investigator}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={case_.riskScore}
                          size="small"
                          sx={{
                            backgroundColor: alpha(getRiskScoreColor(case_.riskScore), 0.1),
                            color: getRiskScoreColor(case_.riskScore),
                            border: `1px solid ${getRiskScoreColor(case_.riskScore)}`,
                            fontWeight: 600,
                          }}
                        />
                      </TableCell>
                      <TableCell>{case_.dateCreated}</TableCell>
                      <TableCell>
                        <IconButton 
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCaseAction(case_.id, 'View Details');
                          }}
                        >
                          <Visibility fontSize="small" />
                        </IconButton>
                        <IconButton 
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCaseAction(case_.id, 'More Actions');
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
        </AnalyticsCard>
      </TabPanel>
    </Box>
  );
};

export default ExcelAnalytics;