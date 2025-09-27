/**
 * FraudGuard 360 Classic Dashboard
 * Excel 2010 style professional interface with Power BI visualizations
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Divider,
  LinearProgress,
} from '@mui/material';
import { AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

// Types
interface FraudMetrics {
  // Core Fraud Detection
  totalCalls: number;
  fraudDetected: number;
  riskScore: number;
  blockedCalls: number;
  falsePositives: number;
  accuracy: number;
  
  // Enterprise Performance Metrics (NFRs)
  ingestionTPS: number;
  processingLatencyP95: number;
  systemAvailability: number;
  mlInferenceLatency: number;
  
  // Microservices Health
  kafkaHealth: number;
  flinkJobsRunning: number;
  neo4jConnections: number;
  redisHitRate: number;
  sagemakerEndpoints: number;
  
  // Advanced Analytics
  fraudRingsDetected: number;
  coordinatedAttacks: number;
  graphSageAccuracy: number;
}

interface FraudAlert {
  id: string;
  phoneNumber: string;
  riskScore: number;
  location: string;
  description: string;
  timestamp: string;
  status: 'Active' | 'Resolved' | 'Investigating';
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
}

interface TimeSeriesData {
  time: string;
  calls: number;
  frauds: number;
  blocked: number;
}

const ClassicDashboard: React.FC = () => {
  const [selectedView, setSelectedView] = useState('overview');
  const [selectedPeriod, setSelectedPeriod] = useState('today');
  const [refreshing, setRefreshing] = useState(false);
  
  // Dashboard data
  const [metrics, setMetrics] = useState<FraudMetrics>({
    // Core Fraud Detection
    totalCalls: 0,
    fraudDetected: 0,
    riskScore: 0,
    blockedCalls: 0,
    falsePositives: 0,
    accuracy: 0,
    
    // Enterprise Performance Metrics (NFRs)
    ingestionTPS: 0,
    processingLatencyP95: 0,
    systemAvailability: 0,
    mlInferenceLatency: 0,
    
    // Microservices Health
    kafkaHealth: 0,
    flinkJobsRunning: 0,
    neo4jConnections: 0,
    redisHitRate: 0,
    sagemakerEndpoints: 0,
    
    // Advanced Analytics
    fraudRingsDetected: 0,
    coordinatedAttacks: 0,
    graphSageAccuracy: 0,
  });

  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);

  // Generate realistic data
  useEffect(() => {
    const generateMetrics = () => {
      setMetrics({
        // Core Fraud Detection
        totalCalls: Math.floor(Math.random() * 5000) + 45000,
        fraudDetected: Math.floor(Math.random() * 50) + 89,
        riskScore: Math.floor(Math.random() * 15) + 15,
        blockedCalls: Math.floor(Math.random() * 100) + 156,
        falsePositives: Math.floor(Math.random() * 10) + 8,
        accuracy: Math.floor(Math.random() * 3) + 96,
        
        // Enterprise Performance Metrics (NFRs)
        ingestionTPS: Math.floor(Math.random() * 15000) + 8000, // 8k-23k TPS range
        processingLatencyP95: Math.floor(Math.random() * 100) + 120, // 120-220ms P95
        systemAvailability: 99.90 + Math.random() * 0.09, // 99.90-99.99%
        mlInferenceLatency: Math.floor(Math.random() * 30) + 45, // 45-75ms
        
        // Microservices Health
        kafkaHealth: 95 + Math.floor(Math.random() * 5), // 95-100%
        flinkJobsRunning: Math.floor(Math.random() * 3) + 8, // 8-11 jobs
        neo4jConnections: Math.floor(Math.random() * 50) + 150, // 150-200 connections
        redisHitRate: 92 + Math.floor(Math.random() * 7), // 92-99%
        sagemakerEndpoints: Math.floor(Math.random() * 2) + 5, // 5-7 endpoints
        
        // Advanced Analytics
        fraudRingsDetected: Math.floor(Math.random() * 5) + 12, // 12-17 rings
        coordinatedAttacks: Math.floor(Math.random() * 8) + 3, // 3-11 attacks
        graphSageAccuracy: 94.5 + Math.random() * 4, // 94.5-98.5%
      });
    };

    const generateAlerts = () => {
      const locations = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ', 'Philadelphia, PA'];
      const descriptions = [
        'Robocall pattern detected - Mass dialing',
        'Caller ID spoofing identified',
        'High frequency calling pattern',
        'International fraud attempt',
        'Voice phishing (Vishing) detected',
        'Automated dialing system flagged'
      ];
      
      const newAlerts: FraudAlert[] = Array.from({ length: 12 }, (_, i) => ({
        id: `FRD-${String(Date.now() + i).slice(-6)}`,
        phoneNumber: `+1 (${Math.floor(Math.random() * 900) + 100}) ${Math.floor(Math.random() * 900) + 100}-${Math.floor(Math.random() * 9000) + 1000}`,
        riskScore: Math.floor(Math.random() * 100),
        location: locations[Math.floor(Math.random() * locations.length)],
        description: descriptions[Math.floor(Math.random() * descriptions.length)],
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 3600000)).toLocaleString(),
        status: ['Active', 'Resolved', 'Investigating'][Math.floor(Math.random() * 3)] as any,
        severity: ['Low', 'Medium', 'High', 'Critical'][Math.floor(Math.random() * 4)] as any,
      }));
      
      setAlerts(newAlerts);
    };

    const generateTimeSeriesData = () => {
      const data: TimeSeriesData[] = [];
      const now = new Date();
      
      for (let i = 23; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000);
        data.push({
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          calls: Math.floor(Math.random() * 500) + 200,
          frauds: Math.floor(Math.random() * 15) + 2,
          blocked: Math.floor(Math.random() * 20) + 5,
        });
      }
      
      setTimeSeriesData(data);
    };

    generateMetrics();
    generateAlerts();
    generateTimeSeriesData();

    const interval = setInterval(() => {
      generateMetrics();
      generateAlerts();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
      // Refresh data here
    }, 1000);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical': return '#e1665e';
      case 'High': return '#d6b656';
      case 'Medium': return '#4472c4';
      case 'Low': return '#70ad47';
      default: return '#7f7f7f';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Active': return '#e1665e';
      case 'Investigating': return '#d6b656';
      case 'Resolved': return '#70ad47';
      default: return '#7f7f7f';
    }
  };

  // Chart colors - Power BI style
  const chartColors = {
    primary: '#4472c4',
    secondary: '#70ad47',
    accent1: '#d6b656',
    accent2: '#e1665e',
    accent3: '#9e4d96',
    neutral: '#7f7f7f'
  };

  return (
    <Box sx={{ 
      minHeight: '100vh', 
      backgroundColor: '#f4f4f4',
      fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
    }}>
      {/* Classic Windows Menu Bar */}
      <Paper sx={{ 
        borderRadius: 0,
        border: '1px solid #d4d4d4',
        borderBottom: '2px solid #ababab',
        background: 'linear-gradient(to bottom, #ffffff 0%, #e5e5e5 100%)',
        mb: 0
      }}>
        <Box sx={{ p: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6" sx={{ 
            color: '#2b2b2b', 
            fontWeight: 'bold',
            fontSize: '14px',
            mr: 3
          }}>
            FraudGuard 360 - Professional Edition
          </Typography>
          
          <Button 
            variant={selectedView === 'overview' ? 'contained' : 'outlined'}
            onClick={() => setSelectedView('overview')}
            sx={{ minWidth: 80, fontSize: '11px' }}
          >
            Overview
          </Button>
          <Button 
            variant={selectedView === 'alerts' ? 'contained' : 'outlined'}
            onClick={() => setSelectedView('alerts')}
            sx={{ minWidth: 80, fontSize: '11px' }}
          >
            Alerts
          </Button>
          <Button 
            variant={selectedView === 'analysis' ? 'contained' : 'outlined'}
            onClick={() => setSelectedView('analysis')}
            sx={{ minWidth: 80, fontSize: '11px' }}
          >
            Analysis
          </Button>
          <Button 
            variant={selectedView === 'reports' ? 'contained' : 'outlined'}
            onClick={() => setSelectedView('reports')}
            sx={{ minWidth: 80, fontSize: '11px' }}
          >
            Reports
          </Button>
          <Button 
            variant={selectedView === 'system' ? 'contained' : 'outlined'}
            onClick={() => setSelectedView('system')}
            sx={{ minWidth: 80, fontSize: '11px' }}
          >
            System Health
          </Button>
          <Button 
            variant={selectedView === 'ml' ? 'contained' : 'outlined'}
            onClick={() => setSelectedView('ml')}
            sx={{ minWidth: 80, fontSize: '11px' }}
          >
            ML Pipeline
          </Button>

          <Box sx={{ flexGrow: 1 }} />

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel sx={{ fontSize: '11px' }}>Time Period</InputLabel>
            <Select
              value={selectedPeriod}
              label="Time Period"
              onChange={(e) => setSelectedPeriod(e.target.value)}
              sx={{ fontSize: '11px', height: 30 }}
            >
              <MenuItem value="today" sx={{ fontSize: '11px' }}>Today</MenuItem>
              <MenuItem value="week" sx={{ fontSize: '11px' }}>This Week</MenuItem>
              <MenuItem value="month" sx={{ fontSize: '11px' }}>This Month</MenuItem>
              <MenuItem value="quarter" sx={{ fontSize: '11px' }}>This Quarter</MenuItem>
            </Select>
          </FormControl>

          <Button 
            onClick={handleRefresh}
            disabled={refreshing}
            sx={{ fontSize: '11px', minWidth: 70 }}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
        </Box>
        {refreshing && <LinearProgress />}
      </Paper>

      {/* Main Content Area */}
      <Box sx={{ p: 2 }}>
        {selectedView === 'overview' && (
          <Grid container spacing={2}>
            {/* KPI Cards - Excel style */}
            <Grid item xs={12}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff'
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  KEY PERFORMANCE INDICATORS
                </Typography>
                
                <Grid container spacing={2}>
                  {/* Row 1: Core Fraud Metrics */}
                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#dae8fc'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.primary,
                        fontSize: '20px'
                      }}>
                        {metrics.totalCalls.toLocaleString()}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        TOTAL CALLS
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#f8cecc'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.accent2,
                        fontSize: '20px'
                      }}>
                        {metrics.fraudDetected}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        FRAUD DETECTED
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#fff2cc'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.accent1,
                        fontSize: '20px'
                      }}>
                        {metrics.ingestionTPS.toLocaleString()}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        INGESTION TPS
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#d5e8d4'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.secondary,
                        fontSize: '20px'
                      }}>
                        {metrics.processingLatencyP95}ms
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        LATENCY P95
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#d5e8d4'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.secondary,
                        fontSize: '20px'
                      }}>
                        {metrics.systemAvailability.toFixed(2)}%
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        AVAILABILITY
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#e6ccff'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.accent3,
                        fontSize: '20px'
                      }}>
                        {metrics.fraudRingsDetected}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        FRAUD RINGS
                      </Typography>
                    </Box>
                  </Grid>

                  {/* Row 2: Advanced Analytics */}
                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#f0f0f0'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.neutral,
                        fontSize: '20px'
                      }}>
                        {metrics.mlInferenceLatency}ms
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        ML INFERENCE
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#dae8fc'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.primary,
                        fontSize: '20px'
                      }}>
                        {metrics.flinkJobsRunning}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        FLINK JOBS
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#d5e8d4'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.secondary,
                        fontSize: '20px'
                      }}>
                        {metrics.redisHitRate}%
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        REDIS HIT RATE
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#fff2cc'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.accent1,
                        fontSize: '20px'
                      }}>
                        {metrics.neo4jConnections}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        NEO4J CONN
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#e6ccff'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.accent3,
                        fontSize: '20px'
                      }}>
                        {metrics.graphSageAccuracy.toFixed(1)}%
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        GRAPHSAGE
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6} md={2}>
                    <Box sx={{ 
                      textAlign: 'center',
                      p: 1.5,
                      border: '1px solid #e5e5e5',
                      backgroundColor: '#f8cecc'
                    }}>
                      <Typography variant="h5" sx={{ 
                        fontWeight: 'bold', 
                        color: chartColors.accent2,
                        fontSize: '20px'
                      }}>
                        {metrics.coordinatedAttacks}
                      </Typography>
                      <Typography variant="caption" sx={{ 
                        color: '#555555',
                        fontSize: '9px',
                        fontWeight: 'bold'
                      }}>
                        COORD ATTACKS
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>

            {/* Charts Row */}
            <Grid item xs={12} md={8}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff',
                height: 400
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  CALL ACTIVITY TRENDS - LAST 24 HOURS
                </Typography>
                
                <ResponsiveContainer width="100%" height={320}>
                  <AreaChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 10, fill: '#555555' }}
                      axisLine={{ stroke: '#d4d4d4' }}
                    />
                    <YAxis 
                      tick={{ fontSize: 10, fill: '#555555' }}
                      axisLine={{ stroke: '#d4d4d4' }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '1px solid #d4d4d4',
                        borderRadius: 0,
                        fontSize: '11px'
                      }}
                    />
                    <Legend 
                      wrapperStyle={{ fontSize: '11px' }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="calls" 
                      stackId="1" 
                      stroke={chartColors.primary}
                      fill={chartColors.primary}
                      fillOpacity={0.3}
                      name="Total Calls"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="frauds" 
                      stackId="2" 
                      stroke={chartColors.accent2}
                      fill={chartColors.accent2}
                      fillOpacity={0.7}
                      name="Fraud Detected"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="blocked" 
                      stackId="3" 
                      stroke={chartColors.accent1}
                      fill={chartColors.accent1}
                      fillOpacity={0.5}
                      name="Blocked Calls"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff',
                height: 400
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  FRAUD TYPE DISTRIBUTION
                </Typography>
                
                <ResponsiveContainer width="100%" height={320}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Robocalls', value: 45, color: chartColors.primary },
                        { name: 'Spoofing', value: 30, color: chartColors.accent2 },
                        { name: 'Phishing', value: 15, color: chartColors.accent1 },
                        { name: 'Other', value: 10, color: chartColors.secondary },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {[
                        { name: 'Robocalls', value: 45, color: chartColors.primary },
                        { name: 'Spoofing', value: 30, color: chartColors.accent2 },
                        { name: 'Phishing', value: 15, color: chartColors.accent1 },
                        { name: 'Other', value: 10, color: chartColors.secondary },
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '1px solid #d4d4d4',
                        borderRadius: 0,
                        fontSize: '11px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
          </Grid>
        )}

        {selectedView === 'alerts' && (
          <Paper sx={{ 
            border: '1px solid #d4d4d4',
            borderRadius: 0,
            background: '#ffffff'
          }}>
            <Box sx={{ p: 2, borderBottom: '1px solid #d4d4d4' }}>
              <Typography variant="subtitle1" sx={{ 
                fontWeight: 'bold',
                fontSize: '12px',
                color: '#2b2b2b'
              }}>
                ACTIVE FRAUD ALERTS
              </Typography>
            </Box>
            
            <TableContainer>
              <Table size="small">
                <TableHead sx={{ backgroundColor: '#f4f4f4' }}>
                  <TableRow>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Alert ID</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Phone Number</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Risk Score</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Location</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Description</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Severity</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Status</TableCell>
                    <TableCell sx={{ fontSize: '11px', fontWeight: 'bold', border: '1px solid #d4d4d4' }}>Timestamp</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {alerts.map((alert) => (
                    <TableRow 
                      key={alert.id}
                      sx={{ 
                        '&:hover': { backgroundColor: '#f8f8f8' },
                        '&:nth-of-type(even)': { backgroundColor: '#fafafa' }
                      }}
                    >
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5' }}>{alert.id}</TableCell>
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5', fontFamily: 'monospace' }}>{alert.phoneNumber}</TableCell>
                      <TableCell sx={{ 
                        fontSize: '10px', 
                        border: '1px solid #e5e5e5',
                        fontWeight: 'bold',
                        color: alert.riskScore > 70 ? chartColors.accent2 : alert.riskScore > 40 ? chartColors.accent1 : chartColors.secondary
                      }}>
                        {alert.riskScore}%
                      </TableCell>
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5' }}>{alert.location}</TableCell>
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5' }}>{alert.description}</TableCell>
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5' }}>
                        <Box sx={{ 
                          display: 'inline-block',
                          px: 1,
                          py: 0.5,
                          backgroundColor: getSeverityColor(alert.severity),
                          color: 'white',
                          fontSize: '9px',
                          fontWeight: 'bold'
                        }}>
                          {alert.severity.toUpperCase()}
                        </Box>
                      </TableCell>
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5' }}>
                        <Box sx={{ 
                          display: 'inline-block',
                          px: 1,
                          py: 0.5,
                          backgroundColor: getStatusColor(alert.status),
                          color: 'white',
                          fontSize: '9px',
                          fontWeight: 'bold'
                        }}>
                          {alert.status.toUpperCase()}
                        </Box>
                      </TableCell>
                      <TableCell sx={{ fontSize: '10px', border: '1px solid #e5e5e5', fontFamily: 'monospace' }}>{alert.timestamp}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        )}

        {selectedView === 'system' && (
          <Grid container spacing={2}>
            {/* Microservices Architecture Health */}
            <Grid item xs={12}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff'
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  MICROSERVICES ARCHITECTURE - SYSTEM HEALTH MONITORING
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ 
                      p: 2,
                      border: '1px solid #d4d4d4',
                      backgroundColor: '#f8f8f8'
                    }}>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 'bold',
                        fontSize: '11px',
                        mb: 2,
                        color: '#2b2b2b'
                      }}>
                        MESSAGE BROKER (KAFKA)
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Health Status:</Typography>
                        <Typography sx={{ 
                          fontSize: '10px', 
                          fontWeight: 'bold',
                          color: metrics.kafkaHealth > 95 ? chartColors.secondary : chartColors.accent2
                        }}>
                          {metrics.kafkaHealth}% UP
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Throughput:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold' }}>
                          {metrics.ingestionTPS.toLocaleString()} TPS
                        </Typography>
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={metrics.kafkaHealth} 
                        sx={{ 
                          height: 4, 
                          borderRadius: 2,
                          backgroundColor: '#e5e5e5',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: metrics.kafkaHealth > 95 ? chartColors.secondary : chartColors.accent2
                          }
                        }} 
                      />
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Box sx={{ 
                      p: 2,
                      border: '1px solid #d4d4d4',
                      backgroundColor: '#f8f8f8'
                    }}>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 'bold',
                        fontSize: '11px',
                        mb: 2,
                        color: '#2b2b2b'
                      }}>
                        STREAM PROCESSING (FLINK)
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Active Jobs:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold', color: chartColors.primary }}>
                          {metrics.flinkJobsRunning} Running
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Processing Latency:</Typography>
                        <Typography sx={{ 
                          fontSize: '10px', 
                          fontWeight: 'bold',
                          color: metrics.processingLatencyP95 < 200 ? chartColors.secondary : chartColors.accent1
                        }}>
                          {metrics.processingLatencyP95}ms P95
                        </Typography>
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={Math.min(100, (250 - metrics.processingLatencyP95) / 250 * 100)} 
                        sx={{ 
                          height: 4, 
                          borderRadius: 2,
                          backgroundColor: '#e5e5e5',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: metrics.processingLatencyP95 < 200 ? chartColors.secondary : chartColors.accent1
                          }
                        }} 
                      />
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Box sx={{ 
                      p: 2,
                      border: '1px solid #d4d4d4',
                      backgroundColor: '#f8f8f8'
                    }}>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 'bold',
                        fontSize: '11px',
                        mb: 2,
                        color: '#2b2b2b'
                      }}>
                        GRAPH DATABASE (NEO4J)
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Connections:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold', color: chartColors.accent1 }}>
                          {metrics.neo4jConnections} Active
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Query Performance:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold', color: chartColors.secondary }}>
                          Optimal
                        </Typography>
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={Math.min(100, metrics.neo4jConnections / 200 * 100)} 
                        sx={{ 
                          height: 4, 
                          borderRadius: 2,
                          backgroundColor: '#e5e5e5',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: chartColors.accent1
                          }
                        }} 
                      />
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Box sx={{ 
                      p: 2,
                      border: '1px solid #d4d4d4',
                      backgroundColor: '#f8f8f8'
                    }}>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 'bold',
                        fontSize: '11px',
                        mb: 2,
                        color: '#2b2b2b'
                      }}>
                        IN-MEMORY CACHE (REDIS)
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Hit Rate:</Typography>
                        <Typography sx={{ 
                          fontSize: '10px', 
                          fontWeight: 'bold',
                          color: metrics.redisHitRate > 95 ? chartColors.secondary : chartColors.accent1
                        }}>
                          {metrics.redisHitRate}%
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Memory Usage:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold', color: chartColors.primary }}>
                          78% Used
                        </Typography>
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={metrics.redisHitRate} 
                        sx={{ 
                          height: 4, 
                          borderRadius: 2,
                          backgroundColor: '#e5e5e5',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: metrics.redisHitRate > 95 ? chartColors.secondary : chartColors.accent1
                          }
                        }} 
                      />
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Box sx={{ 
                      p: 2,
                      border: '1px solid #d4d4d4',
                      backgroundColor: '#f8f8f8'
                    }}>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 'bold',
                        fontSize: '11px',
                        mb: 2,
                        color: '#2b2b2b'
                      }}>
                        ML INFRASTRUCTURE (SAGEMAKER)
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Endpoints:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold', color: chartColors.accent3 }}>
                          {metrics.sagemakerEndpoints} Active
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Inference Latency:</Typography>
                        <Typography sx={{ 
                          fontSize: '10px', 
                          fontWeight: 'bold',
                          color: metrics.mlInferenceLatency < 60 ? chartColors.secondary : chartColors.accent1
                        }}>
                          {metrics.mlInferenceLatency}ms
                        </Typography>
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={Math.min(100, (100 - metrics.mlInferenceLatency) / 100 * 100)} 
                        sx={{ 
                          height: 4, 
                          borderRadius: 2,
                          backgroundColor: '#e5e5e5',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: chartColors.accent3
                          }
                        }} 
                      />
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Box sx={{ 
                      p: 2,
                      border: '1px solid #d4d4d4',
                      backgroundColor: '#f8f8f8'
                    }}>
                      <Typography variant="subtitle2" sx={{ 
                        fontWeight: 'bold',
                        fontSize: '11px',
                        mb: 2,
                        color: '#2b2b2b'
                      }}>
                        SYSTEM AVAILABILITY (NFR)
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>Current Uptime:</Typography>
                        <Typography sx={{ 
                          fontSize: '10px', 
                          fontWeight: 'bold',
                          color: metrics.systemAvailability > 99.9 ? chartColors.secondary : chartColors.accent2
                        }}>
                          {metrics.systemAvailability.toFixed(3)}%
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontSize: '10px' }}>SLA Target:</Typography>
                        <Typography sx={{ fontSize: '10px', fontWeight: 'bold', color: chartColors.neutral }}>
                          99.900%
                        </Typography>
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={Math.min(100, metrics.systemAvailability)} 
                        sx={{ 
                          height: 4, 
                          borderRadius: 2,
                          backgroundColor: '#e5e5e5',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: metrics.systemAvailability > 99.9 ? chartColors.secondary : chartColors.accent2
                          }
                        }} 
                      />
                    </Box>
                  </Grid>

                </Grid>
              </Paper>
            </Grid>

            {/* Event-Driven Architecture Flow */}
            <Grid item xs={12}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff',
                height: 300
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  REAL-TIME DATA PIPELINE PERFORMANCE
                </Typography>
                
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 10, fill: '#555555' }}
                      axisLine={{ stroke: '#d4d4d4' }}
                    />
                    <YAxis 
                      tick={{ fontSize: 10, fill: '#555555' }}
                      axisLine={{ stroke: '#d4d4d4' }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '1px solid #d4d4d4',
                        borderRadius: 0,
                        fontSize: '11px'
                      }}
                    />
                    <Legend 
                      wrapperStyle={{ fontSize: '11px' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="calls" 
                      stroke={chartColors.primary}
                      strokeWidth={2}
                      dot={{ fill: chartColors.primary, strokeWidth: 0, r: 3 }}
                      name="Ingestion Rate"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="frauds" 
                      stroke={chartColors.accent2}
                      strokeWidth={2}
                      dot={{ fill: chartColors.accent2, strokeWidth: 0, r: 3 }}
                      name="ML Processing"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="blocked" 
                      stroke={chartColors.secondary}
                      strokeWidth={2}
                      dot={{ fill: chartColors.secondary, strokeWidth: 0, r: 3 }}
                      name="Actions Taken"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
          </Grid>
        )}

        {selectedView === 'ml' && (
          <Grid container spacing={2}>
            {/* GraphSAGE Model Performance */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff',
                height: 350
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  GRAPHSAGE MODEL PERFORMANCE
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography sx={{ fontSize: '11px', fontWeight: 'bold' }}>Model Accuracy:</Typography>
                    <Typography sx={{ 
                      fontSize: '11px', 
                      fontWeight: 'bold',
                      color: chartColors.secondary
                    }}>
                      {metrics.graphSageAccuracy.toFixed(2)}%
                    </Typography>
                  </Box>
                  
                  <LinearProgress 
                    variant="determinate" 
                    value={metrics.graphSageAccuracy} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      backgroundColor: '#e5e5e5',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: chartColors.secondary
                      }
                    }} 
                  />
                </Box>

                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={[
                    { name: 'Precision', value: metrics.graphSageAccuracy - 2 },
                    { name: 'Recall', value: metrics.graphSageAccuracy - 1 },
                    { name: 'F1-Score', value: metrics.graphSageAccuracy - 0.5 },
                    { name: 'AUC-ROC', value: metrics.graphSageAccuracy + 1 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis 
                      dataKey="name" 
                      tick={{ fontSize: 10, fill: '#555555' }}
                      axisLine={{ stroke: '#d4d4d4' }}
                    />
                    <YAxis 
                      domain={[90, 100]}
                      tick={{ fontSize: 10, fill: '#555555' }}
                      axisLine={{ stroke: '#d4d4d4' }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '1px solid #d4d4d4',
                        borderRadius: 0,
                        fontSize: '11px'
                      }}
                    />
                    <Bar 
                      dataKey="value" 
                      fill={chartColors.accent3}
                      name="Performance %"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            {/* Fraud Ring Detection */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ 
                p: 2, 
                border: '1px solid #d4d4d4',
                borderRadius: 0,
                background: '#ffffff',
                height: 350
              }}>
                <Typography variant="subtitle1" sx={{ 
                  mb: 2, 
                  fontWeight: 'bold',
                  fontSize: '12px',
                  color: '#2b2b2b',
                  borderBottom: '1px solid #d4d4d4',
                  pb: 1
                }}>
                  COORDINATED FRAUD DETECTION
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography sx={{ 
                    fontSize: '24px', 
                    fontWeight: 'bold',
                    color: chartColors.accent2,
                    textAlign: 'center'
                  }}>
                    {metrics.fraudRingsDetected}
                  </Typography>
                  <Typography sx={{ 
                    fontSize: '11px',
                    color: '#555555',
                    textAlign: 'center',
                    fontWeight: 'bold'
                  }}>
                    FRAUD RINGS DETECTED (24H)
                  </Typography>
                </Box>

                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Device Clusters', value: 35, color: chartColors.primary },
                        { name: 'Call Patterns', value: 25, color: chartColors.accent1 },
                        { name: 'Location Based', value: 20, color: chartColors.secondary },
                        { name: 'Behavioral', value: 20, color: chartColors.accent3 },
                      ]}
                      cx="50%"
                      cy="50%"
                      outerRadius={60}
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {[
                        { name: 'Device Clusters', value: 35, color: chartColors.primary },
                        { name: 'Call Patterns', value: 25, color: chartColors.accent1 },
                        { name: 'Location Based', value: 20, color: chartColors.secondary },
                        { name: 'Behavioral', value: 20, color: chartColors.accent3 },
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#ffffff',
                        border: '1px solid #d4d4d4',
                        borderRadius: 0,
                        fontSize: '11px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
    </Box>
  );
};

export default ClassicDashboard;