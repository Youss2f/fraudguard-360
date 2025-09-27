/**
 * Professional Admin Panel Component
 * System administration and user management interface
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
  ListItemSecondaryAction,
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
  Switch,
  FormControlLabel,
  Avatar,
  Badge,
  Divider,
  LinearProgress,
  Alert,
  styled,
} from '@mui/material';
import {
  AdminPanelSettings,
  Close,
  Person,
  Group,
  Security,
  Settings,
  Storage,
  Assessment,
  Add,
  Edit,
  Delete,
  Block,
  CheckCircle,
  Warning,
  Error,
  Info,
  Refresh,
  Download,
  Upload,
  Backup,
  Schedule,
  Notifications,
  VpnKey,
  Policy,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const StatCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)',
  },
}));

interface AdminPanelProps {
  open: boolean;
  onClose: () => void;
}

interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'analyst' | 'viewer';
  status: 'active' | 'inactive' | 'suspended';
  lastLogin: Date;
  department: string;
  permissions: string[];
}

interface SystemStat {
  label: string;
  value: string | number;
  change: number;
  icon: React.ReactNode;
  color: string;
}

const AdminPanel: React.FC<AdminPanelProps> = ({ 
  open, 
  onClose 
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [users, setUsers] = useState<User[]>([]);
  const [systemStats, setSystemStats] = useState<SystemStat[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [showUserDialog, setShowUserDialog] = useState(false);
  const [systemHealth, setSystemHealth] = useState({
    cpu: 45,
    memory: 68,
    storage: 32,
    network: 92
  });

  useEffect(() => {
    // Generate mock users
    const mockUsers: User[] = [
      {
        id: 'usr-001',
        name: 'Sarah Johnson',
        email: 'sarah.johnson@company.com',
        role: 'admin',
        status: 'active',
        lastLogin: new Date(Date.now() - 2 * 60 * 60 * 1000),
        department: 'Security',
        permissions: ['read', 'write', 'admin', 'delete']
      },
      {
        id: 'usr-002',
        name: 'Mike Chen',
        email: 'mike.chen@company.com',
        role: 'analyst',
        status: 'active',
        lastLogin: new Date(Date.now() - 4 * 60 * 60 * 1000),
        department: 'Fraud Detection',
        permissions: ['read', 'write']
      },
      {
        id: 'usr-003',
        name: 'Emma Davis',
        email: 'emma.davis@company.com',
        role: 'analyst',
        status: 'inactive',
        lastLogin: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
        department: 'Risk Management',
        permissions: ['read', 'write']
      },
      {
        id: 'usr-004',
        name: 'John Smith',
        email: 'john.smith@company.com',
        role: 'viewer',
        status: 'suspended',
        lastLogin: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        department: 'Compliance',
        permissions: ['read']
      }
    ];
    setUsers(mockUsers);

    // Generate system stats
    const mockStats: SystemStat[] = [
      {
        label: 'Total Users',
        value: mockUsers.length,
        change: 12,
        icon: <Person />,
        color: customColors.primary[500]
      },
      {
        label: 'Active Sessions',
        value: mockUsers.filter(u => u.status === 'active').length,
        change: -5,
        icon: <Group />,
        color: customColors.success[500]
      },
      {
        label: 'Security Alerts',
        value: 23,
        change: 8,
        icon: <Security />,
        color: customColors.warning[500]
      },
      {
        label: 'System Uptime',
        value: '99.8%',
        change: 0.2,
        icon: <Assessment />,
        color: customColors.success[500]
      }
    ];
    setSystemStats(mockStats);
  }, []);

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return customColors.error[500];
      case 'analyst': return customColors.primary[500];
      case 'viewer': return customColors.success[500];
      default: return customColors.neutral[500];
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return customColors.success[500];
      case 'inactive': return customColors.warning[500];
      case 'suspended': return customColors.error[500];
      default: return customColors.neutral[500];
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle color="success" />;
      case 'inactive': return <Warning color="warning" />;
      case 'suspended': return <Block color="error" />;
      default: return <Info />;
    }
  };

  const handleEditUser = (user: User) => {
    setSelectedUser(user);
    setShowUserDialog(true);
  };

  const handleAddUser = () => {
    setSelectedUser(null);
    setShowUserDialog(true);
  };

  const handleToggleUserStatus = (userId: string) => {
    setUsers(prev => prev.map(user => 
      user.id === userId 
        ? { ...user, status: user.status === 'active' ? 'inactive' : 'active' }
        : user
    ));
  };

  const renderDashboard = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        System Overview
      </Typography>
      
      {/* System Stats */}
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {systemStats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <StatCard>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: stat.color }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                    <Box display="flex" alignItems="center" mt={1}>
                      <Typography 
                        variant="caption" 
                        sx={{ 
                          color: stat.change >= 0 ? customColors.success[600] : customColors.error[600],
                          fontWeight: 600
                        }}
                      >
                        {stat.change >= 0 ? '+' : ''}{stat.change}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                        vs last month
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ color: stat.color }}>
                    {stat.icon}
                  </Box>
                </Box>
              </CardContent>
            </StatCard>
          </Grid>
        ))}
      </Grid>

      {/* System Health */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            System Health
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={6} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">CPU Usage</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.cpu}
                  color={systemHealth.cpu > 80 ? 'error' : systemHealth.cpu > 60 ? 'warning' : 'success'}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="caption">{systemHealth.cpu}%</Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">Memory</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.memory}
                  color={systemHealth.memory > 80 ? 'error' : systemHealth.memory > 60 ? 'warning' : 'success'}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="caption">{systemHealth.memory}%</Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">Storage</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.storage}
                  color={systemHealth.storage > 80 ? 'error' : systemHealth.storage > 60 ? 'warning' : 'success'}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="caption">{systemHealth.storage}%</Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box>
                <Typography variant="body2" color="text.secondary">Network</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={systemHealth.network}
                  color={systemHealth.network > 80 ? 'error' : systemHealth.network > 60 ? 'warning' : 'success'}
                  sx={{ mt: 1, mb: 1 }}
                />
                <Typography variant="caption">{systemHealth.network}%</Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Quick Actions
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Backup />}
                onClick={() => alert('System backup initiated')}
              >
                Backup System
              </Button>
            </Grid>
            <Grid item xs={6} md={3}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Refresh />}
                onClick={() => alert('System refreshed')}
              >
                Refresh Status
              </Button>
            </Grid>
            <Grid item xs={6} md={3}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Download />}
                onClick={() => alert('Logs downloaded')}
              >
                Download Logs
              </Button>
            </Grid>
            <Grid item xs={6} md={3}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Schedule />}
                onClick={() => alert('Maintenance scheduled')}
              >
                Schedule Maintenance
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );

  const renderUserManagement = () => (
    <Box p={3}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          User Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleAddUser}
        >
          Add User
        </Button>
      </Box>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>User</TableCell>
              <TableCell>Role</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Department</TableCell>
              <TableCell>Last Login</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users.map((user) => (
              <TableRow key={user.id}>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={2}>
                    <Avatar sx={{ bgcolor: getRoleColor(user.role) }}>
                      {user.name.charAt(0)}
                    </Avatar>
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {user.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {user.email}
                      </Typography>
                    </Box>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip 
                    size="small" 
                    label={user.role}
                    sx={{ 
                      bgcolor: getRoleColor(user.role),
                      color: 'white'
                    }}
                  />
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    {getStatusIcon(user.status)}
                    <Typography variant="body2">
                      {user.status}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>{user.department}</TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {user.lastLogin.toLocaleString()}
                  </Typography>
                </TableCell>
                <TableCell>
                  <IconButton size="small" onClick={() => handleEditUser(user)}>
                    <Edit />
                  </IconButton>
                  <IconButton 
                    size="small" 
                    onClick={() => handleToggleUserStatus(user.id)}
                    color={user.status === 'active' ? 'error' : 'success'}
                  >
                    {user.status === 'active' ? <Block /> : <CheckCircle />}
                  </IconButton>
                  <IconButton size="small" color="error">
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

  const renderSystemSettings = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        System Settings
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Security Settings
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    <VpnKey />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Two-Factor Authentication"
                    secondary="Require 2FA for all users"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <Schedule />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Session Timeout"
                    secondary="Auto-logout after inactivity"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <Security />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Password Policy"
                    secondary="Enforce strong passwords"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                Notification Settings
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    <Notifications />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Email Notifications"
                    secondary="Send alerts via email"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <Warning />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Critical Alerts"
                    secondary="Immediate critical notifications"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <Assessment />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Daily Reports"
                    secondary="Automated daily summary reports"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const tabLabels = [
    'Dashboard',
    'User Management',
    'System Settings'
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
          <AdminPanelSettings color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Admin Panel
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
        {currentTab === 0 && renderDashboard()}
        {currentTab === 1 && renderUserManagement()}
        {currentTab === 2 && renderSystemSettings()}
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

export default AdminPanel;