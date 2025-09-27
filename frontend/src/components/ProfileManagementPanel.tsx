/**
 * Professional Profile Management Panel Component
 * User profile settings and preferences management
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
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Avatar,
  IconButton,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Switch,
  FormControlLabel,
  Divider,
  Paper,
  Chip,
  Alert,
  RadioGroup,
  Radio,
  Slider,
  styled,
} from '@mui/material';
import {
  Person,
  Close,
  Edit,
  Camera,
  Security,
  Notifications,
  Palette,
  Language,
  Schedule,
  VpnKey,
  Visibility,
  VisibilityOff,
  Save,
  Refresh,
  Delete,
  History,
  Settings,
  Dashboard,
  Email,
  Phone,
  LocationOn,
  Work,
  School,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const ProfileCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  '&:hover': {
    borderColor: customColors.primary[300],
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
  },
}));

interface ProfileManagementPanelProps {
  open: boolean;
  onClose: () => void;
}

interface UserProfile {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  department: string;
  role: string;
  location: string;
  timezone: string;
  avatar: string;
  joinDate: Date;
  lastLogin: Date;
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  dateFormat: string;
  timeFormat: '12h' | '24h';
  notifications: {
    email: boolean;
    push: boolean;
    desktop: boolean;
    critical: boolean;
    reports: boolean;
    investigations: boolean;
  };
  dashboard: {
    defaultView: string;
    refreshInterval: number;
    showWelcome: boolean;
    compactMode: boolean;
  };
  security: {
    twoFactorEnabled: boolean;
    sessionTimeout: number;
    passwordLastChanged: Date;
  };
}

const ProfileManagementPanel: React.FC<ProfileManagementPanelProps> = ({ 
  open, 
  onClose 
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [profile, setProfile] = useState<UserProfile>({
    id: 'usr-001',
    firstName: 'Sarah',
    lastName: 'Johnson',
    email: 'sarah.johnson@company.com',
    phone: '+1 (555) 123-4567',
    department: 'Fraud Detection',
    role: 'Senior Fraud Analyst',
    location: 'New York, NY',
    timezone: 'America/New_York',
    avatar: '',
    joinDate: new Date('2022-03-15'),
    lastLogin: new Date(),
  });

  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: 'light',
    language: 'en',
    dateFormat: 'MM/dd/yyyy',
    timeFormat: '12h',
    notifications: {
      email: true,
      push: true,
      desktop: false,
      critical: true,
      reports: true,
      investigations: true,
    },
    dashboard: {
      defaultView: 'overview',
      refreshInterval: 30,
      showWelcome: true,
      compactMode: false,
    },
    security: {
      twoFactorEnabled: true,
      sessionTimeout: 60,
      passwordLastChanged: new Date('2024-01-15'),
    },
  });

  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [showPasswords, setShowPasswords] = useState({
    current: false,
    new: false,
    confirm: false,
  });

  const handleProfileChange = (field: keyof UserProfile, value: any) => {
    setProfile(prev => ({ ...prev, [field]: value }));
  };

  const handlePreferenceChange = (section: keyof UserPreferences, field: string, value: any) => {
    setPreferences(prev => ({
      ...prev,
      [section]: {
        ...(prev[section] as any),
        [field]: value
      } as any
    }));
  };

  const handleSaveProfile = () => {
    // In a real app, this would save to the backend
    alert('Profile saved successfully!');
  };

  const handleChangePassword = () => {
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      alert('Passwords do not match!');
      return;
    }
    
    if (passwordForm.newPassword.length < 8) {
      alert('Password must be at least 8 characters long!');
      return;
    }
    
    // In a real app, this would validate and change the password
    alert('Password changed successfully!');
    setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
  };

  const renderProfileInfo = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Profile Information
      </Typography>
      
      {/* Avatar Section */}
      <ProfileCard sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={3}>
            <Box position="relative">
              <Avatar
                sx={{ 
                  width: 120, 
                  height: 120,
                  bgcolor: customColors.primary[500],
                  fontSize: '2rem',
                  fontWeight: 600
                }}
                src={profile.avatar}
              >
                {profile.firstName.charAt(0)}{profile.lastName.charAt(0)}
              </Avatar>
              <IconButton
                sx={{
                  position: 'absolute',
                  bottom: 0,
                  right: 0,
                  bgcolor: customColors.primary[500],
                  color: 'white',
                  '&:hover': { bgcolor: customColors.primary[600] }
                }}
                size="small"
              >
                <Camera />
              </IconButton>
            </Box>
            
            <Box flex={1}>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {profile.firstName} {profile.lastName}
              </Typography>
              <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 1 }}>
                {profile.role}
              </Typography>
              <Chip 
                size="small" 
                label={profile.department}
                color="primary"
                variant="outlined"
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                Member since {profile.joinDate.toLocaleDateString()}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </ProfileCard>

      {/* Basic Information */}
      <ProfileCard sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Basic Information
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="First Name"
                value={profile.firstName}
                onChange={(e) => handleProfileChange('firstName', e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Email Address"
                value={profile.email}
                onChange={(e) => handleProfileChange('email', e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Department"
                value={profile.department}
                onChange={(e) => handleProfileChange('department', e.target.value)}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Last Name"
                value={profile.lastName}
                onChange={(e) => handleProfileChange('lastName', e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Phone Number"
                value={profile.phone}
                onChange={(e) => handleProfileChange('phone', e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Role/Title"
                value={profile.role}
                onChange={(e) => handleProfileChange('role', e.target.value)}
              />
            </Grid>
          </Grid>
        </CardContent>
      </ProfileCard>

      {/* Location & Time */}
      <ProfileCard>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Location & Time Settings
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Location"
                value={profile.location}
                onChange={(e) => handleProfileChange('location', e.target.value)}
                sx={{ mb: 2 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Timezone</InputLabel>
                <Select
                  value={profile.timezone}
                  onChange={(e) => handleProfileChange('timezone', e.target.value)}
                >
                  <MenuItem value="America/New_York">Eastern Time (EST/EDT)</MenuItem>
                  <MenuItem value="America/Chicago">Central Time (CST/CDT)</MenuItem>
                  <MenuItem value="America/Denver">Mountain Time (MST/MDT)</MenuItem>
                  <MenuItem value="America/Los_Angeles">Pacific Time (PST/PDT)</MenuItem>
                  <MenuItem value="UTC">UTC</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </ProfileCard>
      
      <Box display="flex" gap={2} mt={3}>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSaveProfile}
        >
          Save Changes
        </Button>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
        >
          Reset
        </Button>
      </Box>
    </Box>
  );

  const renderPreferences = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        User Preferences
      </Typography>
      
      {/* Appearance */}
      <ProfileCard sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Appearance & Display
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Theme</InputLabel>
                <Select
                  value={preferences.theme}
                  onChange={(e) => handlePreferenceChange('theme', 'theme', e.target.value)}
                >
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="auto">Auto (System)</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl fullWidth>
                <InputLabel>Language</InputLabel>
                <Select
                  value={preferences.language}
                  onChange={(e) => handlePreferenceChange('language', 'language', e.target.value)}
                >
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="es">Spanish</MenuItem>
                  <MenuItem value="fr">French</MenuItem>
                  <MenuItem value="de">German</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Date Format</InputLabel>
                <Select
                  value={preferences.dateFormat}
                  onChange={(e) => handlePreferenceChange('dateFormat', 'dateFormat', e.target.value)}
                >
                  <MenuItem value="MM/dd/yyyy">MM/DD/YYYY</MenuItem>
                  <MenuItem value="dd/MM/yyyy">DD/MM/YYYY</MenuItem>
                  <MenuItem value="yyyy-MM-dd">YYYY-MM-DD</MenuItem>
                </Select>
              </FormControl>
              
              <RadioGroup
                value={preferences.timeFormat}
                onChange={(e) => handlePreferenceChange('timeFormat', 'timeFormat', e.target.value)}
              >
                <FormControlLabel value="12h" control={<Radio />} label="12-hour format" />
                <FormControlLabel value="24h" control={<Radio />} label="24-hour format" />
              </RadioGroup>
            </Grid>
          </Grid>
        </CardContent>
      </ProfileCard>

      {/* Dashboard Preferences */}
      <ProfileCard sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Dashboard Preferences
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Default View</InputLabel>
                <Select
                  value={preferences.dashboard.defaultView}
                  onChange={(e) => handlePreferenceChange('dashboard', 'defaultView', e.target.value)}
                >
                  <MenuItem value="overview">Overview</MenuItem>
                  <MenuItem value="alerts">Alerts</MenuItem>
                  <MenuItem value="investigations">Investigations</MenuItem>
                  <MenuItem value="reports">Reports</MenuItem>
                </Select>
              </FormControl>
              
              <Typography variant="body2" gutterBottom>
                Auto-refresh Interval: {preferences.dashboard.refreshInterval}s
              </Typography>
              <Slider
                value={preferences.dashboard.refreshInterval}
                onChange={(_, value) => handlePreferenceChange('dashboard', 'refreshInterval', value)}
                min={10}
                max={300}
                step={10}
                marks={[
                  { value: 10, label: '10s' },
                  { value: 60, label: '1m' },
                  { value: 300, label: '5m' }
                ]}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.dashboard.showWelcome}
                    onChange={(e) => handlePreferenceChange('dashboard', 'showWelcome', e.target.checked)}
                  />
                }
                label="Show Welcome Message"
                sx={{ mb: 2 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.dashboard.compactMode}
                    onChange={(e) => handlePreferenceChange('dashboard', 'compactMode', e.target.checked)}
                  />
                }
                label="Compact Mode"
              />
            </Grid>
          </Grid>
        </CardContent>
      </ProfileCard>

      {/* Notification Preferences */}
      <ProfileCard>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Notification Preferences
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                Delivery Methods
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.notifications.email}
                    onChange={(e) => handlePreferenceChange('notifications', 'email', e.target.checked)}
                  />
                }
                label="Email Notifications"
                sx={{ mb: 1 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.notifications.push}
                    onChange={(e) => handlePreferenceChange('notifications', 'push', e.target.checked)}
                  />
                }
                label="Push Notifications"
                sx={{ mb: 1 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.notifications.desktop}
                    onChange={(e) => handlePreferenceChange('notifications', 'desktop', e.target.checked)}
                  />
                }
                label="Desktop Notifications"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                Notification Types
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.notifications.critical}
                    onChange={(e) => handlePreferenceChange('notifications', 'critical', e.target.checked)}
                  />
                }
                label="Critical Alerts"
                sx={{ mb: 1 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.notifications.reports}
                    onChange={(e) => handlePreferenceChange('notifications', 'reports', e.target.checked)}
                  />
                }
                label="Report Updates"
                sx={{ mb: 1 }}
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={preferences.notifications.investigations}
                    onChange={(e) => handlePreferenceChange('notifications', 'investigations', e.target.checked)}
                  />
                }
                label="Investigation Updates"
              />
            </Grid>
          </Grid>
        </CardContent>
      </ProfileCard>
    </Box>
  );

  const renderSecurity = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Security Settings
      </Typography>
      
      {/* Two-Factor Authentication */}
      <ProfileCard sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Two-Factor Authentication
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Add an extra layer of security to your account
              </Typography>
            </Box>
            <Switch
              checked={preferences.security.twoFactorEnabled}
              onChange={(e) => handlePreferenceChange('security', 'twoFactorEnabled', e.target.checked)}
            />
          </Box>
          
          {preferences.security.twoFactorEnabled && (
            <Alert severity="success">
              Two-factor authentication is enabled. Your account is protected.
            </Alert>
          )}
        </CardContent>
      </ProfileCard>

      {/* Password Management */}
      <ProfileCard sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Change Password
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Last changed: {preferences.security.passwordLastChanged.toLocaleDateString()}
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Current Password"
                type={showPasswords.current ? 'text' : 'password'}
                value={passwordForm.currentPassword}
                onChange={(e) => setPasswordForm(prev => ({ ...prev, currentPassword: e.target.value }))}
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowPasswords(prev => ({ ...prev, current: !prev.current }))}
                    >
                      {showPasswords.current ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  )
                }}
                sx={{ mb: 2 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="New Password"
                type={showPasswords.new ? 'text' : 'password'}
                value={passwordForm.newPassword}
                onChange={(e) => setPasswordForm(prev => ({ ...prev, newPassword: e.target.value }))}
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowPasswords(prev => ({ ...prev, new: !prev.new }))}
                    >
                      {showPasswords.new ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  )
                }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Confirm New Password"
                type={showPasswords.confirm ? 'text' : 'password'}
                value={passwordForm.confirmPassword}
                onChange={(e) => setPasswordForm(prev => ({ ...prev, confirmPassword: e.target.value }))}
                InputProps={{
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowPasswords(prev => ({ ...prev, confirm: !prev.confirm }))}
                    >
                      {showPasswords.confirm ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  )
                }}
              />
            </Grid>
          </Grid>
          
          <Button
            variant="contained"
            onClick={handleChangePassword}
            sx={{ mt: 2 }}
            disabled={!passwordForm.currentPassword || !passwordForm.newPassword || !passwordForm.confirmPassword}
          >
            Change Password
          </Button>
        </CardContent>
      </ProfileCard>

      {/* Session Management */}
      <ProfileCard>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Session Settings
          </Typography>
          
          <Typography variant="body2" gutterBottom>
            Session Timeout: {preferences.security.sessionTimeout} minutes
          </Typography>
          <Slider
            value={preferences.security.sessionTimeout}
            onChange={(_, value) => handlePreferenceChange('security', 'sessionTimeout', value)}
            min={15}
            max={480}
            step={15}
            marks={[
              { value: 15, label: '15m' },
              { value: 60, label: '1h' },
              { value: 240, label: '4h' },
              { value: 480, label: '8h' }
            ]}
          />
        </CardContent>
      </ProfileCard>
    </Box>
  );

  const tabLabels = [
    'Profile Info',
    'Preferences',
    'Security'
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
          <Person color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Profile Management
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
        {currentTab === 0 && renderProfileInfo()}
        {currentTab === 1 && renderPreferences()}
        {currentTab === 2 && renderSecurity()}
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

export default ProfileManagementPanel;