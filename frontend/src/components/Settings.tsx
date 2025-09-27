import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Divider
} from '@mui/material';

const Settings: React.FC = () => {
  const [settings, setSettings] = React.useState({
    realTimeAlerts: true,
    emailNotifications: true,
    autoInvestigation: false,
    darkMode: false
  });

  const handleSettingChange = (setting: keyof typeof settings) => {
    setSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        System Settings
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Alert Settings
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.realTimeAlerts}
                    onChange={() => handleSettingChange('realTimeAlerts')}
                  />
                }
                label="Real-time Alerts"
              />
              <br />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.emailNotifications}
                    onChange={() => handleSettingChange('emailNotifications')}
                  />
                }
                label="Email Notifications"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Settings
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.autoInvestigation}
                    onChange={() => handleSettingChange('autoInvestigation')}
                  />
                }
                label="Auto Investigation"
              />
              <br />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.darkMode}
                    onChange={() => handleSettingChange('darkMode')}
                  />
                }
                label="Dark Mode"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;