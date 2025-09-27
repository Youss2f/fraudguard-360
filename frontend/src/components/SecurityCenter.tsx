import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid,
  Card,
  CardContent,
  Alert
} from '@mui/material';

const SecurityCenter: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Security Center
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        System security status: All systems operational
      </Alert>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Access Control
              </Typography>
              <Typography color="text.secondary">
                Manage user permissions and access rights
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Audit Logs
              </Typography>
              <Typography color="text.secondary">
                View system and user activity logs
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Security Policies
              </Typography>
              <Typography color="text.secondary">
                Configure security policies and rules
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SecurityCenter;