/**
 * Professional View Options Panel Component
 * Dashboard layout and visualization options
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Switch,
  Slider,
  Divider,
  IconButton,
  Paper,
  styled,
  Chip,
} from '@mui/material';
import {
  ViewModule,
  ViewList,
  ViewQuilt,
  Dashboard,
  Timeline,
  BarChart,
  PieChart,
  TableChart,
  Close,
  Save,
  Refresh,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const ViewCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  '&:hover': {
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    transform: 'translateY(-2px)',
  },
  '&.selected': {
    border: `2px solid ${customColors.primary[500]}`,
    backgroundColor: customColors.primary[50],
  },
}));

const ViewPreview = styled(Box)(({ theme }) => ({
  width: '100%',
  height: '80px',
  backgroundColor: customColors.neutral[100],
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '4px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginBottom: '8px',
}));

interface ViewOptionsProps {
  open: boolean;
  onClose: () => void;
  onApplyView: (viewOptions: any) => void;
  currentView?: any;
}

interface ViewState {
  layout: 'grid' | 'list' | 'cards' | 'timeline';
  density: 'compact' | 'comfortable' | 'spacious';
  showKPIs: boolean;
  showCharts: boolean;
  showAlerts: boolean;
  showTransactions: boolean;
  showUsers: boolean;
  chartType: 'line' | 'bar' | 'pie' | 'area';
  refreshInterval: number;
  theme: 'light' | 'dark' | 'auto';
  animations: boolean;
  realTimeUpdates: boolean;
}

const ViewOptionsPanel: React.FC<ViewOptionsProps> = ({ 
  open, 
  onClose, 
  onApplyView,
  currentView = {}
}) => {
  const [viewOptions, setViewOptions] = useState<ViewState>({
    layout: currentView.layout || 'grid',
    density: currentView.density || 'comfortable',
    showKPIs: currentView.showKPIs ?? true,
    showCharts: currentView.showCharts ?? true,
    showAlerts: currentView.showAlerts ?? true,
    showTransactions: currentView.showTransactions ?? true,
    showUsers: currentView.showUsers ?? true,
    chartType: currentView.chartType || 'line',
    refreshInterval: currentView.refreshInterval || 30,
    theme: currentView.theme || 'light',
    animations: currentView.animations ?? true,
    realTimeUpdates: currentView.realTimeUpdates ?? true,
  });

  const layoutOptions = [
    {
      id: 'grid',
      name: 'Grid Layout',
      icon: <ViewModule />,
      description: 'Cards arranged in a responsive grid'
    },
    {
      id: 'list',
      name: 'List Layout', 
      icon: <ViewList />,
      description: 'Vertical list with detailed information'
    },
    {
      id: 'cards',
      name: 'Card Layout',
      icon: <ViewQuilt />,
      description: 'Large cards with rich visualizations'
    },
    {
      id: 'timeline',
      name: 'Timeline Layout',
      icon: <Timeline />,
      description: 'Chronological event timeline'
    }
  ];

  const chartOptions = [
    { id: 'line', name: 'Line Charts', icon: <BarChart /> },
    { id: 'bar', name: 'Bar Charts', icon: <BarChart /> },
    { id: 'pie', name: 'Pie Charts', icon: <PieChart /> },
    { id: 'area', name: 'Area Charts', icon: <BarChart /> }
  ];

  const handleLayoutChange = (layout: string) => {
    setViewOptions(prev => ({ ...prev, layout: layout as any }));
  };

  const handleToggleSection = (section: keyof ViewState) => {
    setViewOptions(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const handleApplyView = () => {
    onApplyView(viewOptions);
    onClose();
  };

  const resetToDefault = () => {
    setViewOptions({
      layout: 'grid',
      density: 'comfortable',
      showKPIs: true,
      showCharts: true,
      showAlerts: true,
      showTransactions: true,
      showUsers: true,
      chartType: 'line',
      refreshInterval: 30,
      theme: 'light',
      animations: true,
      realTimeUpdates: true,
    });
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '70vh',
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
          <ViewModule color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            View Options
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        {/* Layout Selection */}
        <Box mb={4}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Dashboard Layout
          </Typography>
          <Grid container spacing={2}>
            {layoutOptions.map((layout) => (
              <Grid item xs={12} sm={6} md={3} key={layout.id}>
                <ViewCard 
                  className={viewOptions.layout === layout.id ? 'selected' : ''}
                  onClick={() => handleLayoutChange(layout.id)}
                >
                  <CardContent sx={{ textAlign: 'center', p: 2 }}>
                    <ViewPreview>
                      {layout.icon}
                    </ViewPreview>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {layout.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {layout.description}
                    </Typography>
                  </CardContent>
                </ViewCard>
              </Grid>
            ))}
          </Grid>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Display Density */}
        <Box mb={4}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Display Density
          </Typography>
          <FormControl component="fieldset">
            <RadioGroup
              row
              value={viewOptions.density}
              onChange={(e) => setViewOptions(prev => ({ ...prev, density: e.target.value as any }))}
            >
              <FormControlLabel value="compact" control={<Radio />} label="Compact" />
              <FormControlLabel value="comfortable" control={<Radio />} label="Comfortable" />
              <FormControlLabel value="spacious" control={<Radio />} label="Spacious" />
            </RadioGroup>
          </FormControl>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Dashboard Sections */}
        <Box mb={4}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Dashboard Sections
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.showKPIs}
                    onChange={() => handleToggleSection('showKPIs')}
                    color="primary"
                  />
                }
                label="KPI Metrics"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.showCharts}
                    onChange={() => handleToggleSection('showCharts')}
                    color="primary"
                  />
                }
                label="Charts & Visualizations"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.showAlerts}
                    onChange={() => handleToggleSection('showAlerts')}
                    color="primary"
                  />
                }
                label="Alert Queue"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.showTransactions}
                    onChange={() => handleToggleSection('showTransactions')}
                    color="primary"
                  />
                }
                label="Recent Transactions"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.showUsers}
                    onChange={() => handleToggleSection('showUsers')}
                    color="primary"
                  />
                }
                label="High-Risk Users"
              />
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Chart Type */}
        <Box mb={4}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Default Chart Type
          </Typography>
          <FormControl component="fieldset">
            <RadioGroup
              row
              value={viewOptions.chartType}
              onChange={(e) => setViewOptions(prev => ({ ...prev, chartType: e.target.value as any }))}
            >
              {chartOptions.map((chart) => (
                <FormControlLabel 
                  key={chart.id}
                  value={chart.id} 
                  control={<Radio />} 
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      {chart.icon}
                      {chart.name}
                    </Box>
                  } 
                />
              ))}
            </RadioGroup>
          </FormControl>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Refresh Interval */}
        <Box mb={4}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Auto-Refresh Interval
          </Typography>
          <Box px={2}>
            <Slider
              value={viewOptions.refreshInterval}
              onChange={(_, value) => setViewOptions(prev => ({ ...prev, refreshInterval: value as number }))}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${value}s`}
              min={5}
              max={300}
              step={5}
              marks={[
                { value: 5, label: '5s' },
                { value: 30, label: '30s' },
                { value: 60, label: '1m' },
                { value: 180, label: '3m' },
                { value: 300, label: '5m' }
              ]}
            />
          </Box>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Additional Options */}
        <Box mb={4}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Additional Options
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.animations}
                    onChange={() => handleToggleSection('animations')}
                    color="primary"
                  />
                }
                label="Enable Animations"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewOptions.realTimeUpdates}
                    onChange={() => handleToggleSection('realTimeUpdates')}
                    color="primary"
                  />
                }
                label="Real-time Updates"
              />
            </Grid>
          </Grid>
        </Box>
      </DialogContent>

      <DialogActions sx={{ 
        p: 3, 
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Box display="flex" gap={2} width="100%" justifyContent="space-between">
          <Button
            startIcon={<Refresh />}
            onClick={resetToDefault}
            variant="outlined"
            color="secondary"
          >
            Reset to Default
          </Button>
          <Box display="flex" gap={2}>
            <Button onClick={onClose}>
              Cancel
            </Button>
            <Button
              startIcon={<Save />}
              onClick={handleApplyView}
              variant="contained"
              color="primary"
            >
              Apply View
            </Button>
          </Box>
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default ViewOptionsPanel;