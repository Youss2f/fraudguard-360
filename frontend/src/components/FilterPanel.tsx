/**
 * Professional Filter Panel Component
 * Advanced filtering for fraud detection dashboard
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Chip,
  Typography,
  Divider,
  Switch,
  FormControlLabel,
  Slider,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  Paper,
  styled,
} from '@mui/material';
import {
  FilterList,
  Clear,
  Save,
  Refresh,
  Close,
  DateRange,
  LocationOn,
  Person,
  CreditCard,
  TrendingUp,
} from '@mui/icons-material';

import { customColors } from '../theme/enterpriseTheme';

const FilterCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
  marginBottom: '16px',
}));

const FilterChip = styled(Chip)(({ theme }) => ({
  margin: '2px',
  backgroundColor: customColors.primary[100],
  color: customColors.primary[800],
  '& .MuiChip-deleteIcon': {
    color: customColors.primary[600],
  },
}));

interface FilterPanelProps {
  open: boolean;
  onClose: () => void;
  onApplyFilters: (filters: any) => void;
  currentFilters?: any;
}

interface FilterState {
  dateRange: {
    start: Date | null;
    end: Date | null;
  };
  riskScore: [number, number];
  fraudTypes: string[];
  status: string[];
  severity: string[];
  assignedAnalyst: string;
  location: string[];
  amountRange: [number, number];
  customerTier: string[];
  paymentMethod: string[];
  realTimeOnly: boolean;
  includeResolved: boolean;
}

const FilterPanel: React.FC<FilterPanelProps> = ({ 
  open, 
  onClose, 
  onApplyFilters,
  currentFilters = {}
}) => {
  const [filters, setFilters] = useState<FilterState>({
    dateRange: {
      start: currentFilters.dateRange?.start || null,
      end: currentFilters.dateRange?.end || null,
    },
    riskScore: currentFilters.riskScore || [0, 100],
    fraudTypes: currentFilters.fraudTypes || [],
    status: currentFilters.status || [],
    severity: currentFilters.severity || [],
    assignedAnalyst: currentFilters.assignedAnalyst || '',
    location: currentFilters.location || [],
    amountRange: currentFilters.amountRange || [0, 100000],
    customerTier: currentFilters.customerTier || [],
    paymentMethod: currentFilters.paymentMethod || [],
    realTimeOnly: currentFilters.realTimeOnly || false,
    includeResolved: currentFilters.includeResolved || true,
  });

  const fraudTypeOptions = [
    'Card Testing',
    'Account Takeover', 
    'Velocity Check Failed',
    'Suspicious Pattern',
    'High Risk Transaction',
    'Identity Theft',
    'Money Laundering',
    'Chargeback Fraud'
  ];

  const statusOptions = [
    'New',
    'Investigating', 
    'Escalated',
    'Resolved',
    'Closed',
    'False Positive'
  ];

  const severityOptions = [
    'Critical',
    'High',
    'Medium', 
    'Low'
  ];

  const analystOptions = [
    'Unassigned',
    'John Smith',
    'Sarah Johnson', 
    'Mike Chen',
    'Lisa Rodriguez',
    'David Kim'
  ];

  const locationOptions = [
    'United States',
    'United Kingdom',
    'Canada',
    'Germany',
    'France',
    'Japan',
    'Australia',
    'Brazil'
  ];

  const customerTierOptions = [
    'Premium',
    'Gold',
    'Silver',
    'Bronze',
    'Standard'
  ];

  const paymentMethodOptions = [
    'Credit Card',
    'Debit Card',
    'Bank Transfer',
    'Digital Wallet',
    'Cryptocurrency',
    'Wire Transfer'
  ];

  const handleMultiSelectChange = (field: keyof FilterState, value: string[]) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleRangeChange = (field: keyof FilterState, value: [number, number]) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleClearFilters = () => {
    setFilters({
      dateRange: { start: null, end: null },
      riskScore: [0, 100],
      fraudTypes: [],
      status: [],
      severity: [],
      assignedAnalyst: '',
      location: [],
      amountRange: [0, 100000],
      customerTier: [],
      paymentMethod: [],
      realTimeOnly: false,
      includeResolved: true,
    });
  };

  const handleApplyFilters = () => {
    onApplyFilters(filters);
    onClose();
  };

  const getActiveFilterCount = () => {
    let count = 0;
    if (filters.dateRange.start || filters.dateRange.end) count++;
    if (filters.riskScore[0] > 0 || filters.riskScore[1] < 100) count++;
    if (filters.fraudTypes.length > 0) count++;
    if (filters.status.length > 0) count++;
    if (filters.severity.length > 0) count++;
    if (filters.assignedAnalyst) count++;
    if (filters.location.length > 0) count++;
    if (filters.amountRange[0] > 0 || filters.amountRange[1] < 100000) count++;
    if (filters.customerTier.length > 0) count++;
    if (filters.paymentMethod.length > 0) count++;
    if (filters.realTimeOnly) count++;
    if (!filters.includeResolved) count++;
    return count;
  };

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
            <FilterList color="primary" />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Advanced Filter Panel
            </Typography>
            {getActiveFilterCount() > 0 && (
              <Chip 
                label={`${getActiveFilterCount()} filters active`}
                size="small"
                color="primary"
              />
            )}
          </Box>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </DialogTitle>

        <DialogContent sx={{ p: 3 }}>
          <Grid container spacing={3}>
            {/* Date Range */}
            <Grid item xs={12} md={6}>
              <FilterCard>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <DateRange color="primary" />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Date Range
                    </Typography>
                  </Box>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="Start Date"
                        type="date"
                        value={filters.dateRange.start ? filters.dateRange.start.toISOString().split('T')[0] : ''}
                        onChange={(e) => setFilters(prev => ({
                          ...prev,
                          dateRange: { ...prev.dateRange, start: e.target.value ? new Date(e.target.value) : null }
                        }))}
                        fullWidth
                        size="small"
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="End Date"
                        type="date"
                        value={filters.dateRange.end ? filters.dateRange.end.toISOString().split('T')[0] : ''}
                        onChange={(e) => setFilters(prev => ({
                          ...prev,
                          dateRange: { ...prev.dateRange, end: e.target.value ? new Date(e.target.value) : null }
                        }))}
                        fullWidth
                        size="small"
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Risk Score Range */}
            <Grid item xs={12} md={6}>
              <FilterCard>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <TrendingUp color="primary" />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Risk Score Range
                    </Typography>
                  </Box>
                  <Box px={2}>
                    <Slider
                      value={filters.riskScore}
                      onChange={(_, value) => handleRangeChange('riskScore', value as [number, number])}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(value) => `${value}%`}
                      min={0}
                      max={100}
                      marks={[
                        { value: 0, label: '0%' },
                        { value: 25, label: '25%' },
                        { value: 50, label: '50%' },
                        { value: 75, label: '75%' },
                        { value: 100, label: '100%' }
                      ]}
                    />
                  </Box>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Fraud Types */}
            <Grid item xs={12} md={6}>
              <FilterCard>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    Fraud Types
                  </Typography>
                  <FormControl fullWidth size="small">
                    <InputLabel>Select Fraud Types</InputLabel>
                    <Select
                      multiple
                      value={filters.fraudTypes}
                      onChange={(e) => handleMultiSelectChange('fraudTypes', e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <FilterChip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {fraudTypeOptions.map((type) => (
                        <MenuItem key={type} value={type}>
                          {type}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Status */}
            <Grid item xs={12} md={6}>
              <FilterCard>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    Alert Status
                  </Typography>
                  <FormControl fullWidth size="small">
                    <InputLabel>Select Status</InputLabel>
                    <Select
                      multiple
                      value={filters.status}
                      onChange={(e) => handleMultiSelectChange('status', e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <FilterChip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {statusOptions.map((status) => (
                        <MenuItem key={status} value={status}>
                          {status}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Severity */}
            <Grid item xs={12} md={6}>
              <FilterCard>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    Severity Level
                  </Typography>
                  <FormControl fullWidth size="small">
                    <InputLabel>Select Severity</InputLabel>
                    <Select
                      multiple
                      value={filters.severity}
                      onChange={(e) => handleMultiSelectChange('severity', e.target.value as string[])}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <FilterChip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {severityOptions.map((severity) => (
                        <MenuItem key={severity} value={severity}>
                          {severity}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Assigned Analyst */}
            <Grid item xs={12} md={6}>
              <FilterCard>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <Person color="primary" />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Assigned Analyst
                    </Typography>
                  </Box>
                  <FormControl fullWidth size="small">
                    <InputLabel>Select Analyst</InputLabel>
                    <Select
                      value={filters.assignedAnalyst}
                      onChange={(e) => setFilters(prev => ({ ...prev, assignedAnalyst: e.target.value }))}
                    >
                      {analystOptions.map((analyst) => (
                        <MenuItem key={analyst} value={analyst}>
                          {analyst}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Transaction Amount Range */}
            <Grid item xs={12}>
              <FilterCard>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <CreditCard color="primary" />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Transaction Amount Range
                    </Typography>
                  </Box>
                  <Box px={2}>
                    <Slider
                      value={filters.amountRange}
                      onChange={(_, value) => handleRangeChange('amountRange', value as [number, number])}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(value) => `$${value.toLocaleString()}`}
                      min={0}
                      max={100000}
                      step={1000}
                      marks={[
                        { value: 0, label: '$0' },
                        { value: 25000, label: '$25K' },
                        { value: 50000, label: '$50K' },
                        { value: 75000, label: '$75K' },
                        { value: 100000, label: '$100K+' }
                      ]}
                    />
                  </Box>
                </CardContent>
              </FilterCard>
            </Grid>

            {/* Additional Options */}
            <Grid item xs={12}>
              <FilterCard>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                    Additional Options
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={filters.realTimeOnly}
                            onChange={(e) => setFilters(prev => ({ ...prev, realTimeOnly: e.target.checked }))}
                            color="primary"
                          />
                        }
                        label="Real-time alerts only"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={filters.includeResolved}
                            onChange={(e) => setFilters(prev => ({ ...prev, includeResolved: e.target.checked }))}
                            color="primary"
                          />
                        }
                        label="Include resolved cases"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </FilterCard>
            </Grid>
          </Grid>
        </DialogContent>

        <DialogActions sx={{ 
          p: 3, 
          backgroundColor: customColors.background.ribbon,
          borderTop: `1px solid ${customColors.neutral[200]}`
        }}>
          <Box display="flex" gap={2} width="100%" justifyContent="space-between">
            <Button
              startIcon={<Clear />}
              onClick={handleClearFilters}
              variant="outlined"
              color="secondary"
            >
              Clear All
            </Button>
            <Box display="flex" gap={2}>
              <Button onClick={onClose}>
                Cancel
              </Button>
              <Button
                startIcon={<FilterList />}
                onClick={handleApplyFilters}
                variant="contained"
                color="primary"
              >
                Apply Filters ({getActiveFilterCount()})
              </Button>
            </Box>
          </Box>
        </DialogActions>
      </Dialog>
  );
};

export default FilterPanel;