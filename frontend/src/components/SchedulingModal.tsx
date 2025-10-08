import React, { useState } from 'react';
import { Modal, Box, Typography, Button, TextField, Select, MenuItem, FormControl, InputLabel } from '@mui/material';

interface SchedulingModalProps {
  open: boolean;
  onClose: () => void;
  onSchedule: (schedule: any) => void;
}

const style = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 500,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

export const SchedulingModal: React.FC<SchedulingModalProps> = ({ open, onClose, onSchedule }) => {
  const [reportType, setReportType] = useState('daily_summary');
  const [deliveryTime, setDeliveryTime] = useState('09:00');
  const [recipients, setRecipients] = useState('fraud-team@example.com');

  const handleSchedule = () => {
    onSchedule({ reportType, deliveryTime, recipients });
    onClose();
  };

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={style}>
        <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
          Add New Report Schedule
        </Typography>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Report Type</InputLabel>
          <Select value={reportType} label="Report Type" onChange={(e) => setReportType(e.target.value)}>
            <MenuItem value="daily_summary">Daily Summary</MenuItem>
            <MenuItem value="weekly_deep_dive">Weekly Deep Dive</MenuItem>
            <MenuItem value="high_risk_alerts">High-Risk Alerts</MenuItem>
          </Select>
        </FormControl>
        <TextField
          label="Delivery Time (UTC)"
          type="time"
          value={deliveryTime}
          onChange={(e) => setDeliveryTime(e.target.value)}
          fullWidth
          sx={{ mb: 2 }}
        />
        <TextField
          label="Recipients (comma-separated)"
          value={recipients}
          onChange={(e) => setRecipients(e.target.value)}
          fullWidth
          sx={{ mb: 2 }}
        />
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="contained" onClick={handleSchedule}>Save Schedule</Button>
        </Box>
      </Box>
    </Modal>
  );
};