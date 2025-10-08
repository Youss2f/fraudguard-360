import React from 'react';
import { Drawer, Box, Typography, Button, List, ListItem, ListItemText, Divider } from '@mui/material';

interface ReportsPanelProps {
  open: boolean;
  onClose: () => void;
  onActionClick: (action: string) => void;
}

const ReportsPanel: React.FC<ReportsPanelProps> = ({ open, onClose, onActionClick }) => {
  return (
    <Drawer anchor="right" open={open} onClose={onClose}>
      <Box sx={{ width: 400, p: 3 }}>
        <Typography variant="h5" sx={{ mb: 2 }}>Reports</Typography>
        <List>
          <ListItem button onClick={() => onActionClick('daily')}>
            <ListItemText primary="Generate Daily Report" />
          </ListItem>
          <ListItem button onClick={() => onActionClick('weekly')}>
            <ListItemText primary="Generate Weekly Report" />
          </ListItem>
          <ListItem button onClick={() => onActionClick('custom')}>
            <ListItemText primary="Open Custom Report Builder" />
          </ListItem>
        </List>
        <Divider sx={{ my: 2 }} />
        <Typography variant="h6" sx={{ mb: 1 }}>Scheduling</Typography>
        <Button
          variant="contained"
          fullWidth
          onClick={() => onActionClick('add-schedule')}
        >
          Add Schedule
        </Button>
        <Button onClick={onClose} sx={{ mt: 4, float: 'right' }}>Close</Button>
      </Box>
    </Drawer>
  );
};

export default ReportsPanel;