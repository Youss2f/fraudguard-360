import React, { useState } from 'react';
import { Modal, Box, Typography, Button, Slider, Checkbox, FormControlLabel, TextField, Select, MenuItem } from '@mui/material';

interface RevolutionaryModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  type: 'filter' | 'export' | 'actions' | 'share';
  onApplyFilter?: (filters: any) => void;
  onExportData?: (format: string, options: any) => void;
  onActionExecute?: (action: string, data: any) => void;
  data?: any;
}

export const RevolutionaryModal: React.FC<RevolutionaryModalProps> = ({
  open,
  onClose,
  title,
  type,
  onApplyFilter,
  onExportData,
  onActionExecute,
  data,
}) => {
  const [riskLevel, setRiskLevel] = useState('all');
  const [amount, setAmount] = useState([0, 100000]);
  const [exportFormat, setExportFormat] = useState('csv');

  const handleApplyFilter = () => {
    onApplyFilter?.({ riskLevel, amount });
  };

  const handleExport = () => {
    onExportData?.(exportFormat, {});
  };

  const handleAction = (action: string) => {
    onActionExecute?.(action, data);
  };

  const renderContent = () => {
    switch (type) {
      case 'filter':
        return (
          <Box>
            <Typography>Risk Level</Typography>
            <Select value={riskLevel} onChange={(e) => setRiskLevel(e.target.value)} fullWidth>
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
            </Select>
            <Typography>Amount</Typography>
            <Slider value={amount} onChange={(_, newValue) => setAmount(newValue as number[])} min={0} max={100000} valueLabelDisplay="auto" />
          </Box>
        );
      case 'export':
        return (
          <Box>
            <Typography>Export Format</Typography>
            <Select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)} fullWidth>
              <MenuItem value="csv">CSV</MenuItem>
              <MenuItem value="json">JSON</MenuItem>
              <MenuItem value="pdf">PDF</MenuItem>
            </Select>
          </Box>
        );
      case 'actions':
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Button variant="contained" onClick={() => handleAction('investigate')}>Investigate</Button>
            <Button variant="contained" color="warning" onClick={() => handleAction('block')}>Block User</Button>
            <Button variant="contained" color="success" onClick={() => handleAction('resolve')}>Resolve Alert</Button>
          </Box>
        );
      default:
        return <Typography>Content for {type}</Typography>;
    }
  };

  const renderActions = () => {
    switch (type) {
      case 'filter':
        return (
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 2 }}>
            <Button onClick={onClose}>Cancel</Button>
            <Button variant="contained" onClick={handleApplyFilter}>Apply Filters</Button>
          </Box>
        );
      case 'export':
        return (
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 2 }}>
            <Button onClick={onClose}>Cancel</Button>
            <Button variant="contained" onClick={handleExport}>Export</Button>
          </Box>
        );
      default:
        return (
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 2 }}>
            <Button onClick={onClose}>Close</Button>
          </Box>
        );
    }
  };

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={{ ...style }}>
        <Typography variant="h6" component="h2">{title}</Typography>
        <Box sx={{ mt: 2 }}>{renderContent()}</Box>
        {renderActions()}
      </Box>
    </Modal>
  );
};

const style = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};