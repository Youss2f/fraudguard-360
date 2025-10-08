import React from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, Typography, Box, Paper } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// A generic panel for simple actions
interface GenericPanelProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

const GenericPanel: React.FC<GenericPanelProps> = ({ open, onClose, title, children }) => (
  <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
    <DialogTitle>{title}</DialogTitle>
    <DialogContent>{children}</DialogContent>
    <DialogActions>
      <Button onClick={onClose}>Close</Button>
    </DialogActions>
  </Dialog>
);

// Mock data for pattern analysis
const patternData = [
  { name: 'Velocity Attack', count: 12, amt: 2400 },
  { name: 'Card Cloning', count: 8, amt: 1800 },
  { name: 'SIM Swap', count: 5, amt: 900 },
  { name: 'Phishing', count: 3, amt: 400 },
];

// Specific panel for Pattern Analysis
export const PatternAnalysisPanel: React.FC<{ open: boolean; onClose: () => void; }> = ({ open, onClose }) => (
  <GenericPanel open={open} onClose={onClose} title="Fraud Pattern Analysis">
    <Typography variant="body1" sx={{ mb: 2 }}>
      This chart displays common fraud patterns associated with the selected case or user.
    </Typography>
    <Box sx={{ height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={patternData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="count" fill="#8884d8" name="Number of Incidents" />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  </GenericPanel>
);

// Placeholder for Case Export
export const ExportCasePanel: React.FC<{ open: boolean; onClose: () => void; }> = ({ open, onClose }) => (
  <GenericPanel open={open} onClose={onClose} title="Export Case Details">
    <Typography>Case export options (PDF, DOCX) will be available here.</Typography>
    <Button variant="contained" sx={{ mt: 2 }}>Export as PDF</Button>
  </GenericPanel>
);

// Placeholder for Sharing a Case
export const ShareCasePanel: React.FC<{ open: boolean; onClose: () => void; }> = ({ open, onClose }) => (
  <GenericPanel open={open} onClose={onClose} title="Share Investigation Case">
    <Typography>Share this case with other analysts or teams.</Typography>
    {/* Add a text field and a share button here in a real implementation */}
  </GenericPanel>
);
