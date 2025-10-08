import React from 'react';
import { Modal, Box, Typography, Button, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

interface ReportViewerProps {
  open: boolean;
  onClose: () => void;
  title: string;
  data: any[];
  onPrint: () => void;
  onExport: () => void;
}

const style = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: '80%',
  maxWidth: '1200px',
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
  display: 'flex',
  flexDirection: 'column',
  maxHeight: '90vh',
};

export const ReportViewer: React.FC<ReportViewerProps> = ({ open, onClose, title, data, onPrint, onExport }) => {
  if (!data || data.length === 0) {
    return null; // Or a message indicating no data
  }

  const headers = Object.keys(data[0]);

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={style}>
        <Typography variant="h4" component="h2" sx={{ mb: 2 }}>
          {title}
        </Typography>
        <TableContainer component={Paper} sx={{ flexGrow: 1, overflowY: 'auto' }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                {headers.map((header) => <TableCell key={header}>{header.replace(/_/g, ' ').toUpperCase()}</TableCell>)}
              </TableRow>
            </TableHead>
            <TableBody>
              {data.map((row, rowIndex) => (
                <TableRow key={rowIndex}>
                  {headers.map((header) => <TableCell key={`${rowIndex}-${header}`}>{row[header]}</TableCell>)}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
          <Button variant="outlined" onClick={onPrint}>Print</Button>
          <Button variant="contained" onClick={onExport}>Export as CSV</Button>
          <Button onClick={onClose}>Close</Button>
        </Box>
      </Box>
    </Modal>
  );
};