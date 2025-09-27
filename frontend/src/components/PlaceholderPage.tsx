/**
 * Placeholder Page Component
 * Generic page component for routes under development
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Container,
  Button,
  Stack,
  Chip,
} from '@mui/material';
import {
  Construction,
  ArrowBack,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface PlaceholderPageProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  features?: string[];
}

const PlaceholderPage: React.FC<PlaceholderPageProps> = ({
  title,
  description = 'This section is currently under development.',
  icon = <Construction />,
  features = [],
}) => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper
        sx={{
          p: 6,
          textAlign: 'center',
          borderRadius: 3,
          background: 'linear-gradient(135deg, rgba(25, 118, 210, 0.05) 0%, rgba(156, 39, 176, 0.05) 100%)',
        }}
      >
        <Box sx={{ mb: 3, color: 'primary.main', opacity: 0.7 }}>
          {React.cloneElement(icon as React.ReactElement, { sx: { fontSize: 64 } })}
        </Box>
        
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
          {title}
        </Typography>
        
        <Typography variant="h6" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
          {description}
        </Typography>

        {features.length > 0 && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Planned Features:
            </Typography>
            <Stack direction="row" spacing={1} justifyContent="center" flexWrap="wrap" useFlexGap>
              {features.map((feature, index) => (
                <Chip
                  key={index}
                  label={feature}
                  variant="outlined"
                  size="small"
                  sx={{ mb: 1 }}
                />
              ))}
            </Stack>
          </Box>
        )}

        <Stack direction="row" spacing={2} justifyContent="center">
          <Button
            variant="contained"
            startIcon={<ArrowBack />}
            onClick={() => navigate('/dashboard')}
          >
            Back to Dashboard
          </Button>
        </Stack>
      </Paper>
    </Container>
  );
};

export default PlaceholderPage;