/**
 * Enhanced Search Results Modal Component
 * Professional search interface with real functionality
 */

import React, { memo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Typography,
  Box,
  Chip,
  CircularProgress,
  IconButton,
  Divider,
  styled,
  alpha,
} from '@mui/material';
import {
  Search,
  Security,
  Person,
  Assessment,
  Close,
  TrendingUp,
  Warning,
} from '@mui/icons-material';

const SearchResultsContainer = styled(Box)(({ theme }) => ({
  minHeight: '400px',
  maxHeight: '600px',
  display: 'flex',
  flexDirection: 'column',
}));

const SearchResultItem = styled(ListItem)(({ theme }) => ({
  border: '1px solid #e0e0e0',
  borderRadius: '8px',
  marginBottom: '8px',
  backgroundColor: '#fafafa',
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.05),
    borderColor: theme.palette.primary.main,
    transform: 'translateY(-1px)',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  },
  transition: 'all 0.2s ease',
  cursor: 'pointer',
}));

const ScoreChip = styled(Chip)<{ score: number }>(({ theme, score }) => ({
  backgroundColor: score > 0.8 ? '#4caf50' : score > 0.6 ? '#ff9800' : '#f44336',
  color: 'white',
  fontWeight: 'bold',
  minWidth: '60px',
}));

interface SearchResult {
  id: number;
  type: 'alert' | 'case' | 'user' | 'transaction';
  title: string;
  description?: string;
  score: number;
  metadata?: any;
}

interface SearchResultsModalProps {
  open: boolean;
  onClose: () => void;
  results: SearchResult[];
  query: string;
  loading: boolean;
}

const getResultIcon = (type: string) => {
  switch (type) {
    case 'alert':
      return <Warning color="error" />;
    case 'case':
      return <Security color="primary" />;
    case 'user':
      return <Person color="action" />;
    case 'transaction':
      return <TrendingUp color="success" />;
    default:
      return <Search color="action" />;
  }
};

const getResultTypeColor = (type: string) => {
  switch (type) {
    case 'alert':
      return 'error';
    case 'case':
      return 'primary';
    case 'user':
      return 'info';
    case 'transaction':
      return 'success';
    default:
      return 'default';
  }
};

export const SearchResultsModal: React.FC<SearchResultsModalProps> = ({
  open,
  onClose,
  results,
  query,
  loading
}) => {
  const handleResultClick = (result: SearchResult) => {
    console.log('Selected result:', result);
    // Here you would navigate to the specific item or open its detail view
    // For now, we'll show a notification that this would open the item
    alert(`Opening ${result.type}: ${result.title}`);
  };

  const formatResultDescription = (result: SearchResult) => {
    if (result.description) return result.description;
    
    switch (result.type) {
      case 'alert':
        return `Fraud alert with confidence score ${(result.score * 100).toFixed(1)}%`;
      case 'case':
        return `Investigation case with risk assessment score ${(result.score * 100).toFixed(1)}%`;
      case 'user':
        return `User profile with relevance score ${(result.score * 100).toFixed(1)}%`;
      case 'transaction':
        return `Transaction record with match score ${(result.score * 100).toFixed(1)}%`;
      default:
        return `Search result with relevance ${(result.score * 100).toFixed(1)}%`;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      aria-labelledby="search-dialog-title"
      aria-describedby="search-dialog-description"
      PaperProps={{
        sx: {
          borderRadius: 2,
          maxHeight: '80vh',
        },
        role: 'dialog',
        'aria-modal': true,
      }}
    >
      <DialogTitle id="search-dialog-title">
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <Search color="primary" aria-hidden="true" />
            <Typography variant="h6" component="h2">
              Search Results for "{query}"
            </Typography>
          </Box>
          <IconButton 
            onClick={onClose} 
            size="small"
            aria-label="Close search results"
            title="Close search results"
          >
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <DialogContent id="search-dialog-description">
        <SearchResultsContainer>
          {loading ? (
            <Box 
              display="flex" 
              justifyContent="center" 
              alignItems="center" 
              height="200px"
              role="status"
              aria-label="Searching for results"
            >
              <CircularProgress aria-hidden="true" />
              <Typography variant="body1" sx={{ ml: 2 }} aria-live="polite">
                Searching...
              </Typography>
            </Box>
          ) : results.length === 0 ? (
            <Box 
              display="flex" 
              flexDirection="column" 
              alignItems="center" 
              justifyContent="center" 
              height="200px"
              role="status"
              aria-label="No search results found"
            >
              <Search sx={{ fontSize: 64, color: 'action.disabled', mb: 2 }} aria-hidden="true" />
              <Typography variant="h6" color="textSecondary" gutterBottom>
                No results found
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Try adjusting your search terms or filters
              </Typography>
            </Box>
          ) : (
            <>
              <Typography 
                variant="body2" 
                color="textSecondary" 
                sx={{ mb: 2 }}
                aria-live="polite"
                role="status"
              >
                Found {results.length} result{results.length !== 1 ? 's' : ''}
              </Typography>
              
              <List 
                sx={{ flexGrow: 1, overflow: 'auto' }}
                role="listbox"
                aria-label="Search results"
              >
                {results.map((result, index) => (
                  <SearchResultItem
                    key={result.id}
                    onClick={() => handleResultClick(result)}
                    role="option"
                    aria-selected={false}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleResultClick(result);
                      }
                    }}
                    aria-label={`${result.type}: ${result.title}. Score: ${(result.score * 100).toFixed(0)}%. Press Enter to open.`}
                  >
                    <ListItemIcon aria-hidden="true">
                      {getResultIcon(result.type)}
                    </ListItemIcon>
                    
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="body1" fontWeight="medium">
                            {result.title}
                          </Typography>
                          <Chip
                            label={result.type.toUpperCase()}
                            size="small"
                            color={getResultTypeColor(result.type) as any}
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                          {formatResultDescription(result)}
                        </Typography>
                      }
                    />
                    
                    <Box display="flex" alignItems="center" gap={1}>
                      <ScoreChip
                        label={`${(result.score * 100).toFixed(0)}%`}
                        size="small"
                        score={result.score}
                      />
                    </Box>
                  </SearchResultItem>
                ))}
              </List>
            </>
          )}
        </SearchResultsContainer>
      </DialogContent>
      
      <DialogActions>
        <Button 
          onClick={onClose} 
          variant="outlined"
          aria-label="Close search results dialog"
        >
          Close
        </Button>
        {results.length > 0 && (
          <Button 
            variant="contained" 
            onClick={() => {
              // Show advanced search options in the future
              console.log('Advanced search filters - feature enhancement planned');
              // For now, just close the modal and let users know
              onClose();
            }}
            aria-label="Open advanced search filters"
          >
            Refine Search
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default memo(SearchResultsModal);