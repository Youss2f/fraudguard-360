/**
 * Professional Investigation Tools Panel Component
 * Advanced investigation features for fraud analysts
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  IconButton,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  styled,
} from '@mui/material';
import {
  Search,
  Close,
  Timeline,
  AccountTree,
  PersonSearch,
  TrendingUp,
  Security,
  Link,
  NetworkCheck,
  Psychology,
  Assignment,
  Visibility,
  PlayArrow,
  Stop,
  Refresh,
  Download,
  Share,
  Flag,
  Warning,
  CheckCircle,
  Person,
  CreditCard,
  AccountBalance,
  LocationOn,
  Phone,
  Email,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const ToolCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  cursor: 'pointer',
  '&:hover': {
    borderColor: customColors.primary[500],
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)',
  },
  '&.active': {
    borderColor: customColors.primary[500],
    backgroundColor: customColors.primary[50],
  },
}));

interface InvestigationToolsPanelProps {
  open: boolean;
  onClose: () => void;
}

interface InvestigationCase {
  id: string;
  title: string;
  priority: 'high' | 'medium' | 'low';
  status: 'active' | 'pending' | 'closed';
  assignee: string;
  createdDate: Date;
  entities: string[];
}

interface NetworkNode {
  id: string;
  type: 'user' | 'account' | 'transaction' | 'device' | 'location';
  label: string;
  riskScore: number;
  connections: number;
}

const InvestigationToolsPanel: React.FC<InvestigationToolsPanelProps> = ({ 
  open, 
  onClose 
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeInvestigation, setActiveInvestigation] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cases, setCases] = useState<InvestigationCase[]>([]);
  const [networkNodes, setNetworkNodes] = useState<NetworkNode[]>([]);

  useEffect(() => {
    // Generate mock investigation cases
    const mockCases: InvestigationCase[] = [
      {
        id: 'CASE-2024-0891',
        title: 'Account Takeover Investigation',
        priority: 'high',
        status: 'active',
        assignee: 'Sarah Johnson',
        createdDate: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
        entities: ['USR-7529', 'ACC-4821', 'TXN-9934']
      },
      {
        id: 'CASE-2024-0892',
        title: 'Card Testing Pattern Analysis',
        priority: 'medium',
        status: 'pending',
        assignee: 'Mike Chen',
        createdDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
        entities: ['USR-3412', 'USR-7891', 'USR-5623']
      },
      {
        id: 'CASE-2024-0893',
        title: 'Money Laundering Network',
        priority: 'high',
        status: 'active',
        assignee: 'Emma Davis',
        createdDate: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
        entities: ['ACC-1234', 'ACC-5678', 'ACC-9012', 'TXN-7745']
      }
    ];
    setCases(mockCases);

    // Generate mock network nodes
    const mockNodes: NetworkNode[] = [
      { id: 'USR-7529', type: 'user', label: 'John Smith', riskScore: 95, connections: 12 },
      { id: 'ACC-4821', type: 'account', label: 'Checking ***1234', riskScore: 87, connections: 8 },
      { id: 'TXN-9934', type: 'transaction', label: '$2,500 Transfer', riskScore: 92, connections: 5 },
      { id: 'DEV-5521', type: 'device', label: 'iPhone 12 Pro', riskScore: 76, connections: 15 },
      { id: 'LOC-8834', type: 'location', label: 'New York, NY', riskScore: 45, connections: 23 },
    ];
    setNetworkNodes(mockNodes);
  }, []);

  const investigationTools = [
    {
      id: 'network-expansion',
      title: 'Network Expansion',
      description: 'Expand investigation network to find connected entities',
      icon: <AccountTree />,
      category: 'Analysis'
    },
    {
      id: 'pattern-analysis',
      title: 'Pattern Analysis',
      description: 'Identify suspicious patterns and behaviors',
      icon: <Psychology />,
      category: 'Analysis'
    },
    {
      id: 'timeline-analysis',
      title: 'Timeline Analysis',
      description: 'Chronological analysis of events and transactions',
      icon: <Timeline />,
      category: 'Analysis'
    },
    {
      id: 'entity-search',
      title: 'Entity Search',
      description: 'Search and analyze specific entities',
      icon: <PersonSearch />,
      category: 'Search'
    },
    {
      id: 'risk-assessment',
      title: 'Risk Assessment',
      description: 'Calculate and analyze risk scores',
      icon: <Security />,
      category: 'Assessment'
    },
    {
      id: 'relationship-mapping',
      title: 'Relationship Mapping',
      description: 'Map relationships between entities',
      icon: <NetworkCheck />,
      category: 'Visualization'
    }
  ];

  const handleStartAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  const handleStopAnalysis = () => {
    setIsAnalyzing(false);
    setAnalysisProgress(0);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return customColors.success[500];
      case 'pending': return customColors.warning[500];
      case 'closed': return customColors.neutral[500];
      default: return customColors.neutral[500];
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return customColors.error[500];
      case 'medium': return customColors.warning[500];
      case 'low': return customColors.success[500];
      default: return customColors.neutral[500];
    }
  };

  const getEntityIcon = (type: string) => {
    switch (type) {
      case 'user': return <Person />;
      case 'account': return <AccountBalance />;
      case 'transaction': return <CreditCard />;
      case 'device': return <Phone />;
      case 'location': return <LocationOn />;
      default: return <Search />;
    }
  };

  const renderInvestigationTools = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Investigation Tools
      </Typography>
      
      <Grid container spacing={2}>
        {investigationTools.map((tool) => (
          <Grid item xs={12} sm={6} md={4} key={tool.id}>
            <ToolCard 
              className={selectedTool === tool.id ? 'active' : ''}
              onClick={() => setSelectedTool(tool.id)}
            >
              <CardContent>
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                  <Box 
                    sx={{ 
                      color: customColors.primary[500],
                      display: 'flex',
                      alignItems: 'center'
                    }}
                  >
                    {tool.icon}
                  </Box>
                  <Box>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {tool.title}
                    </Typography>
                    <Chip 
                      size="small" 
                      label={tool.category}
                      variant="outlined"
                    />
                  </Box>
                </Box>
                
                <Typography variant="body2" color="text.secondary">
                  {tool.description}
                </Typography>
              </CardContent>
              
              <CardActions>
                <Button 
                  size="small" 
                  startIcon={<PlayArrow />}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleStartAnalysis();
                  }}
                  disabled={isAnalyzing}
                >
                  Run Analysis
                </Button>
              </CardActions>
            </ToolCard>
          </Grid>
        ))}
      </Grid>

      {/* Analysis Progress */}
      {(isAnalyzing || analysisProgress > 0) && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Analysis Progress
              </Typography>
              <Box display="flex" gap={1}>
                {isAnalyzing && (
                  <Button
                    size="small"
                    startIcon={<Stop />}
                    onClick={handleStopAnalysis}
                    color="error"
                  >
                    Stop
                  </Button>
                )}
                <Button size="small" startIcon={<Refresh />}>
                  Refresh
                </Button>
              </Box>
            </Box>
            
            <LinearProgress 
              variant="determinate" 
              value={analysisProgress}
              sx={{ mb: 1 }}
            />
            <Typography variant="body2" color="text.secondary">
              {analysisProgress}% complete - Analyzing network patterns...
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );

  const renderActiveCases = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Active Cases
      </Typography>
      
      <Grid container spacing={2}>
        {cases.map((caseItem) => (
          <Grid item xs={12} md={6} key={caseItem.id}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {caseItem.title}
                  </Typography>
                  <Box display="flex" gap={1}>
                    <Chip 
                      size="small" 
                      label={caseItem.priority}
                      sx={{ 
                        bgcolor: getPriorityColor(caseItem.priority),
                        color: 'white'
                      }}
                    />
                    <Chip 
                      size="small" 
                      label={caseItem.status}
                      sx={{ 
                        bgcolor: getStatusColor(caseItem.status),
                        color: 'white'
                      }}
                    />
                  </Box>
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Case ID: {caseItem.id}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Assignee: {caseItem.assignee}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Created: {caseItem.createdDate.toLocaleDateString()}
                </Typography>
                
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Entities ({caseItem.entities.length}):
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={0.5} mt={1}>
                    {caseItem.entities.map((entity) => (
                      <Chip 
                        key={entity}
                        size="small" 
                        label={entity} 
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </CardContent>
              
              <CardActions>
                <Button 
                  size="small" 
                  startIcon={<Visibility />}
                  onClick={() => setActiveInvestigation(caseItem.id)}
                >
                  View Details
                </Button>
                <Button size="small" startIcon={<Assignment />}>
                  Add Note
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const renderNetworkAnalysis = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Network Analysis
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Search Entities
          </Typography>
          
          <Box display="flex" gap={2} mb={2}>
            <TextField
              fullWidth
              size="small"
              placeholder="Search by ID, name, or attributes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
              }}
            />
            <Button variant="contained" startIcon={<Search />}>
              Search
            </Button>
          </Box>
        </CardContent>
      </Card>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Entity</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Risk Score</TableCell>
              <TableCell>Connections</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {networkNodes.map((node) => (
              <TableRow key={node.id}>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    {getEntityIcon(node.type)}
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {node.label}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {node.id}
                      </Typography>
                    </Box>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip 
                    size="small" 
                    label={node.type} 
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    <LinearProgress 
                      variant="determinate" 
                      value={node.riskScore}
                      sx={{ width: 60, height: 6 }}
                      color={node.riskScore > 80 ? 'error' : node.riskScore > 60 ? 'warning' : 'success'}
                    />
                    <Typography variant="body2">
                      {node.riskScore}%
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>{node.connections}</TableCell>
                <TableCell>
                  <IconButton size="small" title="Expand Network">
                    <AccountTree />
                  </IconButton>
                  <IconButton size="small" title="View Details">
                    <Visibility />
                  </IconButton>
                  <IconButton size="small" title="Flag Entity">
                    <Flag />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );

  const tabLabels = [
    'Investigation Tools',
    'Active Cases',
    'Network Analysis'
  ];

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
          <Search color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Investigation Tools
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>

      <Box sx={{ backgroundColor: customColors.background.ribbon }}>
        <Tabs
          value={currentTab}
          onChange={(_, newValue) => setCurrentTab(newValue)}
          variant="fullWidth"
        >
          {tabLabels.map((label, index) => (
            <Tab key={label} label={label} />
          ))}
        </Tabs>
      </Box>

      <DialogContent sx={{ p: 0, height: '60vh', overflow: 'auto' }}>
        {currentTab === 0 && renderInvestigationTools()}
        {currentTab === 1 && renderActiveCases()}
        {currentTab === 2 && renderNetworkAnalysis()}
      </DialogContent>

      <DialogActions sx={{ 
        p: 2, 
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default InvestigationToolsPanel;