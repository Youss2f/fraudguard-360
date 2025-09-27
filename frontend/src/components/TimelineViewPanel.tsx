/**
 * Professional Timeline View Panel Component
 * Chronological fraud event analysis with pattern visualization
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
  Grid,
  Chip,
  IconButton,
  Tabs,
  Tab,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  // Timeline components replaced with custom implementation
  Paper,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  LinearProgress,
  styled,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  Close,
  Security,
  Warning,
  Person,
  CreditCard,
  AccountBalance,
  LocationOn,
  Phone,
  Computer,
  TrendingUp,
  Flag,
  Visibility,
  PlayArrow,
  Pause,
  FastForward,
  FastRewind,
  ZoomIn,
  ZoomOut,
  FilterList,
  Search,
  Download,
  Share,
} from '@mui/icons-material';
import { format, formatDistanceToNow, startOfDay, endOfDay } from 'date-fns';
import { customColors } from '../theme/enterpriseTheme';

const TimelineCard = styled(Card)(({ theme, severity }: { theme?: any; severity: string }) => {
  const getBorderColor = () => {
    switch (severity) {
      case 'critical': return customColors.error[500];
      case 'high': return customColors.warning[500];
      case 'medium': return customColors.primary[500];
      case 'low': return customColors.success[500];
      default: return customColors.neutral[200];
    }
  };

  return {
    backgroundColor: customColors.background.paper,
    border: `1px solid ${customColors.neutral[200]}`,
    borderLeft: `4px solid ${getBorderColor()}`,
    marginBottom: '16px',
    transition: 'all 0.2s ease',
    '&:hover': {
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
      transform: 'translateX(4px)',
    },
  };
});

interface TimelineViewPanelProps {
  open: boolean;
  onClose: () => void;
}

interface FraudEvent {
  id: string;
  timestamp: Date;
  type: 'alert' | 'transaction' | 'login' | 'investigation' | 'resolution';
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  actor: string;
  target: string;
  location: string;
  amount?: number;
  status: 'active' | 'investigating' | 'resolved' | 'false_positive';
  tags: string[];
  relatedEvents: string[];
}

interface Pattern {
  id: string;
  name: string;
  description: string;
  confidence: number;
  events: string[];
  timespan: { start: Date; end: Date };
  riskScore: number;
}

const TimelineViewPanel: React.FC<TimelineViewPanelProps> = ({ 
  open, 
  onClose 
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [events, setEvents] = useState<FraudEvent[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<FraudEvent | null>(null);
  const [timeRange, setTimeRange] = useState('24h');
  const [eventFilter, setEventFilter] = useState('all');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    // Generate mock fraud events
    const now = new Date();
    const mockEvents: FraudEvent[] = [
      {
        id: 'evt-001',
        timestamp: new Date(now.getTime() - 15 * 60 * 1000),
        type: 'alert',
        severity: 'critical',
        title: 'Suspicious Card Testing Pattern',
        description: 'Multiple failed transactions detected on card ending in 1234',
        actor: 'Unknown',
        target: 'Card *1234',
        location: 'New York, NY',
        amount: 0,
        status: 'active',
        tags: ['card-testing', 'high-frequency', 'multiple-failures'],
        relatedEvents: ['evt-002', 'evt-003']
      },
      {
        id: 'evt-002',
        timestamp: new Date(now.getTime() - 45 * 60 * 1000),
        type: 'transaction',
        severity: 'high',
        title: 'Large Transaction Attempt',
        description: 'Transaction of $5,000 attempted and failed',
        actor: 'John Smith',
        target: 'Account ACC-1234',
        location: 'Los Angeles, CA',
        amount: 5000,
        status: 'investigating',
        tags: ['large-amount', 'failed-transaction', 'out-of-pattern'],
        relatedEvents: ['evt-001', 'evt-004']
      },
      {
        id: 'evt-003',
        timestamp: new Date(now.getTime() - 75 * 60 * 1000),
        type: 'login',
        severity: 'medium',
        title: 'Unusual Login Location',
        description: 'Login from new geographic location detected',
        actor: 'John Smith',
        target: 'User Account',
        location: 'Miami, FL',
        status: 'resolved',
        tags: ['geo-anomaly', 'new-location', 'successful-login'],
        relatedEvents: ['evt-002']
      },
      {
        id: 'evt-004',
        timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000),
        type: 'investigation',
        severity: 'high',
        title: 'Investigation Initiated',
        description: 'Fraud analyst Sarah Johnson opened investigation CASE-2024-0891',
        actor: 'Sarah Johnson',
        target: 'CASE-2024-0891',
        location: 'System',
        status: 'investigating',
        tags: ['investigation', 'analyst-action', 'case-opened'],
        relatedEvents: ['evt-002', 'evt-005']
      },
      {
        id: 'evt-005',
        timestamp: new Date(now.getTime() - 3 * 60 * 60 * 1000),
        type: 'alert',
        severity: 'medium',
        title: 'Velocity Check Triggered',
        description: 'Multiple transactions within short time frame detected',
        actor: 'System',
        target: 'Account ACC-1234',
        location: 'Multiple',
        status: 'resolved',
        tags: ['velocity-check', 'time-based', 'multiple-locations'],
        relatedEvents: ['evt-004']
      }
    ];
    setEvents(mockEvents.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()));

    // Generate mock patterns
    const mockPatterns: Pattern[] = [
      {
        id: 'pat-001',
        name: 'Card Testing Pattern',
        description: 'Systematic testing of stolen card numbers with small amounts',
        confidence: 95,
        events: ['evt-001', 'evt-002'],
        timespan: { 
          start: new Date(now.getTime() - 2 * 60 * 60 * 1000), 
          end: new Date(now.getTime() - 15 * 60 * 1000) 
        },
        riskScore: 87
      },
      {
        id: 'pat-002',
        name: 'Account Takeover Sequence',
        description: 'Login from new location followed by high-value transaction attempts',
        confidence: 78,
        events: ['evt-003', 'evt-002', 'evt-001'],
        timespan: { 
          start: new Date(now.getTime() - 3 * 60 * 60 * 1000), 
          end: new Date(now.getTime() - 15 * 60 * 1000) 
        },
        riskScore: 72
      }
    ];
    setPatterns(mockPatterns);
  }, []);

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'alert': return <Security />;
      case 'transaction': return <CreditCard />;
      case 'login': return <Person />;
      case 'investigation': return <Visibility />;
      case 'resolution': return <Flag />;
      default: return <Warning />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'primary';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return customColors.error[500];
      case 'investigating': return customColors.warning[500];
      case 'resolved': return customColors.success[500];
      case 'false_positive': return customColors.neutral[500];
      default: return customColors.neutral[500];
    }
  };

  const handlePlaybackToggle = () => {
    setIsPlaying(!isPlaying);
    // In a real implementation, this would control timeline playback
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 5));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 0.5));
  };

  const getFilteredEvents = () => {
    let filtered = events;
    
    if (eventFilter !== 'all') {
      filtered = filtered.filter(event => event.type === eventFilter);
    }
    
    if (searchQuery) {
      filtered = filtered.filter(event => 
        event.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.actor.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    return filtered;
  };

  const renderTimelineView = () => (
    <Box p={3}>
      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                >
                  <MenuItem value="1h">Last Hour</MenuItem>
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Event Type</InputLabel>
                <Select
                  value={eventFilter}
                  onChange={(e) => setEventFilter(e.target.value)}
                >
                  <MenuItem value="all">All Events</MenuItem>
                  <MenuItem value="alert">Alerts Only</MenuItem>
                  <MenuItem value="transaction">Transactions</MenuItem>
                  <MenuItem value="login">Logins</MenuItem>
                  <MenuItem value="investigation">Investigations</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                size="small"
                placeholder="Search events..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                }}
              />
            </Grid>
            
            <Grid item xs={12} md={2}>
              <Box display="flex" gap={1}>
                <Tooltip title="Zoom In">
                  <IconButton size="small" onClick={handleZoomIn}>
                    <ZoomIn />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom Out">
                  <IconButton size="small" onClick={handleZoomOut}>
                    <ZoomOut />
                  </IconButton>
                </Tooltip>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Playback Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="center" gap={2}>
            <IconButton onClick={() => setPlaybackSpeed(0.5)}>
              <FastRewind />
            </IconButton>
            
            <IconButton onClick={handlePlaybackToggle} color="primary">
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>
            
            <IconButton onClick={() => setPlaybackSpeed(2)}>
              <FastForward />
            </IconButton>
            
            <Typography variant="body2" sx={{ minWidth: 80 }}>
              Speed: {playbackSpeed}x
            </Typography>
            
            <Typography variant="body2" sx={{ minWidth: 100 }}>
              Zoom: {Math.round(zoomLevel * 100)}%
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Custom Timeline Implementation */}
      <Box sx={{ position: 'relative' }}>
        {/* Timeline Connector Line */}
        <Box
          sx={{
            position: 'absolute',
            left: '60px',
            top: 0,
            bottom: 0,
            width: '2px',
            backgroundColor: customColors.primary[200],
            zIndex: 0,
          }}
        />
        
        {getFilteredEvents().map((event, index) => (
          <Box key={event.id} sx={{ position: 'relative', mb: 3 }}>
            {/* Time Label */}
            <Box
              sx={{
                position: 'absolute',
                left: 0,
                top: '20px',
                width: '50px',
                textAlign: 'center',
              }}
            >
              <Typography variant="caption" display="block" sx={{ fontWeight: 600 }}>
                {format(event.timestamp, 'HH:mm')}
              </Typography>
              <Typography variant="caption" display="block" color="text.secondary">
                {format(event.timestamp, 'MMM dd')}
              </Typography>
            </Box>
            
            {/* Timeline Dot */}
            <Box
              sx={{
                position: 'absolute',
                left: '51px',
                top: '20px',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                backgroundColor: getSeverityColor(event.severity) === 'error' ? customColors.error[500] :
                                getSeverityColor(event.severity) === 'warning' ? customColors.warning[500] :
                                getSeverityColor(event.severity) === 'success' ? customColors.success[500] :
                                customColors.primary[500],
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                zIndex: 1,
                fontSize: '12px',
              }}
            >
              {getEventIcon(event.type)}
            </Box>
            
            {/* Event Card */}
            <Box sx={{ ml: '90px' }}>
              <TimelineCard severity={event.severity}>
                <CardContent onClick={() => setSelectedEvent(event)} sx={{ cursor: 'pointer' }}>
                  <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {event.title}
                    </Typography>
                    <Box display="flex" gap={1}>
                      <Chip 
                        size="small" 
                        label={event.severity}
                        color={getSeverityColor(event.severity) as any}
                      />
                      <Chip 
                        size="small" 
                        label={event.status}
                        sx={{ 
                          bgcolor: getStatusColor(event.status),
                          color: 'white'
                        }}
                      />
                    </Box>
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {event.description}
                  </Typography>
                  
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">
                        Actor: {event.actor}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">
                        Target: {event.target}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">
                        Location: {event.location}
                      </Typography>
                    </Grid>
                    {event.amount && (
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Amount: ${event.amount.toLocaleString()}
                        </Typography>
                      </Grid>
                    )}
                  </Grid>
                  
                  <Box mt={1}>
                    {event.tags.map((tag) => (
                      <Chip 
                        key={tag}
                        size="small" 
                        label={tag} 
                        variant="outlined"
                        sx={{ mr: 0.5, mt: 0.5 }}
                      />
                    ))}
                  </Box>
                </CardContent>
              </TimelineCard>
            </Box>
          </Box>
        ))}
      </Box>
    </Box>
  );

  const renderPatternAnalysis = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Detected Fraud Patterns
      </Typography>
      
      <Grid container spacing={2}>
        {patterns.map((pattern) => (
          <Grid item xs={12} md={6} key={pattern.id}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="between" alignItems="flex-start" mb={2}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {pattern.name}
                  </Typography>
                  <Chip 
                    size="small" 
                    label={`${pattern.confidence}% confidence`}
                    color={pattern.confidence > 80 ? 'success' : pattern.confidence > 60 ? 'warning' : 'error'}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {pattern.description}
                </Typography>
                
                <Box mb={2}>
                  <Typography variant="caption" color="text.secondary">
                    Risk Score
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={pattern.riskScore}
                    color={pattern.riskScore > 80 ? 'error' : pattern.riskScore > 60 ? 'warning' : 'success'}
                    sx={{ mt: 0.5 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {pattern.riskScore}%
                  </Typography>
                </Box>
                
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Timespan: {formatDistanceToNow(pattern.timespan.start)} - {formatDistanceToNow(pattern.timespan.end)}
                </Typography>
                
                <Typography variant="caption" color="text.secondary">
                  Related Events: {pattern.events.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const tabLabels = [
    'Timeline View',
    'Pattern Analysis'
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
          <TimelineIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Timeline Analysis
          </Typography>
        </Box>
        <Box display="flex" gap={1}>
          <IconButton size="small" title="Download Timeline">
            <Download />
          </IconButton>
          <IconButton size="small" title="Share Timeline">
            <Share />
          </IconButton>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
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
        {currentTab === 0 && renderTimelineView()}
        {currentTab === 1 && renderPatternAnalysis()}
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

export default TimelineViewPanel;