import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  ButtonGroup,
  Chip,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  alpha,
  useTheme,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  ZoomOutMap,
  FilterList,
  Fullscreen,
  FullscreenExit,
  Refresh,
  Download,
  Settings,
  Timeline,
  BarChart,
  PieChart,
  ShowChart,
  TrendingUp,
  PlayArrow,
  Pause,
  SkipNext,
  SkipPrevious,
} from '@mui/icons-material';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  Brush,
  ReferenceLine,
  ReferenceArea,
  Cell,
  PieChart as RechartsPieChart,
  Pie,
} from 'recharts';

const excelColors = {
  primary: { main: '#0078d4', dark: '#106ebe', light: '#40e0ff' },
  secondary: { main: '#107c10', dark: '#0b5e0b', light: '#6bb26b' },
  accent: { 
    orange: '#ff8c00', 
    red: '#d83b01', 
    blue: '#0078d4',
    green: '#107c10',
    purple: '#5c2e91',
    teal: '#008272'
  },
  background: {
    default: '#faf9f8',
    paper: '#ffffff',
    border: '#e1dfdd',
    hover: '#f3f2f1',
  },
  text: {
    primary: '#323130',
    secondary: '#605e5c',
  }
};

interface InteractiveChartProps {
  data: any[];
  title: string;
  type: 'area' | 'line' | 'bar' | 'pie';
  height?: number;
  enableZoom?: boolean;
  enableBrush?: boolean;
  enableDrillDown?: boolean;
  enableAnimation?: boolean;
  enableRealTime?: boolean;
}

export const InteractiveChart: React.FC<InteractiveChartProps> = ({
  data,
  title,
  type = 'area',
  height = 300,
  enableZoom = true,
  enableBrush = true,
  enableDrillDown = true,
  enableAnimation = true,
  enableRealTime = false,
}) => {
  const [chartType, setChartType] = useState(type);
  const [zoomDomain, setZoomDomain] = useState<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);
  const [selectedDataPoint, setSelectedDataPoint] = useState<any>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [brushData, setBrushData] = useState<any>({ start: 0, end: data.length - 1 });
  const [animationEnabled, setAnimationEnabled] = useState(enableAnimation);
  const [showGrid, setShowGrid] = useState(true);
  const [showTooltip, setShowTooltip] = useState(true);

  const theme = useTheme();

  // Real-time data simulation
  React.useEffect(() => {
    if (enableRealTime && isPlaying) {
      const interval = setInterval(() => {
        // Simulate data updates
        console.log('Real-time data update');
      }, 1000 / playSpeed);
      return () => clearInterval(interval);
    }
  }, [enableRealTime, isPlaying, playSpeed]);

  const handleZoom = useCallback((domain: any) => {
    setZoomDomain(domain);
  }, []);

  const handleResetZoom = () => {
    setZoomDomain(null);
  };

  const handleBrushChange = useCallback((brushData: any) => {
    if (brushData) {
      setBrushData(brushData);
    }
  }, []);

  const handleDataPointClick = (data: any) => {
    if (enableDrillDown) {
      setSelectedDataPoint(data);
      console.log('Drill down to:', data);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const renderChart = () => {
    const commonProps = {
      data,
      margin: { top: 20, right: 30, left: 20, bottom: 20 },
      onMouseDown: enableZoom ? (e: any) => handleZoom(e) : undefined,
    };

    const chartComponents = {
      area: (
        <AreaChart {...commonProps}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={excelColors.background.border} />}
          <XAxis dataKey="time" stroke={excelColors.text.secondary} />
          <YAxis stroke={excelColors.text.secondary} />
          {showTooltip && <ChartTooltip 
            contentStyle={{
              backgroundColor: excelColors.background.paper,
              border: `1px solid ${excelColors.background.border}`,
              borderRadius: 8,
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            }}
            cursor={{ stroke: excelColors.primary.main, strokeWidth: 2 }}
          />}
          <Legend />
          <Area
            type="monotone"
            dataKey="transactions"
            stackId="1"
            stroke={excelColors.primary.main}
            fill={alpha(excelColors.primary.main, 0.3)}
            strokeWidth={3}
            animationDuration={animationEnabled ? 1500 : 0}
            animationEasing="ease-out"
            onClick={handleDataPointClick}
          />
          <Area
            type="monotone"
            dataKey="fraudulent"
            stackId="1"
            stroke={excelColors.accent.red}
            fill={alpha(excelColors.accent.red, 0.3)}
            strokeWidth={3}
            animationDuration={animationEnabled ? 1500 : 0}
            animationEasing="ease-out"
            onClick={handleDataPointClick}
          />
          {enableBrush && <Brush dataKey="time" height={30} stroke={excelColors.primary.main} onChange={handleBrushChange} />}
        </AreaChart>
      ),
      line: (
        <LineChart {...commonProps}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={excelColors.background.border} />}
          <XAxis dataKey="time" stroke={excelColors.text.secondary} />
          <YAxis stroke={excelColors.text.secondary} />
          {showTooltip && <ChartTooltip 
            contentStyle={{
              backgroundColor: excelColors.background.paper,
              border: `1px solid ${excelColors.background.border}`,
              borderRadius: 8,
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            }}
            cursor={{ stroke: excelColors.primary.main, strokeWidth: 2 }}
          />}
          <Legend />
          <Line
            type="monotone"
            dataKey="transactions"
            stroke={excelColors.primary.main}
            strokeWidth={4}
            dot={{ fill: excelColors.primary.main, strokeWidth: 2, r: 6 }}
            activeDot={{ r: 8, stroke: excelColors.primary.main, strokeWidth: 2, fill: '#fff' }}
            animationDuration={animationEnabled ? 2000 : 0}
            onClick={handleDataPointClick}
          />
          <Line
            type="monotone"
            dataKey="fraudulent"
            stroke={excelColors.accent.red}
            strokeWidth={4}
            dot={{ fill: excelColors.accent.red, strokeWidth: 2, r: 6 }}
            activeDot={{ r: 8, stroke: excelColors.accent.red, strokeWidth: 2, fill: '#fff' }}
            animationDuration={animationEnabled ? 2000 : 0}
            onClick={handleDataPointClick}
          />
          {enableBrush && <Brush dataKey="time" height={30} stroke={excelColors.primary.main} onChange={handleBrushChange} />}
        </LineChart>
      ),
      bar: (
        <RechartsBarChart {...commonProps}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke={excelColors.background.border} />}
          <XAxis dataKey="time" stroke={excelColors.text.secondary} />
          <YAxis stroke={excelColors.text.secondary} />
          {showTooltip && <ChartTooltip 
            contentStyle={{
              backgroundColor: excelColors.background.paper,
              border: `1px solid ${excelColors.background.border}`,
              borderRadius: 8,
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            }}
          />}
          <Legend />
          <Bar 
            dataKey="transactions" 
            fill={excelColors.primary.main}
            radius={[4, 4, 0, 0]}
            animationDuration={animationEnabled ? 1500 : 0}
            onClick={handleDataPointClick}
          />
          <Bar 
            dataKey="fraudulent" 
            fill={excelColors.accent.red}
            radius={[4, 4, 0, 0]}
            animationDuration={animationEnabled ? 1500 : 0}
            onClick={handleDataPointClick}
          />
          {enableBrush && <Brush dataKey="time" height={30} stroke={excelColors.primary.main} onChange={handleBrushChange} />}
        </RechartsBarChart>
      ),
    };

    return chartComponents[chartType as keyof typeof chartComponents] || chartComponents.area;
  };

  return (
    <Card sx={{ 
      height: isFullscreen ? '100vh' : 'auto',
      position: isFullscreen ? 'fixed' : 'relative',
      top: isFullscreen ? 0 : 'auto',
      left: isFullscreen ? 0 : 'auto',
      right: isFullscreen ? 0 : 'auto',
      bottom: isFullscreen ? 0 : 'auto',
      zIndex: isFullscreen ? 9999 : 'auto',
      transition: 'all 0.3s ease',
    }}>
      <CardContent>
        {/* Chart Header with Controls */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
            {title}
            {selectedDataPoint && (
              <Chip 
                label={`Drill-down: ${selectedDataPoint.time}`}
                size="small"
                sx={{ ml: 1 }}
                onDelete={() => setSelectedDataPoint(null)}
              />
            )}
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Chart Type Switcher */}
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Area Chart">
                <Button 
                  onClick={() => setChartType('area')}
                  variant={chartType === 'area' ? 'contained' : 'outlined'}
                >
                  <Timeline />
                </Button>
              </Tooltip>
              <Tooltip title="Line Chart">
                <Button 
                  onClick={() => setChartType('line')}
                  variant={chartType === 'line' ? 'contained' : 'outlined'}
                >
                  <ShowChart />
                </Button>
              </Tooltip>
              <Tooltip title="Bar Chart">
                <Button 
                  onClick={() => setChartType('bar')}
                  variant={chartType === 'bar' ? 'contained' : 'outlined'}
                >
                  <BarChart />
                </Button>
              </Tooltip>
            </ButtonGroup>

            {/* Playback Controls */}
            {enableRealTime && (
              <ButtonGroup size="small" variant="outlined">
                <Tooltip title="Previous">
                  <Button><SkipPrevious /></Button>
                </Tooltip>
                <Tooltip title={isPlaying ? "Pause" : "Play"}>
                  <Button onClick={togglePlayback}>
                    {isPlaying ? <Pause /> : <PlayArrow />}
                  </Button>
                </Tooltip>
                <Tooltip title="Next">
                  <Button><SkipNext /></Button>
                </Tooltip>
              </ButtonGroup>
            )}

            {/* Zoom Controls */}
            {enableZoom && (
              <>
                <Tooltip title="Reset Zoom">
                  <IconButton size="small" onClick={handleResetZoom}>
                    <ZoomOutMap />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom In">
                  <IconButton size="small"><ZoomIn /></IconButton>
                </Tooltip>
                <Tooltip title="Zoom Out">
                  <IconButton size="small"><ZoomOut /></IconButton>
                </Tooltip>
              </>
            )}

            {/* Chart Settings */}
            <Tooltip title="Chart Settings">
              <IconButton size="small" onClick={handleMenuOpen}>
                <Settings />
              </IconButton>
            </Tooltip>

            {/* Fullscreen Toggle */}
            <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
              <IconButton size="small" onClick={toggleFullscreen}>
                {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Real-time Controls */}
        {enableRealTime && (
          <Box sx={{ mb: 2, p: 2, bgcolor: alpha(excelColors.primary.main, 0.1), borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="body2">Speed:</Typography>
              <Slider
                value={playSpeed}
                onChange={(e, value) => setPlaySpeed(value as number)}
                min={0.5}
                max={3}
                step={0.5}
                marks={[
                  { value: 0.5, label: '0.5x' },
                  { value: 1, label: '1x' },
                  { value: 2, label: '2x' },
                  { value: 3, label: '3x' },
                ]}
                sx={{ width: 150 }}
              />
              <Chip 
                label={isPlaying ? "LIVE" : "PAUSED"}
                color={isPlaying ? "success" : "default"}
                variant="filled"
                size="small"
              />
            </Box>
          </Box>
        )}

        {/* Interactive Chart */}
        <ResponsiveContainer width="100%" height={isFullscreen ? 'calc(100vh - 200px)' : height}>
          {renderChart()}
        </ResponsiveContainer>

        {/* Settings Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          PaperProps={{
            sx: {
              minWidth: 250,
              p: 1,
            }
          }}
        >
          <MenuItem>
            <FormControlLabel
              control={
                <Switch
                  checked={animationEnabled}
                  onChange={(e) => setAnimationEnabled(e.target.checked)}
                />
              }
              label="Enable Animations"
            />
          </MenuItem>
          <MenuItem>
            <FormControlLabel
              control={
                <Switch
                  checked={showGrid}
                  onChange={(e) => setShowGrid(e.target.checked)}
                />
              }
              label="Show Grid"
            />
          </MenuItem>
          <MenuItem>
            <FormControlLabel
              control={
                <Switch
                  checked={showTooltip}
                  onChange={(e) => setShowTooltip(e.target.checked)}
                />
              }
              label="Show Tooltips"
            />
          </MenuItem>
          <MenuItem onClick={() => { /* Export logic */ handleMenuClose(); }}>
            <Download sx={{ mr: 1 }} />
            Export Chart
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default InteractiveChart;