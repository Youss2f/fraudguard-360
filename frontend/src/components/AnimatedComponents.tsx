import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Tooltip,
  Grow,
  Zoom,
  Slide,
  Fade,
  Pulse,
  useTheme,
  keyframes,
  styled,
  alpha,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Speed,
  Security,
  Warning,
  CheckCircle,
  Error,
  Info,
  Refresh,
  Fullscreen,
  Close,
  ExpandMore,
  ExpandLess,
} from '@mui/icons-material';

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

// Animated keyframes
const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const glow = keyframes`
  0% { box-shadow: 0 0 5px ${alpha(excelColors.primary.main, 0.5)}; }
  50% { box-shadow: 0 0 20px ${alpha(excelColors.primary.main, 0.8)}, 0 0 30px ${alpha(excelColors.primary.main, 0.6)}; }
  100% { box-shadow: 0 0 5px ${alpha(excelColors.primary.main, 0.5)}; }
`;

const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
`;

const slideInLeft = keyframes`
  0% { transform: translateX(-100%); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
`;

const slideInRight = keyframes`
  0% { transform: translateX(100%); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
`;

const slideInUp = keyframes`
  0% { transform: translateY(100%); opacity: 0; }
  100% { transform: translateY(0); opacity: 1; }
`;

// Styled animated components
const AnimatedCard = styled(Card)(({ theme }) => ({
  cursor: 'pointer',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  position: 'relative',
  overflow: 'hidden',
  '&:before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: '-100%',
    width: '100%',
    height: '100%',
    background: `linear-gradient(90deg, transparent, ${alpha(excelColors.primary.main, 0.2)}, transparent)`,
    transition: 'left 0.5s',
  },
  '&:hover': {
    transform: 'translateY(-8px) scale(1.02)',
    boxShadow: `0 10px 25px ${alpha(excelColors.primary.main, 0.3)}`,
    '&:before': {
      left: '100%',
    },
  },
  '&:active': {
    transform: 'translateY(-4px) scale(0.98)',
  },
}));

const PulsingIcon = styled(Box)(({ theme }) => ({
  animation: `${pulse} 2s infinite`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const GlowingCard = styled(Card)(({ theme }) => ({
  animation: `${glow} 3s infinite`,
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    animation: 'none',
    boxShadow: `0 0 30px ${alpha(excelColors.accent.orange, 0.8)}`,
    transform: 'scale(1.05)',
  },
}));

const FloatingButton = styled(IconButton)(({ theme }) => ({
  animation: `${float} 3s ease-in-out infinite`,
  '&:hover': {
    animation: 'none',
    transform: 'scale(1.2)',
  },
}));

const SlideInCard = styled(Card)<{ direction: 'left' | 'right' | 'up' }>(({ direction }) => ({
  animation: `${
    direction === 'left' ? slideInLeft : 
    direction === 'right' ? slideInRight : 
    slideInUp
  } 0.6s ease-out`,
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'scale(1.03)',
    boxShadow: '0 8px 25px rgba(0,0,0,0.15)',
  },
}));

interface AnimatedComponentProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  animated?: boolean;
  glowing?: boolean;
  floating?: boolean;
  direction?: 'left' | 'right' | 'up';
  delay?: number;
  onClick?: () => void;
}

export const AnimatedKPICard: React.FC<AnimatedComponentProps> = ({
  title,
  value,
  change,
  icon,
  color = 'primary',
  animated = true,
  glowing = false,
  floating = false,
  direction = 'up',
  delay = 0,
  onClick,
}) => {
  const [visible, setVisible] = useState(false);
  const [count, setCount] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  useEffect(() => {
    if (visible && typeof value === 'number') {
      const targetValue = value;
      const increment = targetValue / 50;
      const timer = setInterval(() => {
        setCount(prev => {
          if (prev >= targetValue) {
            clearInterval(timer);
            return targetValue;
          }
          return prev + increment;
        });
      }, 20);
      return () => clearInterval(timer);
    }
  }, [visible, value]);

  const getColorValue = (colorName: string) => {
    switch (colorName) {
      case 'primary': return excelColors.primary.main;
      case 'secondary': return excelColors.secondary.main;
      case 'success': return excelColors.accent.green;
      case 'warning': return excelColors.accent.orange;
      case 'error': return excelColors.accent.red;
      case 'info': return excelColors.accent.blue;
      default: return excelColors.primary.main;
    }
  };

  const CardComponent = glowing ? GlowingCard : animated ? SlideInCard : AnimatedCard;

  return (
    <Grow in={visible} timeout={600}>
      <CardComponent direction={direction} onClick={onClick}>
        <CardContent sx={{ position: 'relative', overflow: 'hidden' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {title}
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: getColorValue(color) }}>
                {typeof value === 'number' ? Math.round(count).toLocaleString() : value}
              </Typography>
              {change !== undefined && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  {change > 0 ? (
                    <TrendingUp sx={{ color: excelColors.accent.green, mr: 0.5 }} />
                  ) : (
                    <TrendingDown sx={{ color: excelColors.accent.red, mr: 0.5 }} />
                  )}
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: change > 0 ? excelColors.accent.green : excelColors.accent.red,
                      fontWeight: 'bold'
                    }}
                  >
                    {Math.abs(change)}%
                  </Typography>
                </Box>
              )}
            </Box>
            {floating ? (
              <FloatingButton size="large" sx={{ color: getColorValue(color) }}>
                {icon}
              </FloatingButton>
            ) : (
              <PulsingIcon sx={{ color: getColorValue(color) }}>
                {icon}
              </PulsingIcon>
            )}
          </Box>
          
          {/* Animated background effect */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: 4,
              background: `linear-gradient(90deg, ${alpha(getColorValue(color), 0.3)}, ${getColorValue(color)})`,
              transform: visible ? 'scaleX(1)' : 'scaleX(0)',
              transformOrigin: 'left',
              transition: 'transform 1s ease-out',
            }}
          />
        </CardContent>
      </CardComponent>
    </Grow>
  );
};

// Interactive Alert Component with animations
interface AnimatedAlertProps {
  severity: 'success' | 'info' | 'warning' | 'error';
  title: string;
  message: string;
  action?: React.ReactNode;
  onClose?: () => void;
  animated?: boolean;
}

export const AnimatedAlert: React.FC<AnimatedAlertProps> = ({
  severity,
  title,
  message,
  action,
  onClose,
  animated = true,
}) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(true);
  }, []);

  const getIcon = () => {
    switch (severity) {
      case 'success': return <CheckCircle />;
      case 'info': return <Info />;
      case 'warning': return <Warning />;
      case 'error': return <Error />;
    }
  };

  const getColor = () => {
    switch (severity) {
      case 'success': return excelColors.accent.green;
      case 'info': return excelColors.accent.blue;
      case 'warning': return excelColors.accent.orange;
      case 'error': return excelColors.accent.red;
    }
  };

  return (
    <Slide direction="down" in={visible} mountOnEnter unmountOnExit>
      <Card
        sx={{
          mb: 2,
          borderLeft: `4px solid ${getColor()}`,
          backgroundColor: alpha(getColor(), 0.05),
          '&:hover': {
            backgroundColor: alpha(getColor(), 0.1),
            transform: 'translateX(5px)',
            transition: 'all 0.3s ease',
          },
        }}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
            <Box sx={{ color: getColor(), mr: 2, mt: 0.5 }}>
              {getIcon()}
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" sx={{ color: getColor(), mb: 1 }}>
                {title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {message}
              </Typography>
              {action && (
                <Box sx={{ mt: 2 }}>
                  {action}
                </Box>
              )}
            </Box>
            {onClose && (
              <IconButton size="small" onClick={onClose} sx={{ color: getColor() }}>
                <Close />
              </IconButton>
            )}
          </Box>
        </CardContent>
      </Card>
    </Slide>
  );
};

// Interactive Progress Component
interface AnimatedProgressProps {
  value: number;
  label: string;
  color?: string;
  animated?: boolean;
}

export const AnimatedProgress: React.FC<AnimatedProgressProps> = ({
  value,
  label,
  color = excelColors.primary.main,
  animated = true,
}) => {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    if (animated) {
      const timer = setTimeout(() => {
        setAnimatedValue(value);
      }, 100);
      return () => clearTimeout(timer);
    } else {
      setAnimatedValue(value);
    }
  }, [value, animated]);

  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 'bold', color }}>
          {animatedValue}%
        </Typography>
      </Box>
      <Box
        sx={{
          height: 8,
          backgroundColor: alpha(color, 0.2),
          borderRadius: 4,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <Box
          sx={{
            height: '100%',
            backgroundColor: color,
            borderRadius: 4,
            width: `${animatedValue}%`,
            transition: animated ? 'width 1.5s cubic-bezier(0.4, 0, 0.2, 1)' : 'none',
            position: 'relative',
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: `linear-gradient(90deg, transparent, ${alpha('#fff', 0.3)}, transparent)`,
              animation: animated ? `${slideInLeft} 2s infinite` : 'none',
            },
          }}
        />
      </Box>
    </Box>
  );
};

export default {
  AnimatedKPICard,
  AnimatedAlert,
  AnimatedProgress,
};