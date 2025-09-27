/**
 * Professional Windows-Style Theme for FraudGuard 360
 * Inspired by Excel 2010 and traditional enterprise software
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';

// Professional color palette inspired by Excel 2010
const colors = {
  // Primary blues (Excel 2010 ribbon style)
  primary: {
    50: '#e8f4fd',
    100: '#c6e2f9',
    200: '#a1cef5',
    300: '#7bb9f0',
    400: '#5ea9ec',
    500: '#4199e8', // Main blue
    600: '#3b8ae1',
    700: '#3278d7',
    800: '#2967ce',
    900: '#1a4dc0',
  },
  
  // Professional grays
  neutral: {
    50: '#fafbfc',
    100: '#f4f6f8',
    200: '#e8ecf0',
    300: '#d6dce4',
    400: '#b8c2ce',
    500: '#9aa6b8', // Medium gray
    600: '#7c8a9e',
    700: '#5e6e84',
    800: '#40526a',
    900: '#223650',
  },
  
  // Status colors
  success: {
    50: '#f0f9f4',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
  },
  
  warning: {
    50: '#fffbeb',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
  },
  
  error: {
    50: '#fef2f2',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
  },
  
  // Professional background colors
  background: {
    default: '#ffffff',
    paper: '#fafbfc',
    ribbon: '#e8ecf0', // Excel-style ribbon background
    toolbar: '#f4f6f8',
    sidebar: '#f8f9fa',
  }
};

const themeOptions: ThemeOptions = {
  palette: {
    mode: 'light',
    primary: {
      main: colors.primary[500],
      light: colors.primary[300],
      dark: colors.primary[700],
      contrastText: '#ffffff',
    },
    secondary: {
      main: colors.neutral[600],
      light: colors.neutral[400],
      dark: colors.neutral[800],
      contrastText: '#ffffff',
    },
    error: {
      main: colors.error[500],
      light: colors.error[50],
      dark: colors.error[700],
    },
    warning: {
      main: colors.warning[500],
      light: colors.warning[50],
      dark: colors.warning[700],
    },
    success: {
      main: colors.success[500],
      light: colors.success[50],
      dark: colors.success[700],
    },
    grey: colors.neutral,
    background: {
      default: colors.background.default,
      paper: colors.background.paper,
    },
    text: {
      primary: colors.neutral[900],
      secondary: colors.neutral[700],
      disabled: colors.neutral[500],
    },
    divider: colors.neutral[200],
  },
  
  typography: {
    fontFamily: '"Segoe UI", "Tahoma", "Geneva", "Verdana", sans-serif',
    fontSize: 14,
    
    h1: {
      fontSize: '2.125rem',
      fontWeight: 600,
      lineHeight: 1.2,
      color: colors.neutral[900],
    },
    h2: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.3,
      color: colors.neutral[900],
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: colors.neutral[900],
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: colors.neutral[900],
    },
    h5: {
      fontSize: '1.125rem',
      fontWeight: 600,
      lineHeight: 1.5,
      color: colors.neutral[900],
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.5,
      color: colors.neutral[900],
    },
    
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      color: colors.neutral[800],
    },
    subtitle2: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.57,
      color: colors.neutral[700],
    },
    
    body1: {
      fontSize: '0.875rem',
      fontWeight: 400,
      lineHeight: 1.5,
      color: colors.neutral[800],
    },
    body2: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.5,
      color: colors.neutral[700],
    },
    
    button: {
      fontSize: '0.875rem',
      fontWeight: 500,
      textTransform: 'none' as const,
      letterSpacing: '0.02em',
    },
    
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.66,
      color: colors.neutral[600],
    },
    
    overline: {
      fontSize: '0.75rem',
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.08em',
      color: colors.neutral[600],
    },
  },
  
  shape: {
    borderRadius: 4, // Professional, not too rounded
  },
  
  components: {
    // Professional button styling
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          textTransform: 'none',
          fontWeight: 500,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: '0 2px 6px rgba(0, 0, 0, 0.15)',
            transform: 'translateY(-1px)',
          },
        },
        contained: {
          backgroundImage: 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%)',
          '&:hover': {
            backgroundImage: 'linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.05) 100%)',
          },
        },
        outlined: {
          borderWidth: '1.5px',
          '&:hover': {
            borderWidth: '1.5px',
            backgroundColor: 'rgba(65, 153, 232, 0.04)',
          },
        },
      },
    },
    
    // Professional card styling
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          border: `1px solid ${colors.neutral[200]}`,
          backgroundImage: 'linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(248,250,252,0.8) 100%)',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.12)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    
    // Professional paper styling
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(250,251,252,0.8) 100%)',
        },
        elevation1: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.12)',
        },
        elevation2: {
          boxShadow: '0 2px 6px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.12)',
        },
        elevation3: {
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.12)',
        },
      },
    },
    
    // Professional table styling
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: colors.background.ribbon,
          borderBottom: `2px solid ${colors.primary[500]}`,
          '& .MuiTableCell-head': {
            fontWeight: 600,
            fontSize: '0.875rem',
            color: colors.neutral[900],
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          },
        },
      },
    },
    
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:nth-of-type(even)': {
            backgroundColor: colors.background.paper,
          },
          '&:hover': {
            backgroundColor: colors.primary[50],
            cursor: 'pointer',
          },
        },
      },
    },
    
    // Professional toolbar styling
    MuiToolbar: {
      styleOverrides: {
        root: {
          backgroundColor: colors.background.toolbar,
          borderBottom: `1px solid ${colors.neutral[200]}`,
          minHeight: '64px !important',
        },
      },
    },
    
    // Professional app bar
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: colors.background.ribbon,
          color: colors.neutral[900],
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          borderBottom: `1px solid ${colors.neutral[200]}`,
        },
      },
    },
    
    // Professional input fields
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          backgroundColor: colors.background.default,
          '& .MuiOutlinedInput-notchedOutline': {
            borderColor: colors.neutral[300],
            borderWidth: '1.5px',
          },
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: colors.primary[400],
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: colors.primary[500],
            borderWidth: '2px',
          },
        },
      },
    },
    
    // Professional tabs
    MuiTabs: {
      styleOverrides: {
        root: {
          backgroundColor: colors.background.ribbon,
          borderBottom: `1px solid ${colors.neutral[200]}`,
          minHeight: '48px',
        },
        indicator: {
          backgroundColor: colors.primary[500],
          height: '3px',
        },
      },
    },
    
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '0.875rem',
          minHeight: '48px',
          color: colors.neutral[700],
          '&.Mui-selected': {
            color: colors.primary[700],
            fontWeight: 600,
          },
        },
      },
    },
  },
};

export const enterpriseTheme = createTheme(themeOptions);

// Custom colors for components
export const customColors = colors;

export default enterpriseTheme;