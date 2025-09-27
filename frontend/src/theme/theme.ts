/**
 * FraudGuard 360 Classic Windows Theme
 * Excel 2010 style professional interface
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';

// Classic Windows 2010 Color Palette (Excel-style)
const colors = {
  primary: {
    50: '#f5f5f5',
    100: '#e8e8e8',
    200: '#d1d1d1',
    300: '#b4b4b4',
    400: '#9e9e9e',
    500: '#4472c4', // Excel blue
    600: '#3c5ba0',
    700: '#34457c',
    800: '#2c3658',
    900: '#242734',
  },
  secondary: {
    50: '#fff8e1',
    100: '#ffecb3',
    200: '#ffe082',
    300: '#ffd54f',
    400: '#ffca28',
    500: '#70ad47', // Excel green
    600: '#609c3c',
    700: '#507a31',
    800: '#405826',
    900: '#30361b',
  },
  success: {
    50: '#e8f5e8',
    100: '#c8e6c9',
    200: '#a5d6a7',
    300: '#81c784',
    400: '#66bb6a',
    500: '#4caf50',
    600: '#43a047',
    700: '#388e3c',
    800: '#2e7d32',
    900: '#1b5e20',
  },
  warning: {
    50: '#fff8e1',
    100: '#ffecb3',
    200: '#ffe082',
    300: '#ffd54f',
    400: '#ffca28',
    500: '#ffc107',
    600: '#ffb300',
    700: '#ffa000',
    800: '#ff8f00',
    900: '#ff6f00',
  },
  error: {
    50: '#ffebee',
    100: '#ffcdd2',
    200: '#ef9a9a',
    300: '#e57373',
    400: '#ef5350',
    500: '#f44336',
    600: '#e53935',
    700: '#d32f2f',
    800: '#c62828',
    900: '#b71c1c',
  },
  grey: {
    50: '#ffffff',
    100: '#f4f4f4',
    200: '#e5e5e5',
    300: '#d6d6d6',
    400: '#c7c7c7',
    500: '#b8b8b8',
    600: '#a9a9a9',
    700: '#7f7f7f',
    800: '#555555',
    900: '#2b2b2b',
  },
  // Classic Windows colors
  office: {
    lightBlue: '#dae8fc',
    mediumBlue: '#4472c4',
    darkBlue: '#2f4f8f',
    lightGreen: '#d5e8d4',
    mediumGreen: '#70ad47',
    lightOrange: '#fff2cc',
    mediumOrange: '#d6b656',
    lightRed: '#f8cecc',
    mediumRed: '#e1665e',
    lightGray: '#f4f4f4',
    mediumGray: '#bfbfbf',
    darkGray: '#7f7f7f',
  },
};

// Enterprise Typography
const typography: ThemeOptions['typography'] = {
  fontFamily: [
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Roboto',
    '"Helvetica Neue"',
    'Arial',
    'sans-serif',
    '"Apple Color Emoji"',
    '"Segoe UI Emoji"',
    '"Segoe UI Symbol"',
  ].join(','),
  h1: {
    fontSize: '2.5rem',
    fontWeight: 600,
    lineHeight: 1.2,
    letterSpacing: '-0.01562em',
  },
  h2: {
    fontSize: '2rem',
    fontWeight: 600,
    lineHeight: 1.3,
    letterSpacing: '-0.00833em',
  },
  h3: {
    fontSize: '1.75rem',
    fontWeight: 600,
    lineHeight: 1.4,
    letterSpacing: '0em',
  },
  h4: {
    fontSize: '1.5rem',
    fontWeight: 600,
    lineHeight: 1.4,
    letterSpacing: '0.00735em',
  },
  h5: {
    fontSize: '1.25rem',
    fontWeight: 600,
    lineHeight: 1.5,
    letterSpacing: '0em',
  },
  h6: {
    fontSize: '1.125rem',
    fontWeight: 600,
    lineHeight: 1.5,
    letterSpacing: '0.0075em',
  },
  subtitle1: {
    fontSize: '1rem',
    fontWeight: 500,
    lineHeight: 1.75,
    letterSpacing: '0.00938em',
  },
  subtitle2: {
    fontSize: '0.875rem',
    fontWeight: 500,
    lineHeight: 1.57,
    letterSpacing: '0.00714em',
  },
  body1: {
    fontSize: '1rem',
    fontWeight: 400,
    lineHeight: 1.5,
    letterSpacing: '0.00938em',
  },
  body2: {
    fontSize: '0.875rem',
    fontWeight: 400,
    lineHeight: 1.43,
    letterSpacing: '0.01071em',
  },
  button: {
    fontSize: '0.875rem',
    fontWeight: 500,
    lineHeight: 1.75,
    letterSpacing: '0.02857em',
    textTransform: 'none' as const,
  },
  caption: {
    fontSize: '0.75rem',
    fontWeight: 400,
    lineHeight: 1.66,
    letterSpacing: '0.03333em',
  },
  overline: {
    fontSize: '0.75rem',
    fontWeight: 400,
    lineHeight: 2.66,
    letterSpacing: '0.08333em',
    textTransform: 'uppercase' as const,
  },
};

// Light Theme
export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: colors.primary[500],
      light: colors.primary[300],
      dark: colors.primary[700],
      contrastText: '#ffffff',
    },
    secondary: {
      main: colors.secondary[500],
      light: colors.secondary[300],
      dark: colors.secondary[700],
      contrastText: '#ffffff',
    },
    error: {
      main: colors.error[500],
      light: colors.error[300],
      dark: colors.error[700],
      contrastText: '#ffffff',
    },
    warning: {
      main: colors.warning[500],
      light: colors.warning[300],
      dark: colors.warning[700],
      contrastText: '#ffffff',
    },
    info: {
      main: colors.primary[500],
      light: colors.primary[300],
      dark: colors.primary[700],
      contrastText: '#ffffff',
    },
    success: {
      main: colors.success[500],
      light: colors.success[300],
      dark: colors.success[700],
      contrastText: '#ffffff',
    },
    grey: colors.grey,
    background: {
      default: '#f4f4f4', // Classic Windows gray
      paper: '#ffffff',
    },
    text: {
      primary: colors.grey[900],
      secondary: colors.grey[700],
      disabled: colors.grey[500],
    },
    divider: colors.grey[300],
  },
  typography,
  shape: {
    borderRadius: 8,
  },
  shadows: [
    'none',
    '0px 2px 1px -1px rgba(0,0,0,0.04), 0px 1px 1px 0px rgba(0,0,0,0.04), 0px 1px 3px 0px rgba(0,0,0,0.08)',
    '0px 3px 1px -2px rgba(0,0,0,0.06), 0px 2px 2px 0px rgba(0,0,0,0.06), 0px 1px 5px 0px rgba(0,0,0,0.12)',
    '0px 3px 3px -2px rgba(0,0,0,0.08), 0px 3px 4px 0px rgba(0,0,0,0.08), 0px 1px 8px 0px rgba(0,0,0,0.16)',
    '0px 2px 4px -1px rgba(0,0,0,0.08), 0px 4px 5px 0px rgba(0,0,0,0.08), 0px 1px 10px 0px rgba(0,0,0,0.16)',
    '0px 3px 5px -1px rgba(0,0,0,0.08), 0px 5px 8px 0px rgba(0,0,0,0.08), 0px 1px 14px 0px rgba(0,0,0,0.16)',
    '0px 3px 5px -1px rgba(0,0,0,0.08), 0px 6px 10px 0px rgba(0,0,0,0.08), 0px 1px 18px 0px rgba(0,0,0,0.16)',
    '0px 4px 5px -2px rgba(0,0,0,0.08), 0px 7px 10px 1px rgba(0,0,0,0.08), 0px 2px 16px 1px rgba(0,0,0,0.16)',
    '0px 5px 5px -3px rgba(0,0,0,0.08), 0px 8px 10px 1px rgba(0,0,0,0.08), 0px 3px 14px 2px rgba(0,0,0,0.16)',
    '0px 5px 6px -3px rgba(0,0,0,0.08), 0px 9px 12px 1px rgba(0,0,0,0.08), 0px 3px 16px 2px rgba(0,0,0,0.16)',
    '0px 6px 6px -3px rgba(0,0,0,0.08), 0px 10px 14px 1px rgba(0,0,0,0.08), 0px 4px 18px 3px rgba(0,0,0,0.16)',
    '0px 6px 7px -4px rgba(0,0,0,0.08), 0px 11px 15px 1px rgba(0,0,0,0.08), 0px 4px 20px 3px rgba(0,0,0,0.16)',
    '0px 7px 8px -4px rgba(0,0,0,0.08), 0px 12px 17px 2px rgba(0,0,0,0.08), 0px 5px 22px 4px rgba(0,0,0,0.16)',
    '0px 7px 8px -4px rgba(0,0,0,0.08), 0px 13px 19px 2px rgba(0,0,0,0.08), 0px 5px 24px 4px rgba(0,0,0,0.16)',
    '0px 7px 9px -4px rgba(0,0,0,0.08), 0px 14px 21px 2px rgba(0,0,0,0.08), 0px 5px 26px 4px rgba(0,0,0,0.16)',
    '0px 8px 9px -5px rgba(0,0,0,0.08), 0px 15px 22px 2px rgba(0,0,0,0.08), 0px 6px 28px 5px rgba(0,0,0,0.16)',
    '0px 8px 10px -5px rgba(0,0,0,0.08), 0px 16px 24px 2px rgba(0,0,0,0.08), 0px 6px 30px 5px rgba(0,0,0,0.16)',
    '0px 8px 11px -5px rgba(0,0,0,0.08), 0px 17px 26px 2px rgba(0,0,0,0.08), 0px 6px 32px 5px rgba(0,0,0,0.16)',
    '0px 9px 11px -5px rgba(0,0,0,0.08), 0px 18px 28px 2px rgba(0,0,0,0.08), 0px 7px 34px 6px rgba(0,0,0,0.16)',
    '0px 9px 12px -6px rgba(0,0,0,0.08), 0px 19px 29px 2px rgba(0,0,0,0.08), 0px 7px 36px 6px rgba(0,0,0,0.16)',
    '0px 10px 13px -6px rgba(0,0,0,0.08), 0px 20px 31px 3px rgba(0,0,0,0.08), 0px 8px 38px 7px rgba(0,0,0,0.16)',
    '0px 10px 13px -6px rgba(0,0,0,0.08), 0px 21px 33px 3px rgba(0,0,0,0.08), 0px 8px 40px 7px rgba(0,0,0,0.16)',
    '0px 10px 14px -6px rgba(0,0,0,0.08), 0px 22px 35px 3px rgba(0,0,0,0.08), 0px 8px 42px 7px rgba(0,0,0,0.16)',
    '0px 11px 14px -7px rgba(0,0,0,0.08), 0px 23px 36px 3px rgba(0,0,0,0.08), 0px 9px 44px 8px rgba(0,0,0,0.16)',
    '0px 11px 15px -7px rgba(0,0,0,0.08), 0px 24px 38px 3px rgba(0,0,0,0.08), 0px 9px 46px 8px rgba(0,0,0,0.16)',
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          fontFamily: typography.fontFamily,
          backgroundColor: '#f4f4f4',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 2,
          textTransform: 'none',
          fontWeight: 400,
          fontSize: '11px',
          minHeight: '23px',
          padding: '2px 8px',
          border: '1px solid #ababab',
          background: 'linear-gradient(to bottom, #ffffff 0%, #e5e5e5 100%)',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.5)',
          color: '#000000',
          '&:hover': {
            background: 'linear-gradient(to bottom, #f0f0f0 0%, #d5d5d5 100%)',
          },
          '&:active': {
            background: 'linear-gradient(to bottom, #d5d5d5 0%, #f0f0f0 100%)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.2)',
          },
        },
        contained: {
          background: 'linear-gradient(to bottom, #4472c4 0%, #3c5ba0 100%)',
          color: '#ffffff',
          border: '1px solid #2f4f8f',
          '&:hover': {
            background: 'linear-gradient(to bottom, #3c5ba0 0%, #34457c 100%)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          boxShadow: 'none',
          border: '1px solid #d4d4d4',
          backgroundColor: '#ffffff',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
        elevation1: {
          boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.05)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: colors.grey[50],
          '& .MuiTableCell-head': {
            fontWeight: 600,
            color: colors.grey[800],
          },
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: colors.grey[50],
          },
        },
      },
    },
  },
});

// Dark Theme
export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: colors.primary[400],
      light: colors.primary[200],
      dark: colors.primary[600],
      contrastText: '#ffffff',
    },
    secondary: {
      main: colors.secondary[400],
      light: colors.secondary[200],
      dark: colors.secondary[600],
      contrastText: '#ffffff',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: colors.grey[400],
      disabled: colors.grey[600],
    },
    divider: colors.grey[700],
  },
  typography,
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: '#1e1e1e',
          border: `1px solid ${colors.grey[800]}`,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: '#1e1e1e',
        },
      },
    },
  },
});

export default lightTheme;