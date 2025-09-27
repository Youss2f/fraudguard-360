/**
 * Professional Technical Software Theme
 * "The Expert's Cockpit" - High-end technical software aesthetic
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';

// Professional Technical Color Palette
const technicalColors = {
  // Monochromatic Gray Scale
  background: {
    primary: '#2C3E50',      // Dark gray - Left navigation
    secondary: '#34495E',    // Slightly lighter - Top bar
    content: '#ECF0F1',      // Very light gray - Main content
    paper: '#FFFFFF',        // Pure white - Cards/panels
    hover: '#BDC3C7',        // Light gray - Hover states
    border: '#D5DBDB',       // Subtle borders
  },
  
  // Functional Accent Color
  accent: {
    primary: '#3498DB',      // Professional blue
    hover: '#2980B9',        // Darker blue for hover
    light: '#AED6F1',        // Light blue for backgrounds
    dark: '#1B4F72',         // Dark blue for text
  },
  
  // High-Contrast Text
  text: {
    primary: '#2C3E50',      // Dark gray on light backgrounds
    secondary: '#5D6D7E',    // Medium gray for secondary text
    inverse: '#FFFFFF',      // White on dark backgrounds
    muted: '#85929E',        // Muted gray for labels
  },
  
  // Status Colors (Functional only)
  status: {
    success: '#27AE60',      // Functional green
    warning: '#F39C12',      // Functional orange
    error: '#E74C3C',        // Functional red
    info: '#3498DB',         // Same as accent
  },
  
  // Grid and Data Display
  grid: {
    header: '#F8F9FA',       // Light gray for table headers
    row: '#FDFDFE',          // Subtle row color
    rowHover: '#F1F2F6',     // Row hover state
    border: '#DEE2E6',       // Grid borders
  }
};

// Professional Typography Scale
const technicalTypography: ThemeOptions['typography'] = {
  fontFamily: [
    'Inter',
    'Roboto',
    'Lato',
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Arial',
    'sans-serif'
  ].join(','),
  
  // Strict Typographic Hierarchy
  h1: {
    fontSize: '24px',
    fontWeight: 700,
    color: technicalColors.text.primary,
    letterSpacing: '-0.5px',
  },
  h2: {
    fontSize: '18px',
    fontWeight: 700,
    color: technicalColors.text.primary,
    letterSpacing: '-0.25px',
  },
  h3: {
    fontSize: '16px',
    fontWeight: 600,
    color: technicalColors.text.primary,
  },
  h4: {
    fontSize: '14px',
    fontWeight: 600,
    color: technicalColors.text.primary,
  },
  h5: {
    fontSize: '12px',
    fontWeight: 600,
    color: technicalColors.text.primary,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  h6: {
    fontSize: '11px',
    fontWeight: 600,
    color: technicalColors.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: '0.75px',
  },
  body1: {
    fontSize: '14px',
    fontWeight: 400,
    color: technicalColors.text.primary,
    lineHeight: 1.4,
  },
  body2: {
    fontSize: '12px',
    fontWeight: 400,
    color: technicalColors.text.secondary,
    lineHeight: 1.3,
  },
  caption: {
    fontSize: '11px',
    fontWeight: 400,
    color: technicalColors.text.muted,
    lineHeight: 1.2,
  },
  button: {
    fontSize: '13px',
    fontWeight: 500,
    textTransform: 'none',
    letterSpacing: '0.25px',
  },
  overline: {
    fontSize: '10px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '1px',
    color: technicalColors.text.muted,
  }
};

// Professional Component Overrides
const technicalComponents: ThemeOptions['components'] = {
  // Global defaults
  MuiCssBaseline: {
    styleOverrides: {
      body: {
        backgroundColor: technicalColors.background.content,
        fontFamily: technicalTypography.fontFamily,
      },
    },
  },
  
  // Button overrides - Sharp, functional
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: '3px',
        textTransform: 'none',
        fontWeight: 500,
        padding: '8px 16px',
        minHeight: '32px',
        border: 'none',
        boxShadow: 'none',
        '&:hover': {
          boxShadow: 'none',
        },
      },
      contained: {
        backgroundColor: technicalColors.accent.primary,
        color: technicalColors.text.inverse,
        '&:hover': {
          backgroundColor: technicalColors.accent.hover,
        },
      },
      outlined: {
        borderColor: technicalColors.background.border,
        color: technicalColors.text.primary,
        backgroundColor: 'transparent',
        '&:hover': {
          backgroundColor: technicalColors.background.hover,
          borderColor: technicalColors.accent.primary,
        },
      },
      text: {
        color: technicalColors.text.secondary,
        '&:hover': {
          backgroundColor: technicalColors.background.hover,
        },
      },
    },
  },
  
  // Card/Paper overrides - Clean, bordered
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundColor: technicalColors.background.paper,
        border: `1px solid ${technicalColors.background.border}`,
        borderRadius: '3px',
        boxShadow: 'none',
      },
    },
  },
  
  MuiCard: {
    styleOverrides: {
      root: {
        backgroundColor: technicalColors.background.paper,
        border: `1px solid ${technicalColors.background.border}`,
        borderRadius: '3px',
        boxShadow: 'none',
      },
    },
  },
  
  // Input field overrides - Clean, functional
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: '3px',
          backgroundColor: technicalColors.background.paper,
          '& fieldset': {
            borderColor: technicalColors.background.border,
          },
          '&:hover fieldset': {
            borderColor: technicalColors.accent.primary,
          },
          '&.Mui-focused fieldset': {
            borderColor: technicalColors.accent.primary,
            borderWidth: '2px',
            boxShadow: `0 0 0 1px ${technicalColors.accent.light}`,
          },
        },
      },
    },
  },
  
  // Table overrides - High density, scannable
  MuiTable: {
    styleOverrides: {
      root: {
        borderCollapse: 'separate',
        borderSpacing: 0,
      },
    },
  },
  
  MuiTableHead: {
    styleOverrides: {
      root: {
        backgroundColor: technicalColors.grid.header,
      },
    },
  },
  
  MuiTableCell: {
    styleOverrides: {
      root: {
        borderBottom: `1px solid ${technicalColors.grid.border}`,
        padding: '8px 12px',
        fontSize: '13px',
      },
      head: {
        backgroundColor: technicalColors.grid.header,
        fontWeight: 600,
        color: technicalColors.text.primary,
        fontSize: '12px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      },
    },
  },
  
  MuiTableRow: {
    styleOverrides: {
      root: {
        '&:hover': {
          backgroundColor: technicalColors.grid.rowHover,
        },
        '&:nth-of-type(even)': {
          backgroundColor: technicalColors.grid.row,
        },
      },
    },
  },
  
  // Tab overrides - Clean, minimal
  MuiTabs: {
    styleOverrides: {
      root: {
        backgroundColor: technicalColors.background.secondary,
        minHeight: '40px',
      },
      indicator: {
        backgroundColor: technicalColors.accent.primary,
        height: '3px',
      },
    },
  },
  
  MuiTab: {
    styleOverrides: {
      root: {
        textTransform: 'none',
        fontWeight: 500,
        fontSize: '13px',
        color: technicalColors.text.inverse,
        minHeight: '40px',
        '&.Mui-selected': {
          color: technicalColors.text.inverse,
        },
      },
    },
  },
  
  // Chip overrides - Functional, minimal
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: '3px',
        height: '22px',
        fontSize: '11px',
        fontWeight: 500,
      },
      filled: {
        backgroundColor: technicalColors.background.hover,
        color: technicalColors.text.primary,
      },
      outlined: {
        borderColor: technicalColors.background.border,
        color: technicalColors.text.primary,
      },
    },
  },
  
  // Icon button overrides
  MuiIconButton: {
    styleOverrides: {
      root: {
        padding: '6px',
        borderRadius: '3px',
        color: technicalColors.text.secondary,
        '&:hover': {
          backgroundColor: technicalColors.background.hover,
          color: technicalColors.text.primary,
        },
      },
    },
  },
  
  // Dialog overrides
  MuiDialog: {
    styleOverrides: {
      paper: {
        borderRadius: '3px',
        boxShadow: '0 8px 32px rgba(44, 62, 80, 0.15)',
      },
    },
  },
  
  MuiDialogTitle: {
    styleOverrides: {
      root: {
        backgroundColor: technicalColors.background.secondary,
        color: technicalColors.text.inverse,
        padding: '12px 16px',
        fontSize: '16px',
        fontWeight: 600,
      },
    },
  },
};

// Create the professional technical theme
export const technicalTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: technicalColors.accent.primary,
      dark: technicalColors.accent.hover,
      light: technicalColors.accent.light,
    },
    secondary: {
      main: technicalColors.background.secondary,
    },
    background: {
      default: technicalColors.background.content,
      paper: technicalColors.background.paper,
    },
    text: {
      primary: technicalColors.text.primary,
      secondary: technicalColors.text.secondary,
    },
    success: {
      main: technicalColors.status.success,
    },
    warning: {
      main: technicalColors.status.warning,
    },
    error: {
      main: technicalColors.status.error,
    },
    info: {
      main: technicalColors.status.info,
    },
  },
  typography: technicalTypography,
  components: technicalComponents,
  shape: {
    borderRadius: 3, // Minimal rounding
  },
  spacing: 8, // Consistent spacing unit
});

// Export color system for direct usage
export const technicalColorSystem = technicalColors;

// Export component-specific styling functions
export const getTechnicalStyling = {
  quickAccessBar: {
    backgroundColor: technicalColors.background.secondary,
    borderBottom: `1px solid ${technicalColors.background.border}`,
    height: '40px',
    padding: '0 16px',
  },
  
  navigationPanel: {
    backgroundColor: technicalColors.background.primary,
    width: '240px',
    collapsedWidth: '60px',
    borderRight: `1px solid ${technicalColors.background.border}`,
  },
  
  contentArea: {
    backgroundColor: technicalColors.background.content,
    padding: '16px',
    marginLeft: '240px', // Account for nav panel
    marginTop: '40px',   // Account for top bar
    minHeight: 'calc(100vh - 40px)',
  },
  
  dataCard: {
    backgroundColor: technicalColors.background.paper,
    border: `1px solid ${technicalColors.background.border}`,
    borderRadius: '3px',
    padding: '16px',
    marginBottom: '16px',
  },
  
  statusIndicator: (status: 'success' | 'warning' | 'error' | 'info') => ({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: technicalColors.status[status],
    display: 'inline-block',
    marginRight: '8px',
  }),
};

export default technicalTheme;