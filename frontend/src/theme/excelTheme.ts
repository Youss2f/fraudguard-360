/**
 * Microsoft Excel Inspired Theme
 * Professional color palette and styling matching Excel's modern interface
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';

// Microsoft Excel Color Palette
export const excelColors = {
  // Primary Excel colors
  primary: {
    main: '#217346',        // Excel Green (primary brand)
    light: '#70AD47',       // Light Excel Green
    dark: '#0E5325',        // Dark Excel Green
    contrastText: '#FFFFFF',
  },
  
  // Secondary colors from Excel ribbon
  secondary: {
    main: '#2F5597',        // Excel Blue
    light: '#5B9BD5',       // Light Excel Blue
    dark: '#1F3864',        // Dark Excel Blue
    contrastText: '#FFFFFF',
  },
  
  // Background colors
  background: {
    default: '#FAFAFA',     // Excel's main background
    paper: '#FFFFFF',       // Excel worksheet/card background
    ribbon: '#F3F2F1',      // Excel ribbon background
    hover: '#F3F2F1',       // Hover state
    selected: '#CCE7F0',    // Selected cell background
    border: '#D1D1D1',      // Excel grid lines
  },
  
  // Text colors
  text: {
    primary: '#323130',     // Excel's primary text
    secondary: '#605E5C',   // Excel's secondary text
    disabled: '#A19F9D',    // Disabled text
    hint: '#8A8886',        // Hint text
  },
  
  // Excel accent colors (from theme colors)
  accent: {
    blue: '#5B9BD5',        // Accent 1
    orange: '#ED7D31',      // Accent 2
    gray: '#A5A5A5',        // Accent 3
    yellow: '#FFC000',      // Accent 4
    lightBlue: '#4472C4',   // Accent 5
    green: '#70AD47',       // Accent 6
  },
  
  // Status and feedback colors
  success: {
    main: '#107C10',        // Excel success green
    light: '#92C353',
    dark: '#0B5A0B',
    contrastText: '#FFFFFF',
  },
  
  warning: {
    main: '#FF8C00',        // Excel warning orange
    light: '#FFAB40',
    dark: '#E65100',
    contrastText: '#FFFFFF',
  },
  
  error: {
    main: '#D13438',        // Excel error red
    light: '#EF5350',
    dark: '#C62828',
    contrastText: '#FFFFFF',
  },
  
  info: {
    main: '#2F5597',        // Excel info blue
    light: '#5B9BD5',
    dark: '#1F3864',
    contrastText: '#FFFFFF',
  },
  
  // Grid and table colors
  grid: {
    header: '#E1DFDD',      // Excel table header
    alternateRow: '#F8F8F8', // Alternate row background
    border: '#D1D1D1',      // Grid borders
    selected: '#CCE7F0',    // Selected row/cell
    hover: '#F3F2F1',       // Hover row
  },
};

// Typography matching Excel's modern interface
export const excelTypography = {
  fontFamily: '"Segoe UI", "Calibri", -apple-system, BlinkMacSystemFont, sans-serif',
  fontSize: 14,
  fontWeightLight: 300,
  fontWeightRegular: 400,
  fontWeightMedium: 500,
  fontWeightBold: 600,
  
  h1: {
    fontSize: '2.125rem',   // 34px
    fontWeight: 600,
    lineHeight: 1.2,
    color: excelColors.text.primary,
  },
  h2: {
    fontSize: '1.875rem',   // 30px
    fontWeight: 600,
    lineHeight: 1.3,
    color: excelColors.text.primary,
  },
  h3: {
    fontSize: '1.5rem',     // 24px
    fontWeight: 500,
    lineHeight: 1.4,
    color: excelColors.text.primary,
  },
  h4: {
    fontSize: '1.25rem',    // 20px
    fontWeight: 500,
    lineHeight: 1.4,
    color: excelColors.text.primary,
  },
  h5: {
    fontSize: '1.125rem',   // 18px
    fontWeight: 500,
    lineHeight: 1.5,
    color: excelColors.text.primary,
  },
  h6: {
    fontSize: '1rem',       // 16px
    fontWeight: 500,
    lineHeight: 1.5,
    color: excelColors.text.primary,
  },
  subtitle1: {
    fontSize: '1rem',
    fontWeight: 400,
    lineHeight: 1.75,
    color: excelColors.text.secondary,
  },
  subtitle2: {
    fontSize: '0.875rem',
    fontWeight: 500,
    lineHeight: 1.57,
    color: excelColors.text.secondary,
  },
  body1: {
    fontSize: '0.875rem',   // 14px - Excel's standard text size
    fontWeight: 400,
    lineHeight: 1.5,
    color: excelColors.text.primary,
  },
  body2: {
    fontSize: '0.75rem',    // 12px
    fontWeight: 400,
    lineHeight: 1.5,
    color: excelColors.text.secondary,
  },
  button: {
    fontSize: '0.875rem',
    fontWeight: 500,
    lineHeight: 1.5,
    textTransform: 'none' as const,
  },
  caption: {
    fontSize: '0.75rem',
    fontWeight: 400,
    lineHeight: 1.66,
    color: excelColors.text.hint,
  },
  overline: {
    fontSize: '0.75rem',
    fontWeight: 500,
    lineHeight: 2.66,
    textTransform: 'uppercase' as const,
    color: excelColors.text.hint,
  },
};

// Excel-inspired component overrides
export const excelTheme = createTheme({
  palette: {
    mode: 'light',
    primary: excelColors.primary,
    secondary: excelColors.secondary,
    background: {
      default: excelColors.background.default,
      paper: excelColors.background.paper,
    },
    text: excelColors.text,
    success: excelColors.success,
    warning: excelColors.warning,
    error: excelColors.error,
    info: excelColors.info,
    divider: excelColors.background.border,
  },
  
  // Responsive breakpoints
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920,
    },
  },
  
  typography: {
    ...excelTypography,
    // Responsive typography
    h1: {
      fontSize: '2.125rem',
      fontWeight: 600,
      lineHeight: 1.2,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '1.75rem',
      },
    },
    h2: {
      fontSize: '1.875rem',
      fontWeight: 600,
      lineHeight: 1.3,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '1.5rem',
      },
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 500,
      lineHeight: 1.4,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '1.25rem',
      },
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 500,
      lineHeight: 1.4,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '1.125rem',
      },
    },
    h5: {
      fontSize: '1.125rem',
      fontWeight: 500,
      lineHeight: 1.5,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '1rem',
      },
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '0.875rem',
      },
    },
    body1: {
      fontSize: '0.875rem',
      fontWeight: 400,
      lineHeight: 1.5,
      color: excelColors.text.primary,
      '@media (max-width:600px)': {
        fontSize: '0.8rem',
      },
    },
    body2: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.5,
      color: excelColors.text.secondary,
      '@media (max-width:600px)': {
        fontSize: '0.7rem',
      },
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.5,
      textTransform: 'none' as const,
      '@media (max-width:600px)': {
        fontSize: '0.8rem',
      },
    },
  },
  
  spacing: 8, // Excel uses 8px base spacing
  
  shape: {
    borderRadius: 4, // Excel's subtle rounded corners
  },
  
  components: {
    // AppBar (Excel Ribbon style)
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: excelColors.background.ribbon,
          color: excelColors.text.primary,
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
          borderBottom: `1px solid ${excelColors.background.border}`,
        },
      },
    },
    
    // Paper (Excel panels/cards)
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: excelColors.background.paper,
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          border: `1px solid ${excelColors.background.border}`,
        },
        elevation1: {
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
        },
        elevation2: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        },
      },
    },
    
    // Buttons (Excel button style)
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          borderRadius: 4,
          padding: '6px 16px',
          fontSize: '0.875rem',
        },
        contained: {
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
          '&:hover': {
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          },
        },
        outlined: {
          borderColor: excelColors.background.border,
          '&:hover': {
            backgroundColor: excelColors.background.hover,
            borderColor: excelColors.primary.main,
          },
        },
      },
    },
    
    // Cards (Excel worksheet sections)
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: excelColors.background.paper,
          border: `1px solid ${excelColors.background.border}`,
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          borderRadius: 8,
        },
      },
    },
    
    // Tables (Excel grid style)
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: excelColors.grid.header,
          '& .MuiTableCell-head': {
            backgroundColor: excelColors.grid.header,
            color: excelColors.text.primary,
            fontWeight: 600,
            fontSize: '0.875rem',
            borderBottom: `2px solid ${excelColors.background.border}`,
          },
        },
      },
    },
    
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:nth-of-type(even)': {
            backgroundColor: excelColors.grid.alternateRow,
          },
          '&:hover': {
            backgroundColor: excelColors.grid.hover,
          },
        },
      },
    },
    
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: `1px solid ${excelColors.grid.border}`,
          padding: '8px 16px',
          fontSize: '0.875rem',
        },
      },
    },
    
    // Form controls (Excel input style)
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: excelColors.background.paper,
            '& fieldset': {
              borderColor: excelColors.background.border,
            },
            '&:hover fieldset': {
              borderColor: excelColors.primary.main,
            },
            '&.Mui-focused fieldset': {
              borderColor: excelColors.primary.main,
              borderWidth: 2,
            },
          },
        },
      },
    },
    
    // Drawer (Excel navigation pane)
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: excelColors.background.ribbon,
          borderRight: `1px solid ${excelColors.background.border}`,
        },
      },
    },
    
    // List items (Excel navigation)
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          margin: '2px 8px',
          '&:hover': {
            backgroundColor: excelColors.background.hover,
          },
          '&.Mui-selected': {
            backgroundColor: excelColors.background.selected,
            '&:hover': {
              backgroundColor: excelColors.background.selected,
            },
          },
        },
      },
    },
    
    // Tabs (Excel ribbon tabs)
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontSize: '0.875rem',
          fontWeight: 500,
          minHeight: 48,
          '&.Mui-selected': {
            color: excelColors.primary.main,
          },
        },
      },
    },
    
    // Chips (Excel tags/labels)
    MuiChip: {
      styleOverrides: {
        root: {
          fontSize: '0.75rem',
          height: 24,
          '&.MuiChip-filled': {
            backgroundColor: excelColors.accent.gray,
            color: excelColors.text.primary,
          },
        },
      },
    },
  },
});

export default excelTheme;