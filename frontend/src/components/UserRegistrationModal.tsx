/**
 * Enhanced User Registration Modal
 * Interactive modal with real form validation and state management
 */

import React, { useState, memo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  Alert,
  CircularProgress,
  Grid,
  IconButton,
  InputAdornment,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Card,
  CardContent,
  Divider,
  Chip,
  styled,
} from '@mui/material';
import {
  Close,
  Visibility,
  VisibilityOff,
  Person,
  Email,
  Phone,
  Work,
  Security,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
} from '@mui/icons-material';
import { useErrorHandler } from '../utils/errorHandling';

const StyledDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiPaper-root': {
    borderRadius: theme.spacing(2),
    maxWidth: '600px',
    width: '100%',
  },
}));

const ValidationMessage = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginTop: theme.spacing(0.5),
  fontSize: '0.875rem',
}));

interface UserRegistrationModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (userData: UserFormData) => Promise<boolean>;
  mode?: 'create' | 'edit';
  existingUser?: Partial<UserFormData>;
}

interface UserFormData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  department: string;
  role: string;
  password: string;
  confirmPassword: string;
}

interface ValidationResult {
  field: string;
  isValid: boolean;
  message: string;
  severity: 'error' | 'warning' | 'success';
}

const UserRegistrationModal: React.FC<UserRegistrationModalProps> = ({
  open,
  onClose,
  onSubmit,
  mode = 'create',
  existingUser = {},
}) => {
  const { handleError } = useErrorHandler();
  
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  const [formData, setFormData] = useState<UserFormData>({
    firstName: existingUser.firstName || '',
    lastName: existingUser.lastName || '',
    email: existingUser.email || '',
    phone: existingUser.phone || '',
    department: existingUser.department || '',
    role: existingUser.role || '',
    password: '',
    confirmPassword: '',
  });

  const [validationResults, setValidationResults] = useState<ValidationResult[]>([]);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Real-time validation
  const validateField = (field: keyof UserFormData, value: string): ValidationResult => {
    switch (field) {
      case 'firstName':
      case 'lastName':
        if (!value.trim()) {
          return {
            field,
            isValid: false,
            message: `${field === 'firstName' ? 'First' : 'Last'} name is required`,
            severity: 'error'
          };
        }
        if (value.length < 2) {
          return {
            field,
            isValid: false,
            message: 'Name must be at least 2 characters',
            severity: 'error'
          };
        }
        return {
          field,
          isValid: true,
          message: 'Valid name',
          severity: 'success'
        };

      case 'email':
        if (!value.trim()) {
          return {
            field,
            isValid: false,
            message: 'Email is required',
            severity: 'error'
          };
        }
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
          return {
            field,
            isValid: false,
            message: 'Please enter a valid email address',
            severity: 'error'
          };
        }
        return {
          field,
          isValid: true,
          message: 'Valid email address',
          severity: 'success'
        };

      case 'phone':
        if (!value.trim()) {
          return {
            field,
            isValid: false,
            message: 'Phone number is required',
            severity: 'error'
          };
        }
        const phoneRegex = /^\+?[\d\s\-\(\)]{10,}$/;
        if (!phoneRegex.test(value)) {
          return {
            field,
            isValid: false,
            message: 'Please enter a valid phone number',
            severity: 'error'
          };
        }
        return {
          field,
          isValid: true,
          message: 'Valid phone number',
          severity: 'success'
        };

      case 'password':
        if (mode === 'edit' && !value) {
          return {
            field,
            isValid: true,
            message: 'Leave blank to keep current password',
            severity: 'warning'
          };
        }
        if (!value) {
          return {
            field,
            isValid: false,
            message: 'Password is required',
            severity: 'error'
          };
        }
        if (value.length < 8) {
          return {
            field,
            isValid: false,
            message: 'Password must be at least 8 characters',
            severity: 'error'
          };
        }
        if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(value)) {
          return {
            field,
            isValid: false,
            message: 'Password must contain uppercase, lowercase, and number',
            severity: 'warning'
          };
        }
        return {
          field,
          isValid: true,
          message: 'Strong password',
          severity: 'success'
        };

      case 'confirmPassword':
        if (mode === 'edit' && !formData.password) {
          return {
            field,
            isValid: true,
            message: '',
            severity: 'success'
          };
        }
        if (value !== formData.password) {
          return {
            field,
            isValid: false,
            message: 'Passwords do not match',
            severity: 'error'
          };
        }
        return {
          field,
          isValid: true,
          message: 'Passwords match',
          severity: 'success'
        };

      default:
        return {
          field,
          isValid: true,
          message: '',
          severity: 'success'
        };
    }
  };

  const handleFieldChange = (field: keyof UserFormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    // Real-time validation
    const validation = validateField(field, value);
    setValidationResults(prev => {
      const filtered = prev.filter(v => v.field !== field);
      return [...filtered, validation];
    });

    // Re-validate confirm password if password changed
    if (field === 'password' && formData.confirmPassword) {
      const confirmValidation = validateField('confirmPassword', formData.confirmPassword);
      setValidationResults(prev => {
        const filtered = prev.filter(v => v.field !== 'confirmPassword');
        return [...filtered, confirmValidation];
      });
    }
  };

  const getFieldValidation = (field: string) => {
    return validationResults.find(v => v.field === field);
  };

  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 0: // Personal Info
        const personalFields = ['firstName', 'lastName', 'email', 'phone'];
        return personalFields.every(field => {
          const validation = getFieldValidation(field);
          return validation?.isValid && formData[field as keyof UserFormData].trim();
        });
      
      case 1: // Professional Info
        return Boolean(formData.department && formData.role);
      
      case 2: // Security
        if (mode === 'edit') {
          // For edit mode, password is optional
          if (!formData.password) return true;
        }
        const passwordValidation = getFieldValidation('password');
        const confirmValidation = getFieldValidation('confirmPassword');
        return Boolean(passwordValidation?.isValid && confirmValidation?.isValid);
      
      default:
        return false;
    }
  };

  const handleNext = () => {
    if (currentStep < 2) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true);
      setSubmitError(null);

      // Final validation
      const allFields = Object.keys(formData) as (keyof UserFormData)[];
      const finalValidations = allFields.map(field => 
        validateField(field, formData[field])
      );

      const hasErrors = finalValidations.some(v => !v.isValid);
      if (hasErrors) {
        setValidationResults(finalValidations);
        setSubmitError('Please fix validation errors before submitting');
        return;
      }

      // Submit the form
      const success = await onSubmit(formData);
      
      if (success) {
        onClose();
        // Reset form
        setFormData({
          firstName: '',
          lastName: '',
          email: '',
          phone: '',
          department: '',
          role: '',
          password: '',
          confirmPassword: '',
        });
        setCurrentStep(0);
        setValidationResults([]);
      } else {
        setSubmitError('Failed to save user. Please try again.');
      }
    } catch (error) {
      handleError(
        error instanceof Error ? error : new Error('Unknown error'),
        'API' as any,
        'MEDIUM' as any,
        'User Registration',
        'Failed to create user account'
      );
      setSubmitError('An unexpected error occurred. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderValidationIcon = (validation?: ValidationResult) => {
    if (!validation || !validation.message) return null;
    
    switch (validation.severity) {
      case 'success':
        return <CheckCircle color="success" fontSize="small" />;
      case 'warning':
        return <Warning color="warning" fontSize="small" />;
      case 'error':
        return <ErrorIcon color="error" fontSize="small" />;
      default:
        return null;
    }
  };

  const steps = [
    'Personal Information',
    'Professional Details',
    'Security Settings'
  ];

  return (
    <StyledDialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">
            {mode === 'create' ? 'Create New User' : 'Edit User Profile'}
          </Typography>
          <IconButton onClick={onClose} size="small">
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        <Stepper activeStep={currentStep} orientation="vertical">
          {/* Step 1: Personal Information */}
          <Step>
            <StepLabel>Personal Information</StepLabel>
            <StepContent>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="First Name"
                    value={formData.firstName}
                    onChange={(e) => handleFieldChange('firstName', e.target.value)}
                    error={getFieldValidation('firstName')?.severity === 'error'}
                    helperText={getFieldValidation('firstName')?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Person />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          {renderValidationIcon(getFieldValidation('firstName'))}
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Last Name"
                    value={formData.lastName}
                    onChange={(e) => handleFieldChange('lastName', e.target.value)}
                    error={getFieldValidation('lastName')?.severity === 'error'}
                    helperText={getFieldValidation('lastName')?.message}
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          {renderValidationIcon(getFieldValidation('lastName'))}
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Email Address"
                    type="email"
                    value={formData.email}
                    onChange={(e) => handleFieldChange('email', e.target.value)}
                    error={getFieldValidation('email')?.severity === 'error'}
                    helperText={getFieldValidation('email')?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Email />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          {renderValidationIcon(getFieldValidation('email'))}
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Phone Number"
                    value={formData.phone}
                    onChange={(e) => handleFieldChange('phone', e.target.value)}
                    error={getFieldValidation('phone')?.severity === 'error'}
                    helperText={getFieldValidation('phone')?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Phone />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          {renderValidationIcon(getFieldValidation('phone'))}
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
              </Grid>
            </StepContent>
          </Step>

          {/* Step 2: Professional Details */}
          <Step>
            <StepLabel>Professional Details</StepLabel>
            <StepContent>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Department</InputLabel>
                    <Select
                      value={formData.department}
                      onChange={(e) => handleFieldChange('department', e.target.value)}
                      startAdornment={
                        <InputAdornment position="start">
                          <Work />
                        </InputAdornment>
                      }
                    >
                      <MenuItem value="fraud-detection">Fraud Detection</MenuItem>
                      <MenuItem value="analytics">Analytics</MenuItem>
                      <MenuItem value="compliance">Compliance</MenuItem>
                      <MenuItem value="it-security">IT Security</MenuItem>
                      <MenuItem value="operations">Operations</MenuItem>
                      <MenuItem value="management">Management</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Role</InputLabel>
                    <Select
                      value={formData.role}
                      onChange={(e) => handleFieldChange('role', e.target.value)}
                    >
                      <MenuItem value="analyst">Analyst</MenuItem>
                      <MenuItem value="senior-analyst">Senior Analyst</MenuItem>
                      <MenuItem value="supervisor">Supervisor</MenuItem>
                      <MenuItem value="manager">Manager</MenuItem>
                      <MenuItem value="admin">Administrator</MenuItem>
                      <MenuItem value="viewer">Viewer</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </StepContent>
          </Step>

          {/* Step 3: Security Settings */}
          <Step>
            <StepLabel>Security Settings</StepLabel>
            <StepContent>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label={mode === 'edit' ? 'New Password (leave blank to keep current)' : 'Password'}
                    type={showPassword ? 'text' : 'password'}
                    value={formData.password}
                    onChange={(e) => handleFieldChange('password', e.target.value)}
                    error={getFieldValidation('password')?.severity === 'error'}
                    helperText={getFieldValidation('password')?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Security />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPassword(!showPassword)}
                            edge="end"
                          >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                          {renderValidationIcon(getFieldValidation('password'))}
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Confirm Password"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={formData.confirmPassword}
                    onChange={(e) => handleFieldChange('confirmPassword', e.target.value)}
                    error={getFieldValidation('confirmPassword')?.severity === 'error'}
                    helperText={getFieldValidation('confirmPassword')?.message}
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                            edge="end"
                          >
                            {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                          {renderValidationIcon(getFieldValidation('confirmPassword'))}
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
              </Grid>
            </StepContent>
          </Step>
        </Stepper>

        {submitError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {submitError}
          </Alert>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 3 }}>
        <Button onClick={onClose} disabled={isSubmitting}>
          Cancel
        </Button>
        
        {currentStep > 0 && (
          <Button onClick={handleBack} disabled={isSubmitting}>
            Back
          </Button>
        )}
        
        {currentStep < 2 ? (
          <Button
            onClick={handleNext}
            variant="contained"
            disabled={!isStepValid(currentStep)}
          >
            Next
          </Button>
        ) : (
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={!isStepValid(currentStep) || isSubmitting}
            startIcon={isSubmitting ? <CircularProgress size={20} /> : null}
          >
            {isSubmitting 
              ? `${mode === 'create' ? 'Creating' : 'Updating'}...` 
              : `${mode === 'create' ? 'Create User' : 'Update User'}`
            }
          </Button>
        )}
      </DialogActions>
    </StyledDialog>
  );
};

export default memo(UserRegistrationModal);