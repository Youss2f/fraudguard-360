/**
 * Enhanced Error Handling Utilities
 * Comprehensive error management system for FraudGuard 360
 */

// Error types for better error categorization
export enum ErrorType {
  NETWORK = 'NETWORK',
  AUTHENTICATION = 'AUTHENTICATION',
  AUTHORIZATION = 'AUTHORIZATION',
  VALIDATION = 'VALIDATION',
  API = 'API',
  UI = 'UI',
  UNKNOWN = 'UNKNOWN'
}

export enum ErrorSeverity {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

// Enhanced error class
export class AppError extends Error {
  public readonly type: ErrorType;
  public readonly severity: ErrorSeverity;
  public readonly timestamp: Date;
  public readonly context?: string;
  public readonly userMessage?: string;
  public readonly originalError?: Error;

  constructor(
    message: string,
    type: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context?: string,
    userMessage?: string,
    originalError?: Error
  ) {
    super(message);
    this.name = 'AppError';
    this.type = type;
    this.severity = severity;
    this.timestamp = new Date();
    this.context = context;
    this.userMessage = userMessage;
    this.originalError = originalError;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, AppError);
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      type: this.type,
      severity: this.severity,
      timestamp: this.timestamp.toISOString(),
      context: this.context,
      userMessage: this.userMessage,
      stack: this.stack,
      originalError: this.originalError?.message
    };
  }
}

// Error reporting service
class ErrorReportingService {
  private static instance: ErrorReportingService;
  private errorQueue: AppError[] = [];
  private maxQueueSize = 100;

  static getInstance(): ErrorReportingService {
    if (!ErrorReportingService.instance) {
      ErrorReportingService.instance = new ErrorReportingService();
    }
    return ErrorReportingService.instance;
  }

  report(error: AppError): void {
    console.error('AppError reported:', error.toJSON());

    // Add to queue
    this.errorQueue.push(error);
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue.shift(); // Remove oldest error
    }

    // In production, send to error reporting service
    if (process.env.NODE_ENV === 'production') {
      this.sendToErrorService(error);
    }

    // Show user notification for critical errors
    if (error.severity === ErrorSeverity.CRITICAL) {
      this.notifyUser(error);
    }
  }

  private async sendToErrorService(error: AppError): Promise<void> {
    try {
      // Example implementation - replace with actual error service
      await fetch('/api/errors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(error.toJSON()),
      });
    } catch (e) {
      console.error('Failed to send error to service:', e);
    }
  }

  private notifyUser(error: AppError): void {
    const message = error.userMessage || 'A critical error occurred. Please contact support.';
    
    // Create a custom notification or use the existing notification system
    if ((window as any).showNotification) {
      (window as any).showNotification({
        type: 'error',
        title: 'Critical Error',
        message,
        duration: 10000,
      });
    } else {
      // Fallback to console for now
      console.error('Critical error notification:', message);
    }
  }

  getRecentErrors(count = 10): AppError[] {
    return this.errorQueue.slice(-count);
  }

  clearErrors(): void {
    this.errorQueue = [];
  }
}

// Async error handler for promises
export const handleAsyncError = async <T>(
  asyncOperation: () => Promise<T>,
  context?: string,
  userMessage?: string
): Promise<T | null> => {
  try {
    return await asyncOperation();
  } catch (error) {
    const appError = new AppError(
      `Async operation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      ErrorType.API,
      ErrorSeverity.MEDIUM,
      context,
      userMessage,
      error instanceof Error ? error : undefined
    );
    
    ErrorReportingService.getInstance().report(appError);
    return null;
  }
};

// Sync error handler
export const handleSyncError = <T>(
  operation: () => T,
  context?: string,
  userMessage?: string,
  fallbackValue?: T
): T | null => {
  try {
    return operation();
  } catch (error) {
    const appError = new AppError(
      `Sync operation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      ErrorType.UI,
      ErrorSeverity.LOW,
      context,
      userMessage,
      error instanceof Error ? error : undefined
    );
    
    ErrorReportingService.getInstance().report(appError);
    return fallbackValue ?? null;
  }
};

// Network error handler with retry logic
export const handleNetworkRequest = async <T>(
  requestFn: () => Promise<T>,
  retries = 3,
  context?: string
): Promise<T | null> => {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await requestFn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown network error');
      
      if (attempt === retries) {
        const appError = new AppError(
          `Network request failed after ${retries} attempts: ${lastError.message}`,
          ErrorType.NETWORK,
          ErrorSeverity.HIGH,
          context,
          'Unable to connect to the server. Please check your internet connection.',
          lastError
        );
        
        ErrorReportingService.getInstance().report(appError);
        return null;
      }

      // Wait before retry (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
    }
  }

  return null;
};

// Validation error helper
export const createValidationError = (
  field: string,
  value: any,
  rule: string,
  userMessage?: string
): AppError => {
  return new AppError(
    `Validation failed for field "${field}" with value "${value}": ${rule}`,
    ErrorType.VALIDATION,
    ErrorSeverity.LOW,
    'Form Validation',
    userMessage || `Please check the ${field} field.`
  );
};

// Authentication error helper
export const createAuthError = (
  action: string,
  userMessage?: string
): AppError => {
  return new AppError(
    `Authentication required for action: ${action}`,
    ErrorType.AUTHENTICATION,
    ErrorSeverity.HIGH,
    'Authentication',
    userMessage || 'Please log in to continue.'
  );
};

// Authorization error helper
export const createAuthorizationError = (
  resource: string,
  userMessage?: string
): AppError => {
  return new AppError(
    `Access denied to resource: ${resource}`,
    ErrorType.AUTHORIZATION,
    ErrorSeverity.MEDIUM,
    'Authorization',
    userMessage || 'You do not have permission to access this resource.'
  );
};

// React hook for error handling
export const useErrorHandler = () => {
  const reportError = (error: AppError) => {
    ErrorReportingService.getInstance().report(error);
  };

  const handleError = (
    error: Error | string,
    type: ErrorType = ErrorType.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context?: string,
    userMessage?: string
  ) => {
    const appError = error instanceof AppError 
      ? error 
      : new AppError(
          typeof error === 'string' ? error : error.message,
          type,
          severity,
          context,
          userMessage,
          error instanceof Error ? error : undefined
        );
    
    reportError(appError);
  };

  const handleAsyncOperation = async <T>(
    operation: () => Promise<T>,
    context?: string,
    userMessage?: string
  ): Promise<T | null> => {
    return handleAsyncError(operation, context, userMessage);
  };

  const handleNetworkOperation = async <T>(
    operation: () => Promise<T>,
    retries = 3,
    context?: string
  ): Promise<T | null> => {
    return handleNetworkRequest(operation, retries, context);
  };

  return {
    handleError,
    handleAsyncOperation,
    handleNetworkOperation,
    reportError
  };
};

// Global error handler for unhandled promise rejections
export const setupGlobalErrorHandlers = () => {
  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    const error = new AppError(
      `Unhandled promise rejection: ${event.reason}`,
      ErrorType.UNKNOWN,
      ErrorSeverity.HIGH,
      'Global Handler',
      'An unexpected error occurred.'
    );
    
    ErrorReportingService.getInstance().report(error);
    event.preventDefault(); // Prevent console logging
  });

  // Handle global errors
  window.addEventListener('error', (event) => {
    const error = new AppError(
      `Global error: ${event.message}`,
      ErrorType.UNKNOWN,
      ErrorSeverity.HIGH,
      `${event.filename}:${event.lineno}:${event.colno}`,
      'An unexpected error occurred.'
    );
    
    ErrorReportingService.getInstance().report(error);
  });
};

export default ErrorReportingService;