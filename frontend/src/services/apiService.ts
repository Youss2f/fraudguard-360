/**
 * Enhanced API Service for FraudGuard 360
 * Centralized API communication with retry logic, caching, and error handling
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

// Types
export interface APIResponse<T> {
  data: T;
  status: number;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    total: number;
    page: number;
    pageSize: number;
    totalPages: number;
  };
}

export interface ErrorResponse {
  error: string;
  message: string;
  statusCode: number;
  timestamp: string;
  path: string;
}

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
const TIMEOUT = 30000;
const MAX_RETRIES = 3;

class APIService {
  private api: AxiosInstance;
  private cache: Map<string, { data: any; timestamp: number }> = new Map();
  private requestInterceptorId!: number;
  private responseInterceptorId!: number;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.requestInterceptorId = this.api.interceptors.request.use(
      (config) => {
        // Add authentication token if available
        const token = localStorage.getItem('fraudguard_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Add request timestamp
        config.headers['X-Request-Time'] = new Date().toISOString();
        
        console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.responseInterceptorId = this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        const duration = response.config.headers['X-Request-Time'] 
          ? Date.now() - new Date(response.config.headers['X-Request-Time'] as string).getTime()
          : 0;
        
        console.log(`✅ API Response: ${response.status} ${response.config.url} (${duration}ms)`);
        return response;
      },
      async (error: AxiosError) => {
        const config = error.config as any;
        
        // Retry logic for network errors
        if (config && !config.__isRetryRequest && this.shouldRetry(error)) {
          config.__isRetryRequest = true;
          config.__retryCount = (config.__retryCount || 0) + 1;
          
          if (config.__retryCount <= MAX_RETRIES) {
            const delay = Math.pow(2, config.__retryCount) * 1000; // Exponential backoff
            console.log(`🔄 Retrying request (${config.__retryCount}/${MAX_RETRIES}) after ${delay}ms`);
            
            await new Promise(resolve => setTimeout(resolve, delay));
            return this.api.request(config);
          }
        }

        this.handleError(error);
        return Promise.reject(error);
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    return (
      !error.response || 
      error.code === 'NETWORK_ERROR' ||
      error.code === 'TIMEOUT' ||
      (error.response.status >= 500 && error.response.status < 600)
    );
  }

  private handleError(error: AxiosError) {
    const status = error.response?.status;
    const message = error.response?.data || error.message;
    
    console.error(`❌ API Error (${status}):`, message);
    
    // Handle specific error cases
    switch (status) {
      case 401:
        this.handleUnauthorized();
        break;
      case 403:
        console.error('Access forbidden - insufficient permissions');
        break;
      case 429:
        console.error('Rate limit exceeded - please try again later');
        break;
      case 500:
        console.error('Internal server error - please contact support');
        break;
    }
  }

  private handleUnauthorized() {
    // Clear stored authentication data
    localStorage.removeItem('fraudguard_token');
    localStorage.removeItem('fraudguard_user');
    
    // Redirect to login (would need router context in real app)
    window.location.href = '/login';
  }

  // Cache management
  private getCacheKey(url: string, params?: any): string {
    return `${url}${params ? JSON.stringify(params) : ''}`;
  }

  private getFromCache(key: string, maxAge: number = 5 * 60 * 1000): any {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < maxAge) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  // Public API methods

  async get<T>(endpoint: string, params?: any, useCache = false): Promise<T> {
    const cacheKey = this.getCacheKey(endpoint, params);
    
    if (useCache) {
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        console.log(`📦 Cache hit: ${endpoint}`);
        return cached;
      }
    }

    const response = await this.api.get<T>(endpoint, { params });
    
    if (useCache) {
      this.setCache(cacheKey, response.data);
    }
    
    return response.data;
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    const response = await this.api.post<T>(endpoint, data);
    return response.data;
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    const response = await this.api.put<T>(endpoint, data);
    return response.data;
  }

  async delete<T>(endpoint: string): Promise<T> {
    const response = await this.api.delete<T>(endpoint);
    return response.data;
  }

  async patch<T>(endpoint: string, data?: any): Promise<T> {
    const response = await this.api.patch<T>(endpoint, data);
    return response.data;
  }

  // Specialized methods for FraudGuard endpoints

  async getDashboardKPIs(useCache = true): Promise<any> {
    return this.get('/dashboard/kpis', undefined, useCache);
  }

  async getDashboardAlerts(limit = 20): Promise<any[]> {
    return this.get('/dashboard/alerts', { limit });
  }

  async getDashboardTransactions(limit = 50) {
    return this.get('/dashboard/transactions', { limit });
  }

  async getUserProfile(userId: string) {
    return this.get(`/users/${userId}`, undefined, true);
  }

  async analyzeTransaction(transactionData: any) {
    return this.post('/fraud/analyze', transactionData);
  }

  async getFraudPatterns(timeRange: string = '24h') {
    return this.get('/fraud/patterns', { timeRange }, true);
  }

  async getNetworkAnalysis(userId: string, depth = 2) {
    return this.get(`/network/analyze/${userId}`, { depth });
  }

  async updateAlertStatus(alertId: string, status: string) {
    return this.patch(`/alerts/${alertId}`, { status });
  }

  async generateReport(reportType: string, filters: any) {
    return this.post('/reports/generate', { type: reportType, filters });
  }

  // WebSocket connection
  createWebSocket(endpoint: string): WebSocket {
    const wsUrl = `${WS_BASE_URL}${endpoint}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log(`🔌 WebSocket connected: ${endpoint}`);
    };
    
    ws.onclose = () => {
      console.log(`🔌 WebSocket disconnected: ${endpoint}`);
    };
    
    ws.onerror = (error) => {
      console.error(`❌ WebSocket error on ${endpoint}:`, error);
    };
    
    return ws;
  }

  // Health check
  async healthCheck(): Promise<any> {
    try {
      return await this.get('/health');
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Cleanup
  destroy() {
    this.api.interceptors.request.eject(this.requestInterceptorId);
    this.api.interceptors.response.eject(this.responseInterceptorId);
    this.cache.clear();
  }
}

// Export singleton instance
export const apiService = new APIService();
export default apiService;