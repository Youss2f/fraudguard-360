/**
 * Service Worker for FraudGuard 360
 * Implements aggressive caching strategy for optimal performance
 */

const CACHE_NAME = 'fraudguard-360-v1.0.0';
const STATIC_CACHE = 'fraudguard-static-v1.0.0';
const DYNAMIC_CACHE = 'fraudguard-dynamic-v1.0.0';

// Critical resources to cache immediately
const STATIC_ASSETS = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/manifest.json',
  '/favicon.ico'
];

// API endpoints to cache with strategy
const API_CACHE_PATTERNS = [
  '/api/dashboard/',
  '/api/analytics/',
  '/api/reports/',
  '/api/monitoring/'
];

// Install event - cache critical resources
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Install event');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[ServiceWorker] Pre-caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[ServiceWorker] Installation complete');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('[ServiceWorker] Installation failed:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activate event');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== CACHE_NAME) {
              console.log('[ServiceWorker] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[ServiceWorker] Activation complete');
        return self.clients.claim();
      })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-HTTP requests
  if (!request.url.startsWith('http')) {
    return;
  }

  // Strategy 1: Cache First for static assets
  if (isStaticAsset(request.url)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Strategy 2: Network First for API calls with offline fallback
  if (isAPIRequest(request.url)) {
    event.respondWith(networkFirstWithCache(request));
    return;
  }

  // Strategy 3: Stale While Revalidate for navigation
  if (request.mode === 'navigate') {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  // Default: Network with cache fallback
  event.respondWith(networkWithCacheFallback(request));
});

// Cache First Strategy - for static assets
async function cacheFirst(request) {
  try {
    const cache = await caches.open(STATIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      console.log('[ServiceWorker] Cache hit for:', request.url);
      return cachedResponse;
    }

    console.log('[ServiceWorker] Cache miss, fetching:', request.url);
    const networkResponse = await fetch(request);
    
    if (networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('[ServiceWorker] Cache first failed:', error);
    return new Response('Network error', { status: 503 });
  }
}

// Network First with Cache - for API calls
async function networkFirstWithCache(request) {
  try {
    const cache = await caches.open(DYNAMIC_CACHE);
    
    try {
      const networkResponse = await fetch(request);
      
      if (networkResponse.status === 200) {
        console.log('[ServiceWorker] API response cached:', request.url);
        cache.put(request, networkResponse.clone());
      }
      
      return networkResponse;
    } catch (networkError) {
      console.log('[ServiceWorker] Network failed, trying cache:', request.url);
      const cachedResponse = await cache.match(request);
      
      if (cachedResponse) {
        return cachedResponse;
      }
      
      // Return offline fallback for API calls
      return new Response(
        JSON.stringify({
          error: 'Offline',
          message: 'This feature is not available offline'
        }),
        {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
  } catch (error) {
    console.error('[ServiceWorker] Network first failed:', error);
    return new Response('Service unavailable', { status: 503 });
  }
}

// Stale While Revalidate - for navigation
async function staleWhileRevalidate(request) {
  try {
    const cache = await caches.open(STATIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    // Start fetch in background
    const fetchPromise = fetch(request).then(response => {
      if (response.status === 200) {
        cache.put(request, response.clone());
      }
      return response;
    });

    // Return cached version immediately if available
    if (cachedResponse) {
      console.log('[ServiceWorker] Serving stale content:', request.url);
      return cachedResponse;
    }

    // Wait for network if no cache
    return await fetchPromise;
  } catch (error) {
    console.error('[ServiceWorker] Stale while revalidate failed:', error);
    return new Response('Network error', { status: 503 });
  }
}

// Network with Cache Fallback - default strategy
async function networkWithCacheFallback(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[ServiceWorker] Network failed, trying cache:', request.url);
    
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response('Network error', { status: 503 });
  }
}

// Helper functions
function isStaticAsset(url) {
  return url.includes('/static/') || 
         url.includes('/favicon.ico') || 
         url.includes('/manifest.json') ||
         url.includes('.css') ||
         url.includes('.js') ||
         url.includes('.png') ||
         url.includes('.jpg') ||
         url.includes('.svg');
}

function isAPIRequest(url) {
  return url.includes('/api/') || 
         API_CACHE_PATTERNS.some(pattern => url.includes(pattern));
}

// Message handling for cache control
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    clearAllCaches().then(() => {
      event.ports[0].postMessage({ success: true });
    });
  }
});

// Clear all caches
async function clearAllCaches() {
  try {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames.map(cacheName => caches.delete(cacheName))
    );
    console.log('[ServiceWorker] All caches cleared');
  } catch (error) {
    console.error('[ServiceWorker] Failed to clear caches:', error);
  }
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(
      // Handle background sync operations
      console.log('[ServiceWorker] Background sync triggered')
    );
  }
});

// Push notifications handling
self.addEventListener('push', (event) => {
  if (event.data) {
    const notificationData = event.data.json();
    
    event.waitUntil(
      self.registration.showNotification(notificationData.title, {
        body: notificationData.body,
        icon: '/favicon.ico',
        badge: '/favicon.ico',
        tag: 'fraudguard-notification',
        requireInteraction: true,
        actions: [
          {
            action: 'view',
            title: 'View Details'
          },
          {
            action: 'dismiss',
            title: 'Dismiss'
          }
        ]
      })
    );
  }
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

console.log('[ServiceWorker] Service Worker loaded successfully');