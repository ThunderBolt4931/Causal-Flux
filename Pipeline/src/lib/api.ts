// API configuration - automatically uses relative URLs in production (Docker)
// and localhost:4000 in development

const isDevelopment = import.meta.env.DEV;

// In Docker: use relative URLs (nginx proxies to backend)
// In development: use direct localhost:4000
export const API_BASE_URL = isDevelopment ? 'http://localhost:4000' : '';

// Helper to build API URLs
export function apiUrl(path: string): string {
    // Ensure path starts with /
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    return `${API_BASE_URL}${normalizedPath}`;
}
