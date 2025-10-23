import axios from 'axios';

const API_URL = 'http://127.0.0.1:48127';

const api = axios.create({
  baseURL: API_URL,
  timeout: 180000, // 3 minutes for LLM processing
});

// Retry connection
const retryRequest = async (fn, retries = 5, delay = 2000) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
};

export const queryErrorCode = async (query) => {
  // Backend uses hybrid mode: instant for direct queries, LLM for complex ones
  const response = await api.post('/api/query', { query, preset: 'hybrid' });
  return response.data;
};

export const healthCheck = async () => {
  return await retryRequest(async () => {
    const response = await api.get('/api/health');
    return response.data;
  });
};

export const getConfig = async () => {
  const response = await api.get('/api/config');
  return response.data;
};
