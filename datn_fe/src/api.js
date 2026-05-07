import axios from 'axios';

const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

// Cấu hình axios để gọi API
const api = axios.create({
    baseURL: API_BASE_URL,
});

export const analyzeAll = () => api.post(`/analyze-all`);
export const getMatchPairs = () => api.get(`/match-pairs`);
export const downloadReportUrl = (batchId) => `${API_BASE_URL}/download-report/${batchId}`;
export const IMAGE_URL = "http://127.0.0.1:8000/data/processed/";

export default api;