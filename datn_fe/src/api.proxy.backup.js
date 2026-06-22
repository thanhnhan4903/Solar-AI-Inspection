import axios from 'axios';

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || "http://127.0.0.1:8000";
const API_BASE_URL = `${API_ORIGIN}/api/v1`;

// Cấu hình axios để gọi API
const api = axios.create({
    baseURL: API_BASE_URL,
});

export const fetchLatestBatch = () => api.get("/latest-batch");
export const analyzeAll = (formData) => api.post("/analyze-all", formData);
export const getAnalyzeProgress = () => api.get("/analyze-progress");
export const uploadDroneData = (formData) => api.post("/upload-drone-data", formData);
export const processThermal = () => api.get("/process-thermal");
export const updateBatchMetadata = (payload) => api.post("/update-batch-metadata", payload);
export const fetchReviewItems = (batchId) => api.get(`/review/items${batchId ? `?batch_id=${batchId}` : ""}`);
export const updateReviewItem = (reviewItemId, payload) => api.post(`/review/items/${reviewItemId}`, payload);
export const syncReview = (batchId) => api.post("/review/sync", { batch_id: batchId });
export const fetchReviewSummary = (batchId) => api.get(`/review/summary${batchId ? `?batch_id=${batchId}` : ""}`);
export const reanalyze = () => api.post("/reanalyze");
export const resetSystem = () => api.post("/reset-system");
export const updateAiModel = (formData) => api.post("/update-ai-model", formData);

export const getMatchPairs = () => api.get(`/match-pairs`);
export const downloadReportUrl = (batchId) => `${API_BASE_URL}/download-report/${batchId}`;
export const IMAGE_URL = `${API_ORIGIN}/data/processed/`;

export const loginUser = async (username, password) => {
    return await api.post("/login", { username, password });
};

export default api;
