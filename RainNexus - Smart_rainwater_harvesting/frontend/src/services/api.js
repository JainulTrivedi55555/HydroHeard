import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:5000/api'
});

// Attach token to every request automatically
API.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth
export const registerUser = (data) => API.post('/auth/register', data);
export const loginUser = (data) => API.post('/auth/login', data);
export const getMe = () => API.get('/auth/me');

// Data Centers
export const getDataCenters = (params) => API.get('/datacenters', { params });
export const getDataCenterById = (id) => API.get(`/datacenters/${id}`);
export const getStates = () => API.get('/datacenters/states');
export const getCounties = (state) => API.get(`/datacenters/states/${state}/counties`);
export const getStats = (params) => API.get('/datacenters/stats', { params });
export const getStateAnalytics = (state) => API.get('/analytics/state/' + state);

// Rainfall
export const getAllRainfall = () => API.get('/rainfall');
export const getRainfallByState = (state) => API.get(`/rainfall/${state}`);

export default API;