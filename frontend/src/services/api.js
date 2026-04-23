import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Send a user query to the backend and return the full response.
 */
export async function postQuery(text) {
  try {
    const response = await axios.post(`${API_URL}/query`, { text });
    return { success: true, data: response.data };
  } catch (error) {
    const message =
      error.response?.data?.detail || error.message || 'Unknown error';
    return { success: false, error: message };
  }
}