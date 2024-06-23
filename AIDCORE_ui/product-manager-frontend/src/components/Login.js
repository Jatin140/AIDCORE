import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useApiKey } from '../context/ApiKeyContext';
import axios from 'axios';

function Login() {
  console.log("Rendering Login component");

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [openaiKey, setOpenaiKey] = useState('');
  const navigate = useNavigate();
  const { login } = useAuth();
  const { setApiKey } = useApiKey();

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Login form submitted");
    try {
      console.log("Sending login request");
      const response = await axios.post('http://localhost:8000/api/token/', { username, password });
      console.log("Login response:", response.data);
      login({ token: response.data.access, ...response.data });
      if (openaiKey) {
        setApiKey(openaiKey);
      }
      console.log("Navigating to:", response.data.is_superuser ? '/admin' : '/user');
      navigate(response.data.is_superuser ? '/admin' : '/user');
    } catch (error) {
      console.error('Login failed', error);
    }
  };

  console.log("Login component rendered");

  return (
    <div style={{ maxWidth: 300, margin: 'auto', marginTop: 50 }}>
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="username">Username:</label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div>
          <label htmlFor="password">Password:</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <div>
          <label htmlFor="openaiKey">OpenAI API Key (Optional):</label>
          <input
            id="openaiKey"
            type="text"
            value={openaiKey}
            onChange={(e) => setOpenaiKey(e.target.value)}
          />
        </div>
        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default Login;