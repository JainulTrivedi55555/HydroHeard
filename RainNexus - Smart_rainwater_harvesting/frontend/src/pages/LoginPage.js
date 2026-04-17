import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import { useAuth } from '../context/AuthContext';
import './LoginPage.css';

const LoginPage = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();
  const { loginWithRedirect } = useAuth0();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isRegister) {
        await register(name, email, password);
      } else {
        await login(email, password);
      }
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong');
    }
    setLoading(false);
  };

  const handleGuest = async () => {
    setError('');
    setLoading(true);
    try {
      try {
        await login('guest@rainuse.com', 'guest123456');
      } catch {
        await register('Guest User', 'guest@rainuse.com', 'guest123456');
      }
      navigate('/dashboard');
    } catch (err) {
      setError('Guest login failed. Try registering.');
    }
    setLoading(false);
  };

  const handleAuth0Login = () => {
    loginWithRedirect();
  };

  return (
    <div className="login-screen">
      <div className="login-particles">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="particle" style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 5}s`,
            animationDuration: `${3 + Math.random() * 4}s`
          }} />
        ))}
      </div>

      <div className="login-box">
        <div className="login-logo">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <path d="M16 3C16 3 6 15 6 21C6 26.5 10.5 29 16 29C21.5 29 26 26.5 26 21C26 15 16 3 16 3Z" fill="#0ea5e9" opacity=".4" stroke="#0ea5e9" strokeWidth="2"/>
            <path d="M16 10C16 10 11 17 11 20C11 23 13 25 16 25C19 25 21 23 21 20C21 17 16 10 16 10Z" fill="#4ade80" opacity=".5"/>
          </svg>
          <span className="login-title">RAINUSE NEXUS</span>
        </div>

        <p className="login-tagline">
          Automated water prospecting engine for commercial &amp; industrial data centers across the continental US
        </p>

        {error && <div className="login-error">{error}</div>}

        <button className="login-btn auth0" onClick={handleAuth0Login}>
          SIGN IN WITH AUTH0
        </button>

        <div className="login-divider">OR USE EMAIL</div>

        <form onSubmit={handleSubmit}>
          {isRegister && (
            <input
              className="login-input"
              type="text"
              placeholder="Full name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          )}
          <input
            className="login-input"
            type="email"
            placeholder="Email address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            className="login-input"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
          />
          <button className="login-btn primary" type="submit" disabled={loading}>
            {loading ? 'PLEASE WAIT...' : isRegister ? 'CREATE ACCOUNT' : 'LOGIN WITH EMAIL'}
          </button>
        </form>

        <div className="login-divider">• • •</div>

        <button className="login-btn guest" onClick={handleGuest} disabled={loading}>
          CONTINUE AS GUEST
        </button>

        <button
          className="login-switch"
          onClick={() => { setIsRegister(!isRegister); setError(''); }}
        >
          {isRegister ? 'Already have an account? Login' : "Don't have an account? Register"}
        </button>

        <div className="login-badge">
          <span className="login-dot" />
          POWERED BY AUTH0
        </div>

        <div className="sponsor-strip">
          <span className="sponsor-tag">GRUNDFOS</span>
          <span className="sponsor-tag">GEMINI AI</span>
          <span className="sponsor-tag">ELEVENLABS</span>
          <span className="sponsor-tag">DIGITALOCEAN</span>
          <span className="sponsor-tag">SOLANA</span>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;