import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    
    try {
      await login(email, password);
      navigate('/');
    } catch (err) {
      const detail = err.response?.data?.detail;
      const errorMsg = typeof detail === 'string' 
        ? detail 
        : (Array.isArray(detail) && detail.length > 0 ? `${detail[0].loc[1]}: ${detail[0].msg}` : 'Failed to login. Please check your credentials.');
      setError(errorMsg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container">
      <div className="auth-container">
        <div className="glass-panel">
          <h1 className="text-center" style={{textAlign: 'center'}}>Welcome Back</h1>
          <p className="text-muted mb-8 text-center" style={{textAlign: 'center', marginBottom: '2rem'}}>Sign in to continue to CineMind AI</p>
          
          <form onSubmit={handleSubmit} className="flex-col gap-4">
            <div>
              <label className="text-muted" style={{display: 'block', marginBottom: '0.5rem'}}>Email</label>
              <input 
                type="email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)} 
                required 
                placeholder="you@example.com"
              />
            </div>
            
            <div>
              <label className="text-muted" style={{display: 'block', marginBottom: '0.5rem'}}>Password</label>
              <input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                required 
                placeholder="••••••••"
              />
            </div>
            
            {error && <div className="error-text">{error}</div>}
            
            <button type="submit" className="btn-primary mt-4 w-full" disabled={isSubmitting}>
              {isSubmitting ? 'Signing in...' : 'Sign In'}
            </button>
          </form>
          
          <p className="text-center mt-8 text-muted" style={{textAlign: 'center', marginTop: '2rem'}}>
            Don&apos;t have an account? <Link to="/register">Register here</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
