import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Register = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    
    try {
      await register(email, password, displayName);
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to register. Please check your inputs.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container">
      <div className="auth-container">
        <div className="glass-panel">
          <h1 style={{textAlign: 'center'}}>Create Account</h1>
          <p className="text-muted" style={{textAlign: 'center', marginBottom: '2rem'}}>Join CineMind AI today</p>
          
          <form onSubmit={handleSubmit} className="flex-col gap-4">
            <div>
              <label className="text-muted" style={{display: 'block', marginBottom: '0.5rem'}}>Display Name</label>
              <input 
                type="text" 
                value={displayName} 
                onChange={(e) => setDisplayName(e.target.value)} 
                required 
                placeholder="John Doe"
              />
            </div>

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
              {isSubmitting ? 'Creating...' : 'Create Account'}
            </button>
          </form>
          
          <p className="mt-8 text-muted" style={{textAlign: 'center', marginTop: '2rem'}}>
            Already have an account? <Link to="/login">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Register;
