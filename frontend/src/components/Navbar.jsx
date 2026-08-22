import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Film, User, LogOut, Bookmark, Sparkles } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Navbar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  if (!user) return null;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="container">
        <Link to="/" className="logo">
          CineMind<span>AI</span>
        </Link>
        <div className="nav-links">
          <Link to="/" className="flex items-center gap-2">
            <Film size={20} /> Movies
          </Link>
          <Link to="/watchlist" className="flex items-center gap-2">
            <Bookmark size={20} /> Watchlist
          </Link>
          <Link to="/recommendations" className="flex items-center gap-2">
            <Sparkles size={20} className="text-red-500" /> For You
          </Link>
          <div className="flex items-center gap-2 ml-4">
            <User size={20} />
            <span>{user.display_name}</span>
          </div>
          <button onClick={handleLogout} className="flex items-center gap-2 text-muted hover:text-white ml-2" style={{background: 'transparent'}}>
            <LogOut size={20} />
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
