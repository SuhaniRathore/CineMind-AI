import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import api from '../api/client';
import MovieCard from '../components/MovieCard';
import LoadingSpinner from '../components/LoadingSpinner';

const Home = () => {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchMovies = async () => {
      setLoading(true);
      setError('');
      try {
        let url = '/movies?limit=24';
        if (searchQuery) {
          url = `/movies/search?limit=24&q=${encodeURIComponent(searchQuery)}`;
        }
        const response = await api.get(url);
        setMovies(response.data.movies);
      } catch (fetchError) {
        console.error("Failed to fetch movies", fetchError);
        setError('Could not load movies. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchMovies();
  }, [searchQuery]);

  const handleSearch = (e) => {
    e.preventDefault();
    setSearchQuery(searchTerm);
  };

  return (
    <div className="container">
      <div className="flex justify-between items-center mb-8">
        <h1>{searchQuery ? `Search Results: ${searchQuery}` : 'Popular Movies'}</h1>
        
        <form onSubmit={handleSearch} className="flex gap-2" style={{maxWidth: '400px', width: '100%'}}>
          <input 
            type="text" 
            placeholder="Search movies..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <button type="submit" className="btn-secondary flex items-center gap-2">
            <Search size={18} />
          </button>
        </form>
      </div>

      {loading ? (
        <LoadingSpinner />
      ) : error ? (
        <div className="glass-panel text-center">
          <p className="error-text">{error}</p>
        </div>
      ) : (
        <>
          {movies.length === 0 ? (
            <div className="glass-panel text-center" style={{textAlign: 'center'}}>
              <h3>No movies found</h3>
              <p className="text-muted mt-2">Try a different search term.</p>
            </div>
          ) : (
            <div className="movie-grid">
              {movies.map(movie => (
                <MovieCard key={movie.id} movie={movie} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Home;
