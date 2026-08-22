import React, { useState, useEffect } from 'react';
import api from '../api/client';
import MovieCard from '../components/MovieCard';
import LoadingSpinner from '../components/LoadingSpinner';

const Watchlist = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchWatchlist = async () => {
      try {
        const response = await api.get('/watchlist');
        setItems(response.data.items);
      } catch (error) {
        console.error("Failed to fetch watchlist", error);
      } finally {
        setLoading(false);
      }
    };

    fetchWatchlist();
  }, []);

  return (
    <div className="container">
      <h1 className="mb-8">My Watchlist</h1>

      {loading ? (
        <LoadingSpinner />
      ) : (
        <>
          {items.length === 0 ? (
            <div className="glass-panel text-center" style={{textAlign: 'center', padding: '4rem 2rem'}}>
              <h3>Your watchlist is empty</h3>
              <p className="text-muted mt-2">Find some movies you want to watch and add them here!</p>
            </div>
          ) : (
            <div className="movie-grid">
              {items.map(item => (
                <MovieCard key={item.id} movie={item.movie} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Watchlist;
