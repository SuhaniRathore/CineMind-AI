import React, { useState, useEffect } from 'react';
import api from '../api/client';
import MovieCard from '../components/MovieCard';
import LoadingSpinner from '../components/LoadingSpinner';

const algorithmLabels = {
  content_based: 'Based on your ratings',
  collaborative: 'Collaborative pick',
  hybrid: 'Hybrid pick',
  popularity_based: 'Popular right now',
};

const Recommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchRecs = async () => {
      try {
        const response = await api.get('/recommendations?limit=12');
        setRecommendations(response.data.recommendations);
      } catch (err) {
        console.error("Failed to fetch recommendations", err);
        setError("Could not generate recommendations at this time.");
      } finally {
        setLoading(false);
      }
    };

    fetchRecs();
  }, []);

  return (
    <div className="container">
      <h1 className="mb-8 flex items-center gap-4">
        Recommended For You
      </h1>

      {loading ? (
        <LoadingSpinner />
      ) : error ? (
        <div className="glass-panel text-center">
          <p className="error-text">{error}</p>
        </div>
      ) : (
        <>
          {recommendations.length === 0 ? (
            <div className="glass-panel text-center" style={{textAlign: 'center', padding: '4rem 2rem'}}>
              <h3>Not enough data yet</h3>
              <p className="text-muted mt-2">
                We need to know what you like! Rate a few movies highly (7 or above), and we&apos;ll suggest more you might enjoy.
              </p>
            </div>
          ) : (
            <div className="movie-grid">
              {recommendations.map(rec => (
                <div key={rec.id} style={{position: 'relative'}}>
                  <MovieCard movie={rec.movie} />
                  <div style={{
                    position: 'absolute', 
                    top: '-10px', right: '-10px', 
                    background: 'var(--accent-red)', 
                    color: 'white', 
                    padding: '0.25rem 0.5rem', 
                    borderRadius: '0.5rem',
                    fontSize: '0.875rem',
                    fontWeight: 'bold',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                  }}>
                    {algorithmLabels[rec.algorithm] || 'Recommended'}
                    {rec.algorithm !== 'popularity_based' && (
                      <> · {Math.round(rec.relevance_score * 100)}% Match</>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Recommendations;
