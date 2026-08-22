import React from 'react';
import { Link } from 'react-router-dom';
import { Star } from 'lucide-react';

const MovieCard = ({ movie }) => {
  const posterUrl = movie.poster_url || 'https://via.placeholder.com/500x750?text=No+Poster';
  const year = movie.release_date ? movie.release_date.substring(0, 4) : 'N/A';
  
  return (
    <Link to={`/movies/${movie.id}`} className="movie-card">
      <img src={posterUrl} alt={movie.title} className="movie-poster" loading="lazy" />
      <div className="movie-info">
        <h3 className="movie-title">{movie.title}</h3>
        <div className="movie-meta">
          <span>{year}</span>
          {movie.vote_average > 0 && (
            <span className="rating-badge">
              <Star size={14} fill="currentColor" />
              {movie.vote_average.toFixed(1)}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
};

export default MovieCard;
