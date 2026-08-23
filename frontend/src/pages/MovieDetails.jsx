import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Bookmark, BookmarkCheck, Clock, Calendar } from 'lucide-react';
import api from '../api/client';
import LoadingSpinner from '../components/LoadingSpinner';
import StarRating from '../components/StarRating';

const MovieDetails = () => {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const [userRating, setUserRating] = useState(0);
  const [inWatchlist, setInWatchlist] = useState(false);
  const [watchlistItemId, setWatchlistItemId] = useState(null);
  
  const [reviews, setReviews] = useState([]);
  const [newReviewTitle, setNewReviewTitle] = useState('');
  const [newReviewBody, setNewReviewBody] = useState('');
  const [reviewError, setReviewError] = useState('');

  useEffect(() => {
    const fetchMovieData = async () => {
      setLoading(true);
      try {
        const movieRes = await api.get(`/movies/${id}`);
        setMovie(movieRes.data);
        
        // Fetch user's rating (we'd ideally have a specific endpoint or check if it's in a list)
        // For MVP, we might just try to fetch their ratings or catch 404
        try {
          // This requires backend support for fetching a specific rating, but we can list ratings and find it
          const ratingsRes = await api.get(`/ratings?movie_id=${id}`);
          if (ratingsRes.data.ratings.length > 0) {
            setUserRating(ratingsRes.data.ratings[0].rating);
          }
        } catch (e) { /* ignore */ }

        // Fetch watchlist status
        try {
          const wlRes = await api.get('/watchlist');
          const foundItem = wlRes.data.items.find(item => item.movie.id === parseInt(id));
          if (foundItem) {
            setInWatchlist(true);
            setWatchlistItemId(foundItem.id);
          } else {
            setInWatchlist(false);
            setWatchlistItemId(null);
          }
        } catch (e) { /* ignore */ }

        // Fetch reviews
        fetchReviews();

      } catch (error) {
        console.error("Failed to fetch movie", error);
      } finally {
        setLoading(false);
      }
    };

    fetchMovieData();
  }, [id]);

  const fetchReviews = async () => {
    try {
      const res = await api.get(`/reviews?movie_id=${id}`);
      setReviews(res.data.reviews);
    } catch (e) {
      console.error("Failed to fetch reviews", e);
    }
  };

  const handleRate = async (ratingValue) => {
    try {
      await api.post('/ratings', { movie_id: parseInt(id), rating: ratingValue });
      setUserRating(ratingValue);
    } catch (e) {
      console.error("Failed to rate", e);
    }
  };

  const toggleWatchlist = async () => {
    try {
      if (inWatchlist && watchlistItemId) {
        await api.delete(`/watchlist/${watchlistItemId}`);
        setInWatchlist(false);
        setWatchlistItemId(null);
      } else {
        const res = await api.post('/watchlist', { movie_id: parseInt(id) });
        setInWatchlist(true);
        setWatchlistItemId(res.data.id);
      }
    } catch (e) {
      console.error("Failed to toggle watchlist", e);
    }
  };

  const submitReview = async (e) => {
    e.preventDefault();
    setReviewError('');
    if (newReviewBody.length < 10) {
      setReviewError("Review body must be at least 10 characters long.");
      return;
    }
    
    try {
      await api.post('/reviews', { 
        movie_id: parseInt(id), 
        title: newReviewTitle, 
        body: newReviewBody 
      });
      setNewReviewTitle('');
      setNewReviewBody('');
      fetchReviews(); // refresh
    } catch (err) {
      setReviewError(err.response?.data?.detail || "Failed to submit review");
    }
  };

  if (loading) return <LoadingSpinner />;
  if (!movie) return <div className="container mt-8">Movie not found</div>;

  const posterUrl = movie.poster_url || 'https://via.placeholder.com/500x750?text=No+Poster';

  return (
    <div className="container">
      <div className="flex gap-4 flex-col" style={{flexDirection: 'row', gap: '3rem', flexWrap: 'wrap'}}>
        
        {/* Left Col: Poster & Actions */}
        <div style={{flex: '1', minWidth: '300px', maxWidth: '400px'}}>
          <img 
            src={posterUrl} 
            alt={movie.title} 
            className="w-full" 
            style={{borderRadius: '1rem', boxShadow: '0 20px 40px rgba(0,0,0,0.5)'}}
          />
          
          <div className="glass-panel mt-8">
            <h3>Your Rating</h3>
            <div className="mt-4">
              <StarRating rating={userRating} onRate={handleRate} />
            </div>

            <button 
              onClick={toggleWatchlist} 
              className={`btn-${inWatchlist ? 'secondary' : 'primary'} w-full mt-8 flex items-center justify-center gap-2`}
            >
              {inWatchlist ? <BookmarkCheck size={20} /> : <Bookmark size={20} />}
              {inWatchlist ? 'Remove from Watchlist' : 'Add to Watchlist'}
            </button>
          </div>
        </div>

        {/* Right Col: Info & Reviews */}
        <div style={{flex: '2', minWidth: '300px'}}>
          <h1 style={{fontSize: '3.5rem', marginBottom: '0.5rem', lineHeight: '1.1'}}>{movie.title}</h1>
          
          <div className="flex items-center gap-4 text-muted mb-8" style={{gap: '1.5rem'}}>
            <span className="flex items-center gap-2"><Calendar size={18}/> {movie.release_date || 'Unknown'}</span>
            <span className="flex items-center gap-2"><Clock size={18}/> {movie.runtime ? `${movie.runtime} min` : 'Unknown'}</span>
            <span className="rating-badge" style={{fontSize: '1.1rem'}}>★ {movie.vote_average?.toFixed(1) || '0.0'}</span>
          </div>

          <div className="flex gap-2 mb-8" style={{flexWrap: 'wrap'}}>
            {movie.genres.map(g => (
              <span key={g.id} style={{
                background: 'rgba(255,255,255,0.1)', 
                padding: '0.25rem 1rem', 
                borderRadius: '2rem',
                fontSize: '0.875rem'
              }}>
                {g.name}
              </span>
            ))}
          </div>

          <h2 className="mt-8">Overview</h2>
          <p style={{fontSize: '1.1rem', lineHeight: '1.6', color: 'var(--text-secondary)'}}>
            {movie.overview || "No overview available."}
          </p>

          <hr style={{margin: '3rem 0', borderColor: 'var(--border-color)', borderBottom: 'none'}} />

          <h2>Reviews</h2>
          
          {/* Write a review */}
          <div className="glass-panel mb-8">
            <h3 className="mb-4">Write a Review</h3>
            <form onSubmit={submitReview} className="flex-col gap-4">
              <input 
                type="text" 
                placeholder="Review Title" 
                value={newReviewTitle}
                onChange={e => setNewReviewTitle(e.target.value)}
                required
                minLength={3}
                maxLength={150}
              />
              <textarea 
                placeholder="What did you think of this movie?"
                rows={4}
                value={newReviewBody}
                onChange={e => setNewReviewBody(e.target.value)}
                required
                minLength={10}
                maxLength={5000}
              />
              {reviewError && <div className="error-text">{reviewError}</div>}
              <button type="submit" className="btn-primary" style={{alignSelf: 'flex-start'}}>Submit Review</button>
            </form>
          </div>

          {/* List reviews */}
          <div className="flex-col gap-4">
            {reviews.length === 0 ? (
              <p className="text-muted">No reviews yet. Be the first!</p>
            ) : (
              reviews.map(r => (
                <div key={r.id} className="glass-panel" style={{padding: '1.5rem'}}>
                  <div className="flex justify-between items-center mb-2">
                    <h3 style={{margin: 0}}>{r.title}</h3>
                    <span className="text-muted" style={{fontSize: '0.875rem'}}>User #{r.user_id}</span>
                  </div>
                  <p style={{color: 'var(--text-secondary)', lineHeight: '1.5'}}>{r.body}</p>
                </div>
              ))
            )}
          </div>

        </div>
      </div>
    </div>
  );
};

export default MovieDetails;
