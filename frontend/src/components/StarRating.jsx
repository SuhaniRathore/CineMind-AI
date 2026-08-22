import React, { useState } from 'react';
import { Star } from 'lucide-react';

const StarRating = ({ rating, onRate, readOnly = false }) => {
  const [hoverRating, setHoverRating] = useState(0);

  const stars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

  return (
    <div className="flex items-center gap-1">
      {stars.map((star) => {
        const isFilled = star <= (hoverRating || rating);
        
        return (
          <button
            key={star}
            type="button"
            style={{ 
              background: 'transparent', 
              padding: '0.25rem',
              color: isFilled ? '#fbbf24' : 'var(--border-color)',
              cursor: readOnly ? 'default' : 'pointer'
            }}
            onMouseEnter={() => !readOnly && setHoverRating(star)}
            onMouseLeave={() => !readOnly && setHoverRating(0)}
            onClick={() => !readOnly && onRate(star)}
            disabled={readOnly}
          >
            <Star 
              size={24} 
              fill={isFilled ? 'currentColor' : 'none'} 
              strokeWidth={isFilled ? 0 : 2}
            />
          </button>
        );
      })}
      <span className="ml-2 text-muted font-bold" style={{marginLeft: '0.5rem', fontWeight: 600}}>
        {hoverRating || rating || 0} / 10
      </span>
    </div>
  );
};

export default StarRating;
