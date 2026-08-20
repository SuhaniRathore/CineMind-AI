"""
SQLAlchemy ORM model for the 'users' table.

Milestone 3 scope: column structure only.
Milestone 5 (Authentication) adds password hashing and JWT logic to
the service layer — the hashed_password column is defined here now so
the schema is created correctly in this migration.

Design decisions (per DEVELOPMENT_RULES.md rule 5):
- CASCADE DELETE is enforced via ondelete="CASCADE" on the FK in each
  child table (ratings, reviews, watchlist, watch_history,
  recommendation_results). This means DB-level cascades work correctly
  even before the SQLAlchemy ORM relationships are wired.
- hashed_password stores only the bcrypt/argon2 hash — never plaintext.
- email uniqueness is enforced at the DB level (unique=True), not just app.
- is_active flag allows soft-disabling an account without losing data.

Relationships to child tables are added at the milestone that creates each:
  - ratings              → Milestone 6
  - reviews, watchlist   → Milestone 7
  - recommendation_results → Milestone 8
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class User(Base):
    """Registered user account."""

    __tablename__ = "users"
    __table_args__ = (
        # Named constraint so the migration and ORM model agree on the
        # constraint name; the test asserts 'uq_users_email' is present.
        UniqueConstraint("email", name="uq_users_email"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,   # sets col.unique=True for ORM introspection
        index=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"

    # Relationships
    ratings: Mapped[list["Rating"]] = relationship(  # noqa: F821
        "Rating",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    reviews: Mapped[list["Review"]] = relationship(  # noqa: F821
        "Review",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    watchlist: Mapped[list["WatchlistItem"]] = relationship(  # noqa: F821
        "WatchlistItem",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    recommendation_results: Mapped[list["RecommendationResult"]] = relationship(  # noqa: F821
        "RecommendationResult",
        back_populates="user",
        cascade="all, delete-orphan",
    )
