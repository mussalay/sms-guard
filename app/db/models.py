"""Persistence and ownership constraints for the local application."""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(30), unique=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    date_joined: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    profile: Mapped["UserProfile"] = relationship(cascade="all, delete-orphan", uselist=False)


class UserProfile(Base):
    __tablename__ = "user_profile"
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    store_sms_text: Mapped[bool] = mapped_column(Boolean, default=False)
    preferred_language: Mapped[str] = mapped_column(String(5), default="ru")


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class Check(Base):
    __tablename__ = "checks"
    __table_args__ = (
        Index("ix_checks_user_created", "user_id", "created_at"),
        Index("ix_checks_category", "category"),
        UniqueConstraint("user_id", "request_key"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    request_key: Mapped[str] = mapped_column(String(100))
    check_type: Mapped[str] = mapped_column(String(3))
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_masked: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10))
    category: Mapped[str] = mapped_column(String(20))
    risk_level: Mapped[str] = mapped_column(String(10))
    risk_score: Mapped[int] = mapped_column(Integer)
    p_sms: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation: Mapped[list] = mapped_column(JSON)
    recommendation: Mapped[str] = mapped_column(Text)
    sms_model_version: Mapped[str | None] = mapped_column(String(30), nullable=True)
    url_model_version: Mapped[str | None] = mapped_column(String(30), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    urls: Mapped[list["UrlResult"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
    detected: Mapped[list["DetectedFeature"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class UrlResult(Base):
    __tablename__ = "url_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    check_id: Mapped[int] = mapped_column(ForeignKey("checks.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(Text)
    p_url: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String(20))
    features_json: Mapped[dict] = mapped_column(JSON)


class DetectedFeature(Base):
    __tablename__ = "detected_features"
    id: Mapped[int] = mapped_column(primary_key=True)
    check_id: Mapped[int] = mapped_column(ForeignKey("checks.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, default=1.0)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_type: Mapped[str] = mapped_column(String(5))
    version: Mapped[str] = mapped_column(String(30))
    algorithm: Mapped[str] = mapped_column(String(80))
    metrics_json: Mapped[dict] = mapped_column(JSON)
    threshold: Mapped[float] = mapped_column(Float)
    trained_at: Mapped[str] = mapped_column(String(60))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
