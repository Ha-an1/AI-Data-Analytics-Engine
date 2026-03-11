import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# By default, use SQLite locally. Can be overridden with an environment variable for Postgres.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///analytics_engine.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database, creating all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
