# backend/app/database/db_setup.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# -------------------------------
# PostgreSQL connection settings
# -------------------------------
DB_USER = "stockuser"
DB_PASSWORD = "yourpassword"  # replace with your actual password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "stockdb"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# -------------------------------
# SQLAlchemy engine & session
# -------------------------------
engine = create_engine(DATABASE_URL, echo=True)  # echo=True for SQL logging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# -------------------------------
# Dependency for FastAPI
# -------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
