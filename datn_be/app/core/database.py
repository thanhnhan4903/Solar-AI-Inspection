# app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# mysql+pymysql://user:password@localhost:3306/database_name
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:nhan4903@localhost:3306/solar_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Hàm bổ trợ để lấy Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()