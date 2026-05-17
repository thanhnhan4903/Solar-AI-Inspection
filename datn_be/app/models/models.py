# app/models/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    password_hash = Column(String(255))
    role = Column(String(20), default="user") # admin, user
    
    batches = relationship("UploadBatch", back_populates="user")

class UploadBatch(Base):
    __tablename__ = "upload_batches"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Allow null for now if no auth
    name = Column(String(255))
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="Completed") # Processing, Completed, Failed
    
    user = relationship("User", back_populates="batches")
    images = relationship("Image", back_populates="batch", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="batch", cascade="all, delete-orphan")

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("upload_batches.id"))
    filename = Column(String(255))
    image_type = Column(String(50)) # Thermal, RGB
    path = Column(String(500)) # e.g., data/results/img1.jpg
    
    batch = relationship("UploadBatch", back_populates="images")
    ai_results = relationship("AiResult", back_populates="image", cascade="all, delete-orphan")

class Panel(Base):
    __tablename__ = "panels"
    id = Column(Integer, primary_key=True, index=True)
    local_id = Column(String(50), unique=True, index=True) # Hàng_X_Cột_Y
    x_coord = Column(Float)
    y_coord = Column(Float)
    
    ai_results = relationship("AiResult", back_populates="panel")

class AiResult(Base):
    __tablename__ = "ai_results"
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    panel_id = Column(Integer, ForeignKey("panels.id"))
    defect_type = Column(Text, nullable=True) # Hotspot, Dust, etc., or Healthy
    loss_pct = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    
    image = relationship("Image", back_populates="ai_results")
    panel = relationship("Panel", back_populates="ai_results")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("upload_batches.id"))
    file_path = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    batch = relationship("UploadBatch", back_populates="reports")