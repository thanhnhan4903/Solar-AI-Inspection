# app/models/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base
from sqlalchemy.sql import func
from sqlalchemy import Text

class InspectionBatch(Base):
    __tablename__ = "inspection_batches"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255)) # Ví dụ: "Đợt bay khu vực A - 05/2026"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    panels = relationship("PanelAnalysis", back_populates="batch")

class PanelAnalysis(Base):
    __tablename__ = "panel_analyses"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("inspection_batches.id"))
    filename = Column(String(255))
    
    # Định vị X, Y theo pixel (Thay cho GPS)
    x_coord = Column(Float)
    y_coord = Column(Float)
    local_id = Column(String(50)) # Hàng_X_Cột_Y
    
    # Kết quả phân tích
    defect_type = Column(Text, nullable=True)
    loss_pct = Column(Float)
    confidence = Column(Float)
    
    batch = relationship("InspectionBatch", back_populates="panels")