import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import SessionLocal
from app.models import models
import requests

db = SessionLocal()
try:
    batches = db.query(models.UploadBatch).all()
    print(f"Total Batches: {len(batches)}")
    for b in batches:
        images_count = db.query(models.Image).filter(models.Image.batch_id == b.id).count()
        print(f"  Batch ID: {b.id}, Name: {b.name}, Images Count: {images_count}, Date: {b.upload_date}")
        
    print("\nImages in DB:")
    images = db.query(models.Image).all()
    print(f"Total Images: {len(images)}")
    for img in images[:10]:
        print(f"  Image ID: {img.id}, Batch ID: {img.batch_id}, Filename: {img.filename}")
    if len(images) > 10:
        print("  ...")
        
    print("\nProgress State via API:")
    try:
        res = requests.get("http://127.0.0.1:8000/api/v1/analyze-progress")
        print(res.json())
    except Exception as e:
        print(f"Could not connect to API: {e}")
finally:
    db.close()
