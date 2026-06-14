import sys
import os
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from app.core.database import SessionLocal
from app.models import models

BASE_URL = "http://127.0.0.1:8000"

def main():
    db = SessionLocal()
    try:
        # Get the latest batch
        latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
        if not latest_batch:
            print("No latest batch found. Run test_ai_pipeline.py first.")
            return
            
        print(f"Current Latest Batch ID: {latest_batch.id}")
        images = db.query(models.Image).filter(models.Image.batch_id == latest_batch.id).all()
        print(f"Images in latest batch: {len(images)}")
        
        # Simulate that 3 images are newly uploaded (by deleting their DB records, but keeping them on disk in precalib)
        simulated_new = [img.filename for img in images[-3:]]
        print(f"Simulating 3 new images: {simulated_new}")
        
        # Delete their records from database, respecting foreign keys
        image_ids = [img.id for img in images[-3:]]
        db.query(models.AiResult).filter(models.AiResult.image_id.in_(image_ids)).delete(synchronize_session=False)
        db.query(models.Image).filter(models.Image.id.in_(image_ids)).delete(synchronize_session=False)
        db.commit()
        
        # Verify the database state
        images_left = db.query(models.Image).all()
        print(f"Images left in DB (old images): {len(images_left)}")
        
        print("\n--- Starting analysis for new uploads ---")
        data = {
            "project_name": "Test Project with Added Photos",
            "location": "Ninh Thuan",
            "scan_time": "2026-06-14",
            "operator": "AI Tester",
            "device": "Test Drone",
            "scope": "Block 1",
            "panel_power": 600.0
        }
        res = requests.post(f"{BASE_URL}/api/v1/analyze-all", data=data)
        res_json = res.json()
        print(f"Analyze-all response: {res_json}")
        
        new_batch_id = res_json["batch_id"]
        print(f"New Batch ID: {new_batch_id}")
        
        # Poll progress
        last_current = -1
        while True:
            progress_res = requests.get(f"{BASE_URL}/api/v1/analyze-progress")
            progress = progress_res.json()
            
            current = progress.get("current", 0)
            total = progress.get("total", 0)
            step = progress.get("step", "")
            running = progress.get("running", False)
            done = progress.get("done", False)
            filename = progress.get("filename", "")
            
            if current != last_current or step != progress.get("step", ""):
                print(f"Progress: {current}/{total} | Step: {step} | File: {filename} | Running: {running} | Done: {done}")
                last_current = current
                
            if done and not running:
                print("Analysis of new photos complete!")
                break
                
            time.sleep(1)
            
        # Verify that the new batch has ALL 23 images (20 old + 3 new)
        total_in_new_batch = db.query(models.Image).filter(models.Image.batch_id == new_batch_id).count()
        print(f"\nFinal images in new Batch {new_batch_id}: {total_in_new_batch} (expected: 23)")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
