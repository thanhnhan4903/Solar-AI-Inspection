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

def check_db():
    db = SessionLocal()
    try:
        batches = db.query(models.UploadBatch).all()
        images = db.query(models.Image).all()
        print(f"[DB Status] Batches: {len(batches)}, Images: {len(images)}")
    finally:
        db.close()

def main():
    print("--- 1. Resetting database via /api/v1/reanalyze ---")
    res = requests.post(f"{BASE_URL}/api/v1/reanalyze")
    print(f"Reanalyze response: {res.json()}")
    check_db()
    
    print("\n--- 2. Starting analysis via /api/v1/analyze-all ---")
    data = {
        "project_name": "Test Project",
        "location": "Ninh Thuan",
        "scan_time": "2026-06-14",
        "operator": "AI Tester",
        "device": "Test Drone",
        "scope": "Block 1",
        "panel_power": 600.0
    }
    res = requests.post(f"{BASE_URL}/api/v1/analyze-all", data=data)
    print(f"Analyze-all response: {res.json()}")
    
    if "error" in res.json():
        print("Error starting analysis. Exiting.")
        return
        
    batch_id = res.json()["batch_id"]
    print(f"Started analysis for Batch ID: {batch_id}")
    
    print("\n--- 3. Polling progress ---")
    start_time = time.time()
    last_current = -1
    while True:
        res = requests.get(f"{BASE_URL}/api/v1/analyze-progress")
        progress = res.json()
        
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
            print(f"Analysis complete in {time.time() - start_time:.1f} seconds!")
            break
            
        time.sleep(1)
        
    print("\n--- 4. Checking DB after completion ---")
    check_db()

if __name__ == "__main__":
    main()
