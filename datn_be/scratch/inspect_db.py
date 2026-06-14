import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models import models

def main():
    db = SessionLocal()
    try:
        # Get latest batch
        latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
        if not latest_batch:
            with open("scratch/defect_details.txt", "w", encoding="utf-8") as f:
                f.write("No upload batches found!")
            return
        
        lines = []
        lines.append(f"Latest Batch: ID={latest_batch.id}, Name={latest_batch.name}")
        
        images = db.query(models.Image).filter(models.Image.batch_id == latest_batch.id).all()
        lines.append(f"Total images in batch: {len(images)}")
        
        defect_count = 0
        all_results_count = 0
        for img in images:
            ai_results = db.query(models.AiResult).filter(models.AiResult.image_id == img.id).all()
            all_results_count += len(ai_results)
            for r in ai_results:
                has_defect = False
                defects_in_json = []
                if r.defect_type:
                    if r.defect_type.startswith("{"):
                        try:
                            d = json.loads(r.defect_type)
                            defects_in_json = d.get("defects", [])
                            if defects_in_json:
                                has_defect = True
                        except Exception:
                            pass
                    elif r.defect_type != "Healthy":
                        has_defect = True
                
                if has_defect:
                    defect_count += 1
                    panel = db.query(models.Panel).filter(models.Panel.id == r.panel_id).first()
                    local_id = panel.local_id if panel else "N/A"
                    lines.append(f"Defect found: Image={img.filename}, Panel={local_id}")
                    lines.append(f"  Raw defect_type in DB: {r.defect_type}")
                    lines.append(f"  Parsed defects: {defects_in_json}")
                    
        lines.append(f"Total AiResults for this batch: {all_results_count}")
        lines.append(f"Total faulty/defect AiResults found by check: {defect_count}")
        
        with open("scratch/defect_details.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    finally:
        db.close()

if __name__ == '__main__':
    main()
