import sys
import os

# Force UTF-8 encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models import models

def main():
    db = SessionLocal()
    try:
        batches = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).all()
        print(f"Total upload batches: {len(batches)}")
        for b in batches:
            print(f"Batch ID={b.id}, Name={repr(b.name)}, Date={b.upload_date}, Status={b.status}")
    finally:
        db.close()

if __name__ == '__main__':
    main()
