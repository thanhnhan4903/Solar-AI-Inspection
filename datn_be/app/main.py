# app/main.py
import os
import cv2
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from sqlalchemy.orm import Session
from sqlalchemy import text
from shapely.geometry import Polygon

# Import DB
from app.core.database import get_db, engine
from app.models import models

# Import Services
from app.services.file_handler import FileService
from app.services.image_processor import ImageProcessor
from app.services.registration import RegistrationService
from app.services.ai_engine import AIEngine
from app.services.analyzer import SolarAnalyzer
from app.services.report_generator import ReportGenerator

# Khởi tạo bảng Database
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Solar Inspection API")

# ================================
# ✅ 1. CẤU HÌNH STATIC FILES
# ================================
app.mount("/data", StaticFiles(directory="data"), name="data")

# ================================
# ✅ 2. CẤU HÌNH CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# ✅ 3. LOAD AI MODEL (best.pt)
# ================================
WEIGHTS_PATH = "weights/best.pt"
ai_engine = AIEngine(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else None


@app.get("/")
def welcome():
    return {"status": "Online", "message": "Backend Solar AI đã sẵn sàng!"}


# ================================
# --- KHỐI 1: UPLOAD DỮ LIỆU ---
# ================================
@app.post("/api/v1/upload-drone-data")
async def upload_zip(file: UploadFile = File(...)):
    upload_dir = "data/raw"
    files = FileService.save_and_extract_zip(file, upload_dir)
    return {"message": "Đã nhận và giải nén thành công!", "total_files": len(files)}


# ================================
# --- KHỐI 2: TIỀN HIỆU CHỈNH ẢNH NHIỆT (Pre-Calibration) ---
# ================================
@app.get("/api/v1/process-thermal")
async def process_images():
    raw_dir = "data/raw"
    output_dir = "data/precalib"
    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    for filename in os.listdir(raw_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            result = ImageProcessor.preprocess_thermal(os.path.join(raw_dir, filename))
            if result is not None:
                cv2.imwrite(os.path.join(output_dir, filename), result)
                processed_count += 1

    return {"message": f"Đã tiền hiệu chỉnh xong {processed_count} ảnh!"}


# ================================
# --- KHỐI 3: GHÉP CẶP ẢNH (RGB-THERMAL) ---
# ================================
@app.get("/api/v1/match-pairs")
async def match_images():
    raw_dir = "data/raw"
    pairs = RegistrationService.match_thermal_rgb(raw_dir)
    return {"total_pairs": len(pairs), "pairs": pairs}


# ================================
# --- KHỐI 4-5: CHẠY AI (best.pt) + PHÂN TÍCH + LƯU DB ---
# Pipeline: Chạy YOLO trực tiếp trên ảnh RAW gốc
#           → Lưu ảnh annotated vào data/results
#           → Trích xuất tọa độ, lỗi, tính toán
# ================================
@app.post("/api/v1/analyze-all")
async def start_analysis(db: Session = Depends(get_db)):
    if ai_engine is None:
        return {"error": "Chưa tìm thấy file weights/best.pt"}

    raw_dir = "data/raw"
    precalib_dir = "data/precalib"
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(precalib_dir) or len(os.listdir(precalib_dir)) == 0:
        return {"error": "Thư mục precalib trống. Hãy chạy tiền xử lý trước!"}

    # Tạo Batch mới trong DB
    new_batch = models.InspectionBatch(name="Đợt kiểm tra tự động")
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    final_report = []

    # Ghép cặp Thermal và RGB
    pairs = RegistrationService.match_thermal_rgb(raw_dir)
    thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}
    rgb_images = set([p['rgb'] for p in pairs])

    for filename in os.listdir(precalib_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Bỏ qua ảnh RGB vì AI chỉ chạy trên ảnh Nhiệt
        if filename in rgb_images:
            continue

        img_path = os.path.join(precalib_dir, filename)

        # ========================================
        # BƯỚC 1: Chạy AI trên ảnh Precalib
        # ========================================
        raw_detections, yolo_result = ai_engine.detect_and_segment(img_path)
        img_h, img_w = yolo_result.orig_shape

        # ========================================
        # BƯỚC 2: Lưu ảnh annotated (YOLO vẽ sẵn) vào data/results
        # ========================================
        annotated_img = yolo_result.plot()
        cv2.imwrite(os.path.join(results_dir, filename), annotated_img)

        # ========================================
        # BƯỚC 3: Trích xuất & phân tích panel + lỗi
        # ========================================
        panels = [d for d in raw_detections if d['class_name'].lower() == 'panel']
        defects = [d for d in raw_detections if d['class_name'].lower() != 'panel']

        processed_panels = []

        for p in panels:
            p_poly = p.get('polygon')
            p_box = p.get('box')
            p_conf = p.get('confidence', 0)

            if not p_poly or len(p_poly) < 3: continue

            try:
                p_geom = Polygon(p_poly)
                if not p_geom.is_valid: continue
            except: continue

            p_loss = 0
            p_defects_list = []

            for d in defects:
                d_poly = d.get('polygon')
                d_box = d.get('box')
                if not d_poly or len(d_poly) < 3: continue
                try:
                    d_geom = Polygon(d_poly)
                    if not d_geom.is_valid: continue
                except: continue

                if p_geom.contains(d_geom.centroid):
                    loss, _ = SolarAnalyzer.calculate_metrics(p_poly, d_poly, d['class_name'])
                    p_loss += loss
                    p_defects_list.append({
                        "type": d['class_name'],
                        "loss": loss,
                        "box": d_box
                    })

            processed_panels.append({
                "x": round(p_geom.centroid.x, 2),
                "y": round(p_geom.centroid.y, 2),
                "box": p_box,
                "polygon": p_poly,
                "confidence": round(p_conf, 2),
                "defects": p_defects_list,
                "total_panel_loss": min(p_loss, 100.0)
            })

        # Gán Grid ID (Panel_01, Panel_02...) - trái→phải, trên→dưới
        final_panels = SolarAnalyzer.assign_grid_ids(processed_panels)

        # Lưu dữ liệu từng tấm pin vào MySQL
        for p in final_panels:
            db_panel = models.PanelAnalysis(
                batch_id=new_batch.id,
                filename=filename,
                x_coord=p['x'],
                y_coord=p['y'],
                local_id=p['local_id'],
                defect_type=", ".join(list(set([d['type'] for d in p['defects']]))) if p['defects'] else "Healthy",
                loss_pct=p['total_panel_loss'],
                confidence=p.get('confidence', 0.9)
            )
            db.add(db_panel)

        final_report.append({
            "filename": filename,
            "rgb_image": thermal_to_rgb.get(filename),
            "image_width": img_w,
            "image_height": img_h,
            "total_panels": len(final_panels),
            "panels": final_panels
        })

    db.commit()

    return {
        "message": "AI đã phân tích và lưu dữ liệu thành công!",
        "batch_id": new_batch.id,
        "data": final_report
    }


# ================================
# --- KHỐI 6: CẬP NHẬT TRỌNG SỐ AI ---
# ================================
@app.post("/api/v1/update-ai-model")
async def update_ai_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pt"):
        return {"error": "Chỉ chấp nhận file trọng số định dạng .pt"}
    
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    file_path = os.path.join(weights_dir, "best.pt")

    # Lưu đè file trọng số mới
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Tải lại model vào RAM
    try:
        ai_engine.reload_model()
        return {"message": "Cập nhật trọng số AI thành công và đã load vào hệ thống!"}
    except Exception as e:
        return {"error": f"Lỗi khi load model: {str(e)}"}

# ================================
# --- KHỐI 7: LẤY ẢNH TỪNG TẤM PIN ---
# ================================
@app.get("/api/v1/panel-image")
async def get_panel_image(filename: str, x1: float, y1: float, x2: float, y2: float, polygon: str = None):
    # Dùng ảnh precalib để vẽ box mới (không lấy ảnh kết quả có sẵn)
    img_path = os.path.join("data/precalib", filename)
    if not os.path.exists(img_path):
        return {"error": "Image not found"}
        
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Could not read image"}
        
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = img.shape[:2]
    
    # Zoom cỡ 50% tấm ảnh gốc (kích thước khung cắt bằng 50% kích thước ảnh)
    crop_w = int(w * 0.5)
    crop_h = int(h * 0.5)
    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Tính tọa độ khung cắt
    cx1 = max(0, cx - crop_w // 2)
    cy1 = max(0, cy - crop_h // 2)
    cx2 = min(w, cx + crop_w // 2)
    cy2 = min(h, cy + crop_h // 2)
    
    if cx2 - cx1 < crop_w:
        if cx1 == 0: cx2 = min(w, cx1 + crop_w)
        if cx2 == w: cx1 = max(0, cx2 - crop_w)
    if cy2 - cy1 < crop_h:
        if cy1 == 0: cy2 = min(h, cy1 + crop_h)
        if cy2 == h: cy1 = max(0, cy2 - crop_h)
        
    # Cắt bằng Numpy
    cropped_img = img[cy1:cy2, cx1:cx2].copy()
    
    # Tọa độ bounding box tương đối với ảnh đã cắt
    box_x1 = max(0, x1 - cx1)
    box_y1 = max(0, y1 - cy1)
    box_x2 = min(cx2 - cx1, x2 - cx1)
    box_y2 = min(cy2 - cy1, y2 - cy1)
    
    # Vẽ khung màu xanh lá
    if polygon:
        try:
            # polygon string format: "x1,y1,x2,y2,x3,y3..."
            pts = [float(v) for v in polygon.split(",")]
            poly_pts = np.array(pts, dtype=np.float32).reshape((-1, 2))
            
            # Nắn thẳng đa giác (giảm số điểm) để tạo viền thẳng nhưng giữ được phối cảnh
            epsilon = 0.02 * cv2.arcLength(poly_pts, True)
            approx = cv2.approxPolyDP(poly_pts, epsilon, True)
            approx = approx.reshape(-1, 2)
            
            # offset theo khung cắt
            approx[:, 0] -= cx1
            approx[:, 1] -= cy1
            cv2.polylines(cropped_img, [np.int32(approx)], isClosed=True, color=(0, 255, 0), thickness=2)
        except Exception:
            # fallback
            cv2.rectangle(cropped_img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
    else:
        cv2.rectangle(cropped_img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', cropped_img)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# ================================
# --- KHỐI 8: GIS MOCK DATA ---
# ================================
@app.get("/api/v1/mock-gis")
def get_mock_gis():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "PNL-001", "status": "Hotspot", "loss": "15.5%"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [106.6297, 10.8231]
                }
            }
        ]
    }


# ================================
# --- DOWNLOAD BÁO CÁO PDF ---
# ================================
@app.get("/api/v1/download-report/{batch_id}")
async def download_report(batch_id: int, db: Session = Depends(get_db)):
    panels = db.query(models.PanelAnalysis).filter(models.PanelAnalysis.batch_id == batch_id).all()
    
    if not panels:
        return {"error": "Không tìm thấy dữ liệu báo cáo"}

    report_name = f"Report_Batch_{batch_id}.pdf"
    report_path = os.path.join("data", report_name)
    
    ReportGenerator.generate_inspection_report(batch_id, panels, report_path)

    return FileResponse(path=report_path, filename=report_name, media_type='application/pdf')


# ================================
# --- RESET TOÀN BỘ HỆ THỐNG ---
# ================================
@app.post("/api/v1/reset-system")
async def reset_system(db: Session = Depends(get_db)):
    # 1. Xóa sạch file trong các thư mục data
    folders = ["data/raw", "data/precalib", "data/results", "data/panels"] 
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # Xóa các file báo cáo PDF trong data/
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith(".pdf"):
                try:
                    os.remove(os.path.join("data", f))
                except:
                    pass

    # Xóa thư mục cũ nếu còn tồn tại
    old_processed = "data/processed"
    if os.path.exists(old_processed):
        shutil.rmtree(old_processed)

    # 2. Xóa sạch dữ liệu trong MySQL
    try:
        db.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        db.execute(text("DELETE FROM panel_analyses;"))
        db.execute(text("DELETE FROM inspection_batches;"))
        db.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        db.commit()
    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    return {"message": "Hệ thống đã được làm mới hoàn toàn!"}