# app/main.py
import os
import cv2
import shutil
import uuid
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from sqlalchemy.orm import Session
from sqlalchemy import text

# Import DB
from app.core.database import get_db, engine
from app.models import models

# Import Services
from app.services.file_handler import FileService
from app.services.image_processor import ImageProcessor
from app.services.registration import RegistrationService
from app.services.ai_engine import AIEngine, draw_custom_annotation
from app.services.analyzer import SolarAnalyzer
from app.services.panel_geometry import assign_row_col_ids
from app.services.defect_logic import assign_defects_to_panels
from app.services.report_generator import ReportGenerator

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("solar_ai")

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
ai_engine: Optional[AIEngine] = AIEngine(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else None


@app.get("/")
def welcome():
    return {"status": "Online", "message": "Backend Solar AI đã sẵn sàng!"}


# ================================
# --- AUTHENTICATION ---
# ================================
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/v1/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == req.username).first()
    if not user or user.password_hash != req.password:
        return {"error": "Tài khoản hoặc mật khẩu không chính xác!"}
    
    return {
        "message": "Đăng nhập thành công",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        }
    }


# ================================
# --- GET LATEST DATA ---
# ================================
@app.get("/api/v1/latest-batch")
async def get_latest_batch(db: Session = Depends(get_db)):
    latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
    if not latest_batch:
        return {"data": [], "batch_id": None}
    
    images = db.query(models.Image).filter(models.Image.batch_id == latest_batch.id).all()
    
    final_report = []
    for img in images:
        ai_results = db.query(models.AiResult).filter(models.AiResult.image_id == img.id).all()
        
        panels_data = []
        for r in ai_results:
            panel_info = db.query(models.Panel).filter(models.Panel.id == r.panel_id).first()
            
            defects_list = []
            if r.defect_type and r.defect_type != "Healthy":
                defect_types = r.defect_type.split(", ")
                for dt in defect_types:
                    defects_list.append({"type": dt, "class_name": dt})
            
            panels_data.append({
                "local_id": panel_info.local_id,
                "x": panel_info.x_coord,
                "y": panel_info.y_coord,
                "total_panel_loss": r.loss_pct,
                "total_defect_area_ratio_percent": r.loss_pct,
                "confidence": r.confidence,
                "defects": defects_list,
                "status": "faulty" if defects_list else "healthy",
            })
            
        final_report.append({
            "filename": img.filename,
            "rgb_image": img.filename.replace("_thermal", ""),
            "total_panels": len(panels_data),
            "panels": panels_data
        })
        
    return {
        "batch_id": latest_batch.id,
        "data": final_report
    }


# ================================
# --- KHỐI 1: UPLOAD DỮ LIỆU ---
# ================================
from typing import List, Optional

@app.post("/api/v1/upload-drone-data")
async def upload_drone_data(files: List[UploadFile] = File(...)):
    upload_dir = "data/raw"
    extracted_files = FileService.process_uploads(files, upload_dir)
    return {"message": "Đã nhận và xử lý thành công!", "total_files": len(extracted_files)}


# ================================
# --- KHỐI 2: TIỀN HIỆU CHỈNH ẢNH NHIỆT (Pre-Calibration) ---
# ================================
@app.get("/api/v1/process-thermal")
async def process_images():
    """
    Tiền xử lý ảnh thermal từ data/raw → data/precalib.

    Pipeline hiện tại: BPR + Bilateral filter + black-border restore.
    QUAN TRỌNG: pipeline này phải GIỐNG với preprocessing dùng khi train model.
    Nếu model được train trên ảnh raw (không qua BPR/Bilateral), hãy tắt
    bằng cách set BYPASS_PREPROCESSING=True bên dưới.
    """
    # Option: bypass preprocessing mạnh, chỉ copy raw sang precalib
    # Đặt True nếu model train trên ảnh raw (không qua BPR/Bilateral)
    BYPASS_PREPROCESSING = False

    raw_dir = "data/raw"
    output_dir = "data/precalib"
    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    for filename in os.listdir(raw_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src_path = os.path.join(raw_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        if BYPASS_PREPROCESSING:
            # Không xử lý, chỉ copy để đảm bảo ảnh inference = ảnh raw
            shutil.copy2(src_path, dst_path)
        else:
            result = ImageProcessor.preprocess_thermal(src_path)
            if result is not None:
                cv2.imwrite(dst_path, result)
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
# --- KHỐI 4-5: CHẠY AI + PHÂN TÍCH + LƯU DB ---
# ================================
@app.post("/api/v1/analyze-all")
async def start_analysis(user_id: int = Form(None), db: Session = Depends(get_db)):
    """
    Pipeline phân tích hoàn chỉnh:
    1. Chạy YOLOv8-seg trên ảnh precalib
    2. Tách panel / defect
    3. Refine panel polygon (minAreaRect)
    4. Gán defect vào panel bằng overlap area
    5. Gán hàng/cột (R01_C03)
    6. Tính area_ratio, severity, recommendation
    7. Định vị lỗi trong panel (upper-left, ...)
    8. Lưu DB và trả kết quả
    """
    if ai_engine is None:
        return {"error": "Chưa tìm thấy file weights/best.pt"}

    raw_dir = "data/raw"
    precalib_dir = "data/precalib"
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(precalib_dir) or len(os.listdir(precalib_dir)) == 0:
        return {"error": "Thư mục precalib trống. Hãy chạy tiền xử lý trước!"}

    # Tạo Batch mới trong DB
    new_batch = models.UploadBatch(name="Đợt kiểm tra tự động", user_id=user_id)
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    final_report = []

    # Ghép cặp Thermal và RGB
    pairs = RegistrationService.match_thermal_rgb(raw_dir)
    thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}
    rgb_images = set([p['rgb'] for p in pairs])

    logger.info(f"[analyze-all] Model classes: {ai_engine.model.names}")
    logger.info(f"[analyze-all] Bắt đầu xử lý thư mục: {precalib_dir}")

    for filename in os.listdir(precalib_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if filename in rgb_images:
            continue  # Bỏ qua ảnh RGB, chỉ chạy AI trên thermal

        img_path = os.path.join(precalib_dir, filename)
        logger.info(f"[analyze-all] Inference: {img_path}")

        # ── BƯỚC 1: Chạy YOLOv8-seg ──
        raw_detections, yolo_result = ai_engine.detect_and_segment(img_path)
        img_h, img_w = yolo_result.orig_shape

        # Tách panel và defect
        panels_raw = [d for d in raw_detections if d["category"] == "panel"]
        defects_raw = [d for d in raw_detections if d["category"] == "defect"]

        logger.info(
            f"  → YOLO raw: {len(panels_raw)} panel, {len(defects_raw)} defect | "
            f"Ảnh: {img_w}x{img_h}"
        )

        # ── BƯỚC 2: Gán UUID cho panel ──
        for p in panels_raw:
            p["id"] = str(uuid.uuid4())

        # ── BƯỚC 3: Gán defect vào panel bằng overlap area ──
        panels_with_defects, unassigned = assign_defects_to_panels(panels_raw, defects_raw)

        logger.info(
            f"  → Defect assigned: {sum(len(p['defects']) for p in panels_with_defects)}, "
            f"unassigned: {len(unassigned)}"
        )

        # ── BƯỚC 4: Gán hàng/cột (R01_C03) ──
        final_panels = assign_row_col_ids(panels_with_defects)

        n_faulty = sum(1 for p in final_panels if p["status"] == "faulty")
        n_healthy = sum(1 for p in final_panels if p["status"] == "healthy")
        logger.info(f"  → Panel faulty: {n_faulty}, healthy: {n_healthy}")

        # ── BƯỚC 5 (debug): Lưu ảnh annotated ──
        try:
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                # Vẽ result.plot() cho debug YOLO raw
                annotated_raw = yolo_result.plot()
                cv2.imwrite(os.path.join(results_dir, "raw_" + filename), annotated_raw)

                # Vẽ custom annotation với refined polygon
                annotated_custom = draw_custom_annotation(orig_img, final_panels)
                cv2.imwrite(os.path.join(results_dir, filename), annotated_custom)
        except Exception as e:
            logger.warning(f"  → Không thể lưu ảnh annotated: {e}")

        # ── BƯỚC 6: Lưu Image vào DB ──
        db_image = models.Image(
            batch_id=new_batch.id,
            filename=filename,
            image_type="Thermal",
            path=os.path.join(results_dir, filename).replace("\\", "/")
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        # ── BƯỚC 7: Lưu panel và AI Result vào DB ──
        for p in final_panels:
            local_id = p.get("local_id", f"R00_C00")
            center = p.get("center", [0, 0])

            db_panel = db.query(models.Panel).filter(models.Panel.local_id == local_id).first()
            if not db_panel:
                db_panel = models.Panel(
                    local_id=local_id,
                    x_coord=center[0],
                    y_coord=center[1],
                )
                db.add(db_panel)
                db.commit()
                db.refresh(db_panel)
            else:
                db_panel.x_coord = center[0]
                db_panel.y_coord = center[1]
                db.commit()

            # Defect types string
            defect_str = ", ".join(list(set([d["class_name"] for d in p["defects"]]))) if p["defects"] else "Healthy"

            db_ai_result = models.AiResult(
                image_id=db_image.id,
                panel_id=db_panel.id,
                defect_type=defect_str,
                loss_pct=p.get("total_defect_area_ratio_percent", 0.0),
                confidence=p.get("confidence", 0.0),
            )
            db.add(db_ai_result)

        final_report.append({
            "filename": filename,
            "rgb_image": thermal_to_rgb.get(filename),
            "image_width": img_w,
            "image_height": img_h,
            "total_panels": len(final_panels),
            "panels": _serialize_panels(final_panels),
        })

    db.commit()

    # ── Auto-generate report PDF ──
    ai_results = db.query(models.AiResult).join(models.Image).filter(
        models.Image.batch_id == new_batch.id
    ).all()
    if ai_results:
        report_name = f"Report_Batch_{new_batch.id}.pdf"
        report_path = os.path.join("data", report_name)
        ReportGenerator.generate_inspection_report(new_batch.id, ai_results, report_path)
        db_report = models.Report(batch_id=new_batch.id, file_path=report_path)
        db.add(db_report)
        db.commit()

    logger.info(f"[analyze-all] Hoàn tất! Batch ID: {new_batch.id}, {len(final_report)} ảnh.")

    return {
        "message": "AI đã phân tích và lưu dữ liệu thành công!",
        "batch_id": new_batch.id,
        "data": final_report,
    }


def _serialize_panels(panels: List) -> List:
    """
    Serialize panel list cho JSON response.
    Đảm bảo numpy arrays được convert về Python native types.
    """
    result = []
    for p in panels:
        panel_out = {
            "id":           p.get("id", ""),
            "local_id":     p.get("local_id", ""),
            "row":          p.get("row", 0),
            "col":          p.get("col", 0),
            "class_name":   p.get("class_name", "panel"),
            "confidence":   p.get("confidence", 0.0),
            "bbox":         _to_list(p.get("bbox") or p.get("box", [])),
            "box":          _to_list(p.get("bbox") or p.get("box", [])),  # backward compat
            "polygon":      _to_list(p.get("polygon", [])),
            "area":         p.get("area", 0.0),
            "center":       _to_list(p.get("center", [0, 0])),
            "status":       p.get("status", "healthy"),
            "defects":      _serialize_defects(p.get("defects", [])),
            "total_defect_area_ratio_percent": p.get("total_defect_area_ratio_percent", 0.0),
            "max_defect_area_ratio_percent":   p.get("max_defect_area_ratio_percent", 0.0),
            "worst_severity":   p.get("worst_severity", "healthy"),
            "recommendation":   p.get("recommendation", "No action"),
            "main_defect_class": p.get("main_defect_class"),
            # Backward compat cho frontend cũ
            "total_panel_loss": p.get("total_defect_area_ratio_percent", 0.0),
            "x": p.get("center", [0, 0])[0],
            "y": p.get("center", [0, 0])[1],
        }
        result.append(panel_out)
    return result


def _serialize_defects(defects: List) -> List:
    result = []
    for d in defects:
        result.append({
            "class_name":        d.get("class_name", ""),
            "confidence":        d.get("confidence", 0.0),
            "bbox":              _to_list(d.get("bbox") or d.get("box", [])),
            "box":               _to_list(d.get("bbox") or d.get("box", [])),
            "polygon":           _to_list(d.get("polygon", [])),
            "area":              d.get("area", 0.0),
            "center":            _to_list(d.get("center", [0, 0])),
            "area_ratio_percent": d.get("area_ratio_percent", 0.0),
            "overlap_ratio":     d.get("overlap_ratio", 0.0),
            "area_inside_panel": d.get("area_inside_panel", 0.0),
            "relative_position": d.get("relative_position", {"u": 0, "v": 0}),
            "location_in_panel": d.get("location_in_panel", ""),
            "severity":          d.get("severity", ""),
            "recommendation":    d.get("recommendation", ""),
            # Backward compat
            "type": d.get("class_name", ""),
            "loss": d.get("area_ratio_percent", 0.0),
        })
    return result


def _to_list(val):
    """Convert numpy arrays / nested numpy về Python list."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_list(v) for v in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ================================
# --- KHỐI 6: CẬP NHẬT TRỌNG SỐ AI ---
# ================================
@app.post("/api/v1/update-ai-model")
async def update_ai_model(file: UploadFile = File(...)):
    global ai_engine

    if not file.filename.endswith(".pt"):
        return {"error": "Chỉ chấp nhận file trọng số định dạng .pt"}

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    file_path = os.path.join(weights_dir, "best.pt")

    # Backup file cũ nếu có
    if os.path.exists(file_path):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(weights_dir, f"best_backup_{timestamp}.pt")
        shutil.move(file_path, backup_path)

    # Lưu file trọng số mới
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load / reload model
    try:
        if ai_engine is None:
            # Lần đầu tiên có weights
            ai_engine = AIEngine(file_path)
        else:
            # Reload với path mới
            ai_engine.reload_model(new_path=file_path)

        logger.info(f"[update-model] Đã load weights mới: {file_path}")
        return {"message": f"Thành công! Trọng số từ '{file.filename}' đã được load làm model chính."}
    except Exception as e:
        return {"error": f"Lỗi khi load model: {str(e)}"}


# ================================
# --- KHỐI 7: LẤY ẢNH TỪNG TẤM PIN ---
# ================================
@app.get("/api/v1/panel-image")
async def get_panel_image(filename: str, x1: float, y1: float, x2: float, y2: float, polygon: str = None):
    # Dùng ảnh precalib — cùng loại với ảnh dùng cho AI inference
    img_path = os.path.join("data/precalib", filename)
    if not os.path.exists(img_path):
        return {"error": "Image not found"}
        
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Could not read image"}
        
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = img.shape[:2]
    
    crop_w = int(w * 0.5)
    crop_h = int(h * 0.5)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
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
        
    cropped_img = img[cy1:cy2, cx1:cx2].copy()
    
    box_x1 = max(0, x1 - cx1)
    box_y1 = max(0, y1 - cy1)
    box_x2 = min(cx2 - cx1, x2 - cx1)
    box_y2 = min(cy2 - cy1, y2 - cy1)
    
    if polygon:
        try:
            pts = [float(v) for v in polygon.split(",")]
            poly_pts = np.array(pts, dtype=np.float32).reshape((-1, 2))
            epsilon = 0.02 * cv2.arcLength(poly_pts, True)
            approx = cv2.approxPolyDP(poly_pts, epsilon, True)
            approx = approx.reshape(-1, 2)
            approx[:, 0] -= cx1
            approx[:, 1] -= cy1
            cv2.polylines(cropped_img, [np.int32(approx)], isClosed=True, color=(0, 255, 0), thickness=2)
        except Exception:
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
    ai_results = db.query(models.AiResult).join(models.Image).filter(
        models.Image.batch_id == batch_id
    ).all()
    
    if not ai_results:
        return {"error": "Không tìm thấy dữ liệu báo cáo"}

    report_name = f"Report_Batch_{batch_id}.pdf"
    report_path = os.path.join("data", report_name)
    
    ReportGenerator.generate_inspection_report(batch_id, ai_results, report_path)

    existing_report = db.query(models.Report).filter(models.Report.batch_id == batch_id).first()
    if not existing_report:
        db_report = models.Report(batch_id=batch_id, file_path=report_path)
        db.add(db_report)
        db.commit()

    return FileResponse(path=report_path, filename=report_name, media_type='application/pdf')


# ================================
# --- RESET TOÀN BỘ HỆ THỐNG ---
# ================================
@app.post("/api/v1/reset-system")
async def reset_system(db: Session = Depends(get_db)):
    folders = ["data/raw", "data/precalib", "data/results", "data/panels"]
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith(".pdf"):
                try:
                    os.remove(os.path.join("data", f))
                except:
                    pass

    old_processed = "data/processed"
    if os.path.exists(old_processed):
        shutil.rmtree(old_processed)

    try:
        db.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        db.execute(text("DELETE FROM reports;"))
        db.execute(text("DELETE FROM ai_results;"))
        db.execute(text("DELETE FROM images;"))
        db.execute(text("DELETE FROM panels;"))
        db.execute(text("DELETE FROM upload_batches;"))
        db.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        db.commit()
    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    return {"message": "Hệ thống đã được làm mới hoàn toàn!"}