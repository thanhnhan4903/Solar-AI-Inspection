# app/main.py
import os
import cv2
import shutil
import uuid
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, Form, BackgroundTasks
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
from app.services.ai_engine import AIEngine
from app.services.panel_processor import draw_custom_annotation
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

# Tự động di chuyển (migration) thêm cột mới vào bảng upload_batches nếu chưa có
with engine.connect() as conn:
    for col_name, col_type in [
        ("project_name", "VARCHAR(255)"),
        ("location", "VARCHAR(255)"),
        ("scan_time", "VARCHAR(255)"),
        ("operator", "VARCHAR(255)"),
        ("device", "VARCHAR(255)"),
        ("scope", "VARCHAR(255)"),
        ("panel_power", "FLOAT DEFAULT 600.0"),
    ]:
        try:
            conn.execute(text(f"ALTER TABLE upload_batches ADD COLUMN {col_name} {col_type}"))
            conn.commit()
            logger.info(f"Da them cot '{col_name}' vao bang upload_batches thanh cong.")
        except Exception:
            # Cột đã tồn tại, bỏ qua
            pass

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

# ── Biến theo dõi tiến trình AI (in-memory, đủ cho single-user) ──
progress_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "filename": "",
    "step": "",
    "done": False,
}


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
    
    # Lấy ảnh thuộc đợt mới nhất
    images = []
    if latest_batch:
        images = db.query(models.Image).filter(models.Image.batch_id == latest_batch.id).all()
    
    # Ghép cặp Thermal và RGB trên thư mục data/raw để có thông tin chính xác
    pairs = RegistrationService.match_thermal_rgb("data/raw")
    thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}
    
    final_report = []
    for img in images:
        ai_results = db.query(models.AiResult).filter(models.AiResult.image_id == img.id).all()
        
        panels_data = []
        for r in ai_results:
            panel_info = db.query(models.Panel).filter(models.Panel.id == r.panel_id).first()
            if not panel_info:
                continue
            
            panel_detail = {}
            defects_list = []
            if r.defect_type and r.defect_type != "Healthy":
                if r.defect_type.startswith("{"):
                    try:
                        import json
                        panel_detail = json.loads(r.defect_type)
                        defects_list = panel_detail.get("defects", [])
                    except Exception:
                        pass
                elif r.defect_type.startswith("["):
                    try:
                        import json
                        defects_list = json.loads(r.defect_type)
                    except Exception:
                        pass
            
            if not defects_list and r.defect_type and r.defect_type != "Healthy" and not r.defect_type.startswith("{") and not r.defect_type.startswith("["):
                defect_types = r.defect_type.split(", ")
                for dt in defect_types:
                    defects_list.append({"type": dt, "class_name": dt})

            # Parse row and col
            import re
            row_val, col_val = 0, 0
            if panel_info.local_id:
                m = re.match(r"R(\d+)_C(\d+)", panel_info.local_id)
                if m:
                    row_val = int(m.group(1))
                    col_val = int(m.group(2))

            panels_data.append({
                "local_id": panel_info.local_id,
                "row": panel_detail.get("row", row_val),
                "col": panel_detail.get("col", col_val),
                "x": panel_info.x_coord,
                "y": panel_info.y_coord,
                "center": [panel_info.x_coord, panel_info.y_coord],
                "bbox": panel_detail.get("bbox", []),
                "box": panel_detail.get("bbox", []),
                "polygon": panel_detail.get("polygon", []),
                "total_panel_loss": r.loss_pct,
                "total_defect_area_ratio_percent": r.loss_pct,
                "confidence": r.confidence,
                "defects": defects_list,
                "status": "faulty" if defects_list else "healthy",
                "worst_severity": panel_detail.get("worst_severity", "healthy"),
                "recommendation": panel_detail.get("recommendation", "Không cần xử lý"),
                "gps_lat": panel_detail.get("gps_lat"),
                "gps_lng": panel_detail.get("gps_lng"),
            })
            
        final_report.append({
            "id": img.filename, # frontend UnifiedDashboard dùng ID như filename
            "filename": img.filename,
            "rgb_image": thermal_to_rgb.get(img.filename, img.filename.replace("_thermal", "")),
            "total_panels": len(panels_data),
            "panels": panels_data,
            "upload_date": img.batch.upload_date.isoformat() if img.batch else None
        })
        
    return {
        "batch_id": latest_batch.id if latest_batch else None,
        "project_name": latest_batch.project_name if latest_batch else None,
        "location": latest_batch.location if latest_batch else None,
        "scan_time": latest_batch.scan_time if latest_batch else None,
        "operator": latest_batch.operator if latest_batch else None,
        "device": latest_batch.device if latest_batch else None,
        "scope": latest_batch.scope if latest_batch else None,
        "panel_power": latest_batch.panel_power if latest_batch else 600.0,
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

    Pipeline: BPR + Bilateral filter + black-border restore + quality assessment.
    Trả về quality_report cho từng ảnh để frontend hiển thị popup xác nhận trước AI.

    Quality tiers:
      ok      — 0 issues
      warning — 1-2 issues (vẫn có thể chạy AI)
      poor    — ≥3 issues  (khuyến nghị chụp lại)
    """
    BYPASS_PREPROCESSING = True  # Bypass tiền xử lý để YOLO nhận ảnh gốc như khi test trực tiếp

    raw_dir = "data/raw"
    output_dir = "data/precalib"
    os.makedirs(output_dir, exist_ok=True)

    quality_report = []

    for filename in os.listdir(raw_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src_path = os.path.join(raw_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        # Đọc ảnh để phân biệt ảnh nhiệt (Thermal) và ảnh quang học (RGB) bằng độ phân giải
        import cv2 as _cv2
        raw_img = _cv2.imread(src_path)
        if raw_img is None:
            continue
            
        h, w = raw_img.shape[:2]
        if w > 1280:
            # Đây là ảnh RGB (ảnh quang học song song), bỏ qua không chạy tiền xử lý/AI
            continue

        # Tránh xử lý lại các file đã có trong precalib
        if os.path.exists(dst_path):
            quality_report.append({
                "filename": filename,
                "quality_status": "ok",
                "issues": [],
                "issues_vi": [],
                "metrics": {},
                "preview_url": f"/data/precalib/{filename}",
            })
            continue

        if BYPASS_PREPROCESSING:
            shutil.copy2(src_path, dst_path)
            quality = ImageProcessor.assess_quality(raw_img) if raw_img is not None else {
                "quality_status": "error", "issues": [], "issues_vi": [], "metrics": {}
            }
        else:
            result, quality = ImageProcessor.preprocess_thermal(src_path, return_quality=True)
            if result is not None:
                cv2.imwrite(dst_path, result)
            else:
                quality = {"quality_status": "error", "issues": ["image_read_failed"], "issues_vi": ["Không đọc được ảnh"], "metrics": {}}

        quality_report.append({
            "filename": filename,
            "quality_status": quality.get("quality_status", "error"),
            "issues": quality.get("issues", []),
            "issues_vi": quality.get("issues_vi", []),
            "metrics": quality.get("metrics", {}),
            "preview_url": f"/data/precalib/{filename}",
        })

    # Tính overall_status: poor nếu có bất kỳ ảnh poor, warning nếu có warning, ok nếu tất cả ok
    statuses = [r["quality_status"] for r in quality_report]
    if "poor" in statuses:
        overall_status = "poor"
    elif "warning" in statuses:
        overall_status = "warning"
    elif "error" in statuses:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "message": f"Đã tiền hiệu chỉnh xong {len(quality_report)} ảnh!",
        "processed_count": len(quality_report),
        "overall_status": overall_status,
        "quality_report": quality_report,
    }


# ================================
# --- KHỐI 3: GHÉP CẶP ẢNH (RGB-THERMAL) ---
# ================================
@app.get("/api/v1/match-pairs")
async def match_images():
    raw_dir = "data/raw"
    pairs = RegistrationService.match_thermal_rgb(raw_dir)
    return {"total_pairs": len(pairs), "pairs": pairs}


# ================================
# --- POLL TIẾN TRÌNH AI ---
# ================================
@app.get("/api/v1/analyze-progress")
def get_analyze_progress():
    """Trả về trạng thái tiến trình phân tích AI hiện tại."""
    return dict(progress_state)


def get_gps_metadata(img_path):
    """
    Trích xuất tọa độ GPS Latitude, Longitude từ EXIF metadata của ảnh RGB.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        
        image = Image.open(img_path)
        exif = image._getexif()
        if not exif:
            return None
            
        gps_info = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
                    
        if not gps_info:
            return None
            
        def _to_degrees(value):
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
            
        lat_value = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef")
        lng_value = gps_info.get("GPSLongitude")
        lng_ref = gps_info.get("GPSLongitudeRef")
        
        if lat_value and lat_ref and lng_value and lng_ref:
            lat = _to_degrees(lat_value)
            if lat_ref != "N":
                lat = -lat
            lng = _to_degrees(lng_value)
            if lng_ref != "E":
                lng = -lng
            return lat, lng
    except Exception as e:
        logger.warning(f"Error reading GPS EXIF: {e}")
    return None


# ================================
# --- KHỐI 4-5: CHẠY AI + PHÂN TÍCH + LƯU DB ---
# ================================
@app.post("/api/v1/analyze-all")
def start_analysis(
    background_tasks: BackgroundTasks,
    user_id: int = Form(None), 
    project_name: str = Form(None),
    location: str = Form(None),
    scan_time: str = Form(None),
    operator: str = Form(None),
    device: str = Form(None),
    scope: str = Form(None),
    panel_power: float = Form(600.0),
    db: Session = Depends(get_db)
):
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

    # Ghép cặp Thermal và RGB
    pairs = RegistrationService.match_thermal_rgb(raw_dir)
    thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}
    rgb_images = set([p['rgb'] for p in pairs])

    # Danh sách file đã phân tích
    processed_images = set([img.filename for img in db.query(models.Image).all()])

    # Danh sách file thermal cần xử lý
    thermal_files = [
        fn for fn in os.listdir(precalib_dir)
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')) 
        and fn not in rgb_images
        and fn not in processed_images
    ]

    if not thermal_files:
        return {"message": "Không có ảnh mới nào cần phân tích!"}

    # ── Reset + bắt đầu progress tracker ──
    progress_state["running"] = True
    progress_state["done"] = False
    progress_state["current"] = 0
    progress_state["total"] = len(thermal_files)
    progress_state["filename"] = ""
    progress_state["step"] = "Khởi động..."

    # Tạo Batch mới trong DB
    new_batch = models.UploadBatch(
        name=project_name or "Đợt kiểm tra tự động", 
        user_id=user_id,
        project_name=project_name,
        location=location,
        scan_time=scan_time,
        operator=operator,
        device=device,
        scope=scope,
        panel_power=panel_power or 600.0
    )
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)

    final_report = []

    logger.info(f"[analyze-all] Model classes: {ai_engine.model.names}")
    logger.info(f"[analyze-all] Bắt đầu xử lý {len(thermal_files)} ảnh")

    for idx, filename in enumerate(thermal_files):
        # ── Cập nhật progress ──
        progress_state["current"] = idx + 1
        progress_state["filename"] = filename
        progress_state["step"] = f"YOLO inference..."

        img_path = os.path.join(precalib_dir, filename)
        logger.info(f"[analyze-all] Inference ({idx+1}/{len(thermal_files)}): {img_path}")

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
        panels_with_defects, unassigned = assign_defects_to_panels(panels_raw, defects_raw, panel_power, image_path=img_path)

        logger.info(
            f"  → Defect assigned: {sum(len(p['defects']) for p in panels_with_defects)}, "
            f"unassigned: {len(unassigned)}"
        )

        # ── BƯỚC 4: Gán hàng/cột (R01_C03) ──
        final_panels = assign_row_col_ids(panels_with_defects)

        # Lấy GPS từ EXIF của ảnh RGB ghép cặp
        lat_exif, lng_exif = None, None
        rgb_filename = thermal_to_rgb.get(filename)
        if rgb_filename:
            rgb_path = os.path.join(raw_dir, rgb_filename)
            gps_coords = get_gps_metadata(rgb_path)
            if gps_coords:
                lat_exif, lng_exif = gps_coords

        for p in final_panels:
            p["gps_lat"] = lat_exif
            p["gps_lng"] = lng_exif

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
        db.flush()

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
                db.flush()
            else:
                db_panel.x_coord = center[0]
                db_panel.y_coord = center[1]

            # Defect types string or JSON representing the panel data
            import json
            defect_str = json.dumps({
                "bbox": _to_list(p.get("bbox") or p.get("box", [])),
                "polygon": _to_list(p.get("polygon", [])),
                "defects": _serialize_defects(p.get("defects", [])),
                "row": p.get("row", 0),
                "col": p.get("col", 0),
                "status": p.get("status", "healthy"),
                "total_panel_loss": p.get("total_panel_loss", 0.0),
                "worst_severity": p.get("worst_severity", "healthy"),
                "recommendation": p.get("recommendation", "Không cần xử lý"),
                "confidence": p.get("confidence", 0.0),
                "gps_lat": lat_exif,
                "gps_lng": lng_exif,
            })

            db_ai_result = models.AiResult(
                image_id=db_image.id,
                panel_id=db_panel.id,
                defect_type=defect_str,
                loss_pct=p.get("total_panel_loss", 0.0),
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

    # ── Auto-generate report PDF (Chạy ngầm dưới nền) ──
    ai_results = db.query(models.AiResult).join(models.Image).filter(
        models.Image.batch_id == new_batch.id
    ).all()
    if ai_results:
        report_name = f"Report_Batch_{new_batch.id}.pdf"
        report_path = os.path.join("data", report_name)
        background_tasks.add_task(ReportGenerator.generate_inspection_report, new_batch.id, ai_results, report_path)
        db_report = models.Report(batch_id=new_batch.id, file_path=report_path)
        db.add(db_report)
        db.commit()

    logger.info(f"[analyze-all] Hoàn tất! Batch ID: {new_batch.id}, {len(final_report)} ảnh.")

    # ── Kết thúc progress tracker ──
    progress_state["running"] = False
    progress_state["done"] = True
    progress_state["step"] = "Hoàn tất!"

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
            # GPS EXIF
            "gps_lat":          p.get("gps_lat"),
            "gps_lng":          p.get("gps_lng"),
            # Backward compat cho frontend cũ
            "total_panel_loss": p.get("total_panel_loss", 0.0),
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
        img_path = os.path.join("data/raw", filename)
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
# --- PHÂN TÍCH LẠI (GIỮ ẢNH GỐC) ---
# ================================
@app.post("/api/v1/reanalyze")
async def reanalyze(db: Session = Depends(get_db)):
    folders = ["data/results", "data/panels"]
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

    return {"message": "Dữ liệu phân tích đã được xóa sạch, sẵn sàng chạy lại AI!"}


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


# ================================
# --- CẬP NHẬT THÔNG TIN DỰ ÁN ---
# ================================
class UpdateBatchMetadataRequest(BaseModel):
    batch_id: int
    project_name: Optional[str] = None
    location: Optional[str] = None
    scan_time: Optional[str] = None
    operator: Optional[str] = None
    device: Optional[str] = None
    scope: Optional[str] = None
    panel_power: Optional[float] = 600.0

@app.post("/api/v1/update-batch-metadata")
def update_batch_metadata(req: UpdateBatchMetadataRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    batch = db.query(models.UploadBatch).filter(models.UploadBatch.id == req.batch_id).first()
    if not batch:
        return {"error": "Không tìm thấy đợt kiểm tra này"}
    
    # Cập nhật thông tin dự án
    batch.project_name = req.project_name
    batch.location = req.location
    batch.scan_time = req.scan_time
    batch.operator = req.operator
    batch.device = req.device
    batch.scope = req.scope
    
    new_power = req.panel_power if req.panel_power is not None else 600.0
    batch.panel_power = new_power
    
    db.commit()
    
    # Tính toán lại công suất tổn thất nếu công suất tấm pin thay đổi
    import json
    ai_results = db.query(models.AiResult).join(models.Image).filter(models.Image.batch_id == batch.id).all()
    for r in ai_results:
        if r.defect_type and r.defect_type != "Healthy":
            try:
                panel_detail = json.loads(r.defect_type)
                
                # Chỉ tính lại hao hụt nếu panel này bị lỗi
                if panel_detail.get("status") == "healthy":
                    panel_detail["total_panel_loss"] = 0.0
                    panel_detail["power_loss_w"] = 0.0
                    r.loss_pct = 0.0
                    r.defect_type = json.dumps(panel_detail)
                    continue
                
                # Tính lại sản lượng hao hụt theo logic chia tấm pin làm 3 phần dọc
                bbox = panel_detail.get("bbox") or panel_detail.get("box", [0, 0, 0, 0])
                defects = panel_detail.get("defects", [])
                
                affected_parts = set()
                has_crack = False
                has_shading = False
                
                for d in defects:
                    cls_name = d.get("class_name", "")
                    if cls_name == "crack":
                        has_crack = True
                    elif cls_name == "shading":
                        has_shading = True
                    elif cls_name in ["hotspot_single_cell", "hotspot_multi_cell"]:
                        pos = d.get("relative_position", {})
                        v = pos.get("v", 0.5)
                        part_idx = min(2, int(v * 3))
                        affected_parts.add(part_idx)
                        
                if has_crack:
                    power_loss_w = new_power
                else:
                    power_loss_w = len(affected_parts) * (new_power / 3.0)
                
                # Cập nhật lại các trường trong panel_detail JSON
                panel_detail["total_panel_loss"] = round(power_loss_w, 2)
                panel_detail["power_loss_w"] = round(power_loss_w, 2)
                
                # Cập nhật lại db ai_result
                r.defect_type = json.dumps(panel_detail)
                r.loss_pct = round(power_loss_w, 2)
            except Exception as e:
                logger.warning(f"Lỗi tính lại power loss cho AiResult {r.id}: {e}")
                
    db.commit()
    
    # Sinh lại tệp báo cáo PDF (Chạy bất đồng bộ dưới nền)
    if ai_results:
        report_name = f"Report_Batch_{batch.id}.pdf"
        report_path = os.path.join("data", report_name)
        # Xóa báo cáo cũ
        if os.path.exists(report_path):
            try:
                os.remove(report_path)
            except Exception:
                pass
        
        # Đưa việc sinh báo cáo vào hàng đợi nền
        background_tasks.add_task(ReportGenerator.generate_inspection_report, batch.id, ai_results, report_path)
        
        db_report = db.query(models.Report).filter(models.Report.batch_id == batch.id).first()
        if not db_report:
            db_report = models.Report(batch_id=batch.id, file_path=report_path)
            db.add(db_report)
        else:
            db_report.file_path = report_path
        db.commit()
        
    return {
        "message": "Cập nhật thông tin dự án thành công!",
        "project_name": batch.project_name,
        "location": batch.location,
        "scan_time": batch.scan_time,
        "operator": batch.operator,
        "device": batch.device,
        "scope": batch.scope,
        "panel_power": batch.panel_power
    }