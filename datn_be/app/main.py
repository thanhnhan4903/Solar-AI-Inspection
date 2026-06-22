# app/main.py
import os
import cv2
import shutil
import uuid
import logging
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, Form, BackgroundTasks, HTTPException
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
from app.services.pv_panel_snapper import snap_panels_with_v61
from app.services.defect_thermal_validator import (
    validate_defects_by_relative_thermal_contrast,
    THERMAL_VALIDATOR_ENABLED,
)

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
        ("scope", "TEXT"),
        ("panel_power", "FLOAT DEFAULT 600.0"),
    ]:
        try:
            conn.execute(text(f"ALTER TABLE upload_batches ADD COLUMN {col_name} {col_type}"))
            conn.commit()
            logger.info(f"Da them cot '{col_name}' vao bang upload_batches thanh cong.")
        except Exception:
            # Cột đã tồn tại, bỏ qua
            pass

    # Đảm bảo cột scope là kiểu TEXT trong MySQL
    try:
        conn.execute(text("ALTER TABLE upload_batches MODIFY COLUMN scope TEXT"))
        conn.commit()
        logger.info("Da thay doi kieu du lieu cot 'scope' sang TEXT thanh cong.")
    except Exception as e:
        logger.warning(f"Khong the modify column scope sang TEXT (co the dang la SQLite hoac da la TEXT): {e}")

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
    
    # Summary calculation variables
    total_panels = 0
    normal_panels = 0
    faulty_panels = 0
    total_defects = 0
    included_defects = 0
    excluded_defects = 0
    unreviewed = 0
    confirmed_defect = 0
    needs_review = 0
    false_positive = 0
    total_power_loss_w = 0.0

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

            raw_polygon = panel_detail.get("polygon", [])
            outer_polygon = panel_detail.get("outer_polygon") or raw_polygon
            inner_polygon = panel_detail.get("inner_polygon", [])
            calc_polygon = panel_detail.get("calc_polygon") or inner_polygon

            # Nếu outer_polygon rỗng hoặc không đủ 3 điểm thì fallback raw polygon.
            if not outer_polygon or len(outer_polygon) < 3:
                outer_polygon = raw_polygon

            geometry_source = panel_detail.get("geometry_source", "unknown")
            snap_source = panel_detail.get("snap_source", "")
            outer_area = panel_detail.get("outer_area", 0.0)
            inner_area = panel_detail.get("inner_area", panel_detail.get("area", 0.0))

            review_status = panel_detail.get("review_status", "unreviewed")
            if review_status == "false_positive":
                include_in_report = False
            elif review_status in ["confirmed_defect", "needs_review", "unreviewed"]:
                include_in_report = True
            else:
                include_in_report = panel_detail.get("include_in_report", True)

            review_label = {
                "unreviewed": "Chưa duyệt",
                "confirmed_defect": "Đã xác nhận",
                "needs_review": "Cần kiểm tra lại",
                "false_positive": "Bỏ qua"
            }.get(review_status, "Chưa duyệt")

            review_note = panel_detail.get("review_note", "")
            maintenance_priority = panel_detail.get("maintenance_priority", "medium")
            reviewer_name = panel_detail.get("reviewer_name", "")
            reviewed_at = panel_detail.get("reviewed_at", "")

            if review_status == "false_positive":
                status_val = "healthy"
            else:
                status_val = "faulty" if defects_list else "healthy"

            # Serialize defects
            serialized_defects = _serialize_defects(defects_list)

            # Update metrics
            total_panels += 1
            if status_val == "healthy":
                normal_panels += 1
            else:
                faulty_panels += 1

            n_defects = len(serialized_defects)
            total_defects += n_defects
            
            if include_in_report:
                included_defects += n_defects
                total_power_loss_w += r.loss_pct
            else:
                excluded_defects += n_defects

            if n_defects > 0:
                if review_status == "confirmed_defect":
                    confirmed_defect += 1
                elif review_status == "needs_review":
                    needs_review += 1
                elif review_status == "false_positive":
                    false_positive += 1
                else:
                    unreviewed += 1

            panels_data.append({
                "local_id": panel_info.local_id,
                "row": panel_detail.get("row", row_val),
                "col": panel_detail.get("col", col_val),
                "x": panel_info.x_coord,
                "y": panel_info.y_coord,
                "center": [panel_info.x_coord, panel_info.y_coord],
                "bbox": panel_detail.get("bbox", []),
                "box": panel_detail.get("bbox", []),
                "polygon": outer_polygon,
                "outer_polygon": outer_polygon,
                "inner_polygon": inner_polygon,
                "calc_polygon": calc_polygon,
                "geometry_source": geometry_source,
                "snap_source": snap_source,
                "outer_area": outer_area,
                "inner_area": inner_area,
                "total_panel_loss": r.loss_pct,
                "total_defect_area_ratio_percent": r.loss_pct,
                "confidence": r.confidence,
                "defects": serialized_defects,
                "status": status_val,
                "worst_severity": panel_detail.get("worst_severity", "healthy"),
                "recommendation": panel_detail.get("recommendation", "Không cần xử lý"),
                "gps_lat": panel_detail.get("gps_lat"),
                "gps_lng": panel_detail.get("gps_lng"),
                "main_defect_class": panel_detail.get("main_defect_class"),
                "review_status": review_status,
                "review_label": review_label,
                "review_note": review_note,
                "include_in_report": include_in_report,
                "maintenance_priority": maintenance_priority,
                "reviewer_name": reviewer_name,
                "reviewed_at": reviewed_at,
            })
            
        # Get image dimensions from file on disk, fallback to 640x512
        img_w, img_h = 640, 512
        try:
            from PIL import Image as PILImage
            precalib_path = os.path.join("data", "precalib", img.filename)
            if os.path.exists(precalib_path):
                with PILImage.open(precalib_path) as pil_img:
                    img_w, img_h = pil_img.size
            elif os.path.exists(img.path):
                with PILImage.open(img.path) as pil_img:
                    img_w, img_h = pil_img.size
        except Exception as e:
            logger.warning(f"Could not get image dimensions for {img.filename}: {e}")

        final_report.append({
            "id": img.filename, # frontend UnifiedDashboard dùng ID như filename
            "filename": img.filename,
            "rgb_image": thermal_to_rgb.get(img.filename, img.filename.replace("_thermal", "")),
            "image_width": img_w,
            "image_height": img_h,
            "total_panels": len(panels_data),
            "panels": panels_data,
            "upload_date": img.batch.upload_date.isoformat() if img.batch else None
        })
        
    scope_val = latest_batch.scope if latest_batch else ""
    system_capacity = ""
    supervisor = ""
    notes = ""
    data_type = ""
    ai_model = ""
    system_version = ""
    if scope_val and scope_val.startswith("{"):
        try:
            import json
            scope_data = json.loads(scope_val)
            scope_val = scope_data.get("s", "")
            system_capacity = scope_data.get("sc", "")
            supervisor = scope_data.get("sv", "")
            notes = scope_data.get("nt", "")
            data_type = scope_data.get("dt", "")
            ai_model = scope_data.get("am", "")
            system_version = scope_data.get("sys", "")
        except Exception:
            pass

    normal_panel_ratio_percent = (normal_panels / total_panels * 100.0) if total_panels > 0 else 100.0

    return {
        "batch_id": latest_batch.id if latest_batch else None,
        "project_name": latest_batch.project_name if latest_batch else None,
        "location": latest_batch.location if latest_batch else None,
        "scan_time": latest_batch.scan_time if latest_batch else None,
        "operator": latest_batch.operator if latest_batch else None,
        "device": latest_batch.device if latest_batch else None,
        "scope": scope_val,
        "panel_power": latest_batch.panel_power if latest_batch else 600.0,
        "system_capacity": system_capacity,
        "supervisor": supervisor,
        "notes": notes,
        "data_type": data_type,
        "ai_model": ai_model,
        "system_version": system_version,
        "summary": {
            "total_images": len(final_report),
            "total_panels": total_panels,
            "normal_panels": normal_panels,
            "faulty_panels": faulty_panels,
            "total_defects": total_defects,
            "included_defects": included_defects,
            "excluded_defects": excluded_defects,
            "unreviewed": unreviewed,
            "confirmed_defect": confirmed_defect,
            "needs_review": needs_review,
            "false_positive": false_positive,
            "total_power_loss_w": round(total_power_loss_w, 2),
            "normal_panel_ratio_percent": round(normal_panel_ratio_percent, 2)
        },
        "data": final_report
    }


# ================================
# --- KHỐI 1: UPLOAD DỮ LIỆU ---
# ================================

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
        if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
            logger.warning(f"Bo qua file anh bi trong hoac khong ton tai: {src_path}")
            continue

        try:
            raw_img = _cv2.imread(src_path)
        except Exception as e:
            logger.error(f"Loi khi doc file anh {src_path}: {e}")
            continue

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
    running = progress_state.get("running", False)
    done = progress_state.get("done", False)
    step = progress_state.get("step", "")
    current = progress_state.get("current", 0)
    total = progress_state.get("total", 0)
    
    status = "idle"
    error_msg = None
    if running:
        status = "running"
    elif done:
        if step and step.startswith("Thất bại"):
            status = "error"
            error_msg = step
        else:
            status = "completed"
            
    percent = (current / total * 100) if total > 0 else 0
    if status == "completed":
        percent = 100
        
    return {
        "status": status,
        "running": running,
        "percent": percent,
        "processed_images": current,
        "total_images": total,
        "current_stage": step,
        "message": step,
        "eta_seconds": None,
        "filename": progress_state.get("filename", ""),
        "error": error_msg
    }


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
def run_analysis_pipeline_task(
    batch_id: int,
    thermal_files: List[str],
    processed_images: List[str],
    thermal_to_rgb: dict,
    raw_dir: str,
    precalib_dir: str,
    results_dir: str,
    panel_power: float
):
    """Background task to run the full analysis pipeline."""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        # Calculate total images and processed count for accurate progress reporting
        all_thermal_in_precalib = [
            fn for fn in os.listdir(precalib_dir)
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')) 
            and fn not in [p for p in thermal_to_rgb.values()]
        ]
        total_images = len(all_thermal_in_precalib)
        processed_count = total_images - len(thermal_files)

        progress_state["running"] = True
        progress_state["done"] = False
        progress_state["current"] = processed_count
        progress_state["total"] = total_images
        progress_state["filename"] = ""
        progress_state["step"] = "Khởi động..."

        # ── Re-associate existing images from previous runs with the new batch ──
        if processed_images:
            db.query(models.Image).filter(models.Image.filename.in_(processed_images)).update(
                {models.Image.batch_id: batch_id},
                synchronize_session=False
            )
            db.commit()

        final_report = []

        logger.info(f"[analyze-all background] Model classes: {ai_engine.model.names if ai_engine else 'None'}")
        logger.info(f"[analyze-all background] Bắt đầu xử lý {len(thermal_files)} ảnh")

        for idx, filename in enumerate(thermal_files):
            # ── Cập nhật progress ──
            progress_state["current"] = processed_count + idx + 1
            progress_state["filename"] = filename
            progress_state["step"] = f"YOLO inference..."

            img_path = os.path.join(precalib_dir, filename)
            logger.info(f"[analyze-all background] Inference ({idx+1}/{len(thermal_files)}): {img_path}")

            # ── BƯỚC 1: Chạy YOLOv8-seg ──
            raw_detections, yolo_result = ai_engine.detect_and_segment(img_path)
            img_h, img_w = yolo_result.orig_shape

            # Tách panel và defect từ YOLO raw output
            panels_yolo = [d for d in raw_detections if d["category"] == "panel"]
            defects_raw = [d for d in raw_detections if d["category"] == "defect"]

            logger.info(
                f"  → YOLO raw: {len(panels_yolo)} panel, {len(defects_raw)} defect | "
                f"Ảnh: {img_w}x{img_h}"
            )

            # ── BƯỚC 1b: Thay panel polygon bằng v62 line-snap ──
            try:
                panels_snapped = snap_panels_with_v61(
                    image_path=img_path,
                    yolo_panels=panels_yolo,
                    output_dir=results_dir,
                    debug=True,
                )
            except Exception as _snap_err:
                logger.exception(f"  → V62 snap failed: {_snap_err}. Falling back to YOLO panels.")
                panels_snapped = []

            if panels_snapped:
                panels_raw = panels_snapped
                logger.info(
                    f"  → V62 line-snap: {len(panels_raw)} panels (outer_polygon from line-snap)"
                )
            else:
                panels_raw = panels_yolo
                for p in panels_raw:
                    p["geometry_source"] = "yolo_fallback"
                logger.warning(
                    f"  → V62 snap returned 0 panels, falling back to {len(panels_raw)} YOLO panels."
                )

            # ── BƯỚC 2: Gán UUID cho panel ──
            for p in panels_raw:
                if not p.get("id"):
                    p["id"] = str(uuid.uuid4())

            # ── BƯỚC 3: Gán defect vào panel bằng overlap area ──
            panels_with_defects, unassigned = assign_defects_to_panels(panels_raw, defects_raw, panel_power, image_path=img_path)

            logger.info(
                f"  → Defect assigned: {sum(len(p['defects']) for p in panels_with_defects)}, "
                f"unassigned: {len(unassigned)}"
            )

            # ── BƯỚC 3b: Kiểm chứng lỗi bằng tương quan nhiệt tương đối ──
            if THERMAL_VALIDATOR_ENABLED:
                try:
                    stem = os.path.splitext(filename)[0]
                    panels_with_defects = validate_defects_by_relative_thermal_contrast(
                        image_path=img_path,
                        panels=panels_with_defects,
                        debug_dir="data/results/debug_logs",
                        image_stem=stem,
                    )
                    logger.info(f"  → Thermal validation: OK ({stem})")
                except Exception as _tv_err:
                    logger.warning(
                        f"  → Thermal validator failed: {_tv_err}. Tiếp tục không có validation."
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
                    annotated_raw = yolo_result.plot()
                    cv2.imwrite(os.path.join(results_dir, "raw_" + filename), annotated_raw)

                    annotated_custom = draw_custom_annotation(orig_img, final_panels)
                    cv2.imwrite(os.path.join(results_dir, filename), annotated_custom)
            except Exception as e:
                logger.warning(f"  → Không thể lưu ảnh annotated: {e}")

            # ── BƯỚC 6: Lưu Image vào DB ──
            db_image = models.Image(
                batch_id=batch_id,
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

                import json
                defect_str = json.dumps({
                    "bbox": _to_list(p.get("bbox") or p.get("box", [])),
                    "polygon": _to_list(p.get("polygon", [])),
                    "outer_polygon": _to_list(p.get("outer_polygon") or p.get("polygon", [])),
                    "inner_polygon": _to_list(p.get("inner_polygon", [])),
                    "calc_polygon": _to_list(p.get("calc_polygon") or p.get("inner_polygon", [])),
                    "geometry_source": p.get("geometry_source", "yolo_fallback"),
                    "snap_source": p.get("snap_source", ""),
                    "outer_area": p.get("outer_area", p.get("area", 0.0)),
                    "inner_area": p.get("inner_area", p.get("area", 0.0)),
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

        # ── Auto-generate report PDF ──
        ai_results = db.query(models.AiResult).join(models.Image).filter(
            models.Image.batch_id == batch_id
        ).all()
        if ai_results:
            report_name = f"Report_Batch_{batch_id}.pdf"
            report_path = os.path.join("data", report_name)
            ReportGenerator.generate_inspection_report(batch_id, ai_results, report_path)
            
            db_report = models.Report(batch_id=batch_id, file_path=report_path)
            db.add(db_report)
            db.commit()

        logger.info(f"[analyze-all background] Hoàn tất! Batch ID: {batch_id}, {len(final_report)} ảnh.")
        
        progress_state["running"] = False
        progress_state["done"] = True
        progress_state["step"] = "Hoàn tất!"

    except Exception as e:
        logger.exception(f"Error in run_analysis_pipeline_task: {e}")
        progress_state["running"] = False
        progress_state["done"] = True
        progress_state["step"] = f"Thất bại: {str(e)}"
    finally:
        db.close()


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
    global progress_state
    
    # ── Reset progress tracker immediately ──
    progress_state.update({
        "running": True,
        "done": False,
        "current": 0,
        "total": 0,
        "filename": "",
        "step": "Đang chuẩn bị..."
    })

    if ai_engine is None:
        progress_state.update({
            "running": False,
            "done": True,
            "step": "Thất bại: Chưa tìm thấy file weights/best.pt"
        })
        return {"error": "Chưa tìm thấy file weights/best.pt"}

    raw_dir = "data/raw"
    precalib_dir = "data/precalib"
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(precalib_dir) or len(os.listdir(precalib_dir)) == 0:
        progress_state.update({
            "running": False,
            "done": True,
            "step": "Thất bại: Thư mục precalib trống"
        })
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
        progress_state.update({
            "running": False,
            "done": True,
            "step": "Hoàn tất! Không có ảnh mới"
        })
        latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
        return {
            "message": "Không có ảnh mới nào cần phân tích!",
            "batch_id": latest_batch.id if latest_batch else None
        }

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

    # Tính toán total và current ban đầu
    all_thermal_in_precalib = [
        fn for fn in os.listdir(precalib_dir)
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')) 
        and fn not in rgb_images
    ]
    total_images = len(all_thermal_in_precalib)
    processed_count = total_images - len(thermal_files)

    progress_state.update({
        "running": True,
        "done": False,
        "current": processed_count,
        "total": total_images,
        "filename": "",
        "step": "Khởi động..."
    })

    # Khởi chạy background task
    background_tasks.add_task(
        run_analysis_pipeline_task,
        new_batch.id,
        thermal_files,
        list(processed_images),
        thermal_to_rgb,
        raw_dir,
        precalib_dir,
        results_dir,
        panel_power or 600.0
    )

    return {
        "message": "Bắt đầu phân tích AI...",
        "batch_id": new_batch.id,
    }


def _serialize_panels(panels: List) -> List:
    """
    Serialize panel list cho JSON response.
    Đảm bảo numpy arrays được convert về Python native types.
    """
    result = []
    for p in panels:
        review_status = p.get("review_status", "unreviewed")
        if review_status == "false_positive":
            include_in_report = False
        else:
            include_in_report = p.get("include_in_report", True)

        review_label = {
            "unreviewed": "Chưa duyệt",
            "confirmed_defect": "Đã xác nhận",
            "needs_review": "Cần kiểm tra lại",
            "false_positive": "Bỏ qua"
        }.get(review_status, "Chưa duyệt")

        center_val = p.get("center") or [0, 0]
        x_val = center_val[0] if len(center_val) > 0 else 0
        y_val = center_val[1] if len(center_val) > 1 else 0

        panel_out = {
            "id":           p.get("id", ""),
            "local_id":     p.get("local_id", ""),
            "row":          p.get("row", 0),
            "col":          p.get("col", 0),
            "class_name":   p.get("class_name", "panel"),
            "confidence":   p.get("confidence", 0.0),
            "bbox":         _to_list(p.get("bbox") or p.get("box", [])),
            "box":          _to_list(p.get("bbox") or p.get("box", [])),  # backward compat
            "polygon":      _to_list(p.get("outer_polygon") or p.get("polygon", [])),
            # V62 line-snap polygon data
            "outer_polygon":   _to_list(p.get("outer_polygon") or p.get("polygon", [])),
            "inner_polygon":   _to_list(p.get("inner_polygon", [])),
            "calc_polygon":    _to_list(p.get("calc_polygon") or p.get("inner_polygon", [])),
            "geometry_source": p.get("geometry_source", "yolo_fallback"),
            "snap_source":      p.get("snap_source", ""),
            "outer_area":      p.get("outer_area", p.get("area", 0.0)),
            "inner_area":      p.get("inner_area", p.get("area", 0.0)),
            "area":         p.get("area", 0.0),
            "center":       _to_list(center_val),
            "status":       p.get("status", "healthy"),
            "defects":      _serialize_defects(p.get("defects", [])),
            "total_defect_area_ratio_percent": p.get("total_defect_area_ratio_percent", 0.0),
            "max_defect_area_ratio_percent":   p.get("max_defect_area_ratio_percent", 0.0),
            "worst_severity":   p.get("worst_severity", "healthy"),
            "recommendation":   p.get("recommendation", "Không cần xử lý"),
            "main_defect_class": p.get("main_defect_class"),
            # GPS EXIF
            "gps_lat":          p.get("gps_lat"),
            "gps_lng":          p.get("gps_lng"),
            # Backward compat cho frontend cũ
            "total_panel_loss": p.get("total_panel_loss", 0.0),
            "x": x_val,
            "y": y_val,
            
            # Default review fields
            "review_status": review_status,
            "review_label": review_label,
            "review_note": p.get("review_note", ""),
            "include_in_report": include_in_report,
            "maintenance_priority": p.get("maintenance_priority", "medium"),
            "reviewer_name": p.get("reviewer_name", ""),
            "reviewed_at": p.get("reviewed_at", ""),
        }
        result.append(panel_out)
    return result


def _serialize_defects(defects: List) -> List:
    result = []
    for d in defects:
        class_name = d.get("class_name") or d.get("type", "")
        polygon = _to_list(d.get("polygon", []))
        area_ratio_percent = d.get("area_ratio_percent", d.get("loss", 0.0))
        
        result.append({
            # ── Core YOLO fields (không thay đổi) ──
            "class_name":        class_name,
            "confidence":        d.get("confidence", 0.0),
            "bbox":              _to_list(d.get("bbox") or d.get("box", [])),
            "box":               _to_list(d.get("bbox") or d.get("box", [])),
            "polygon":           polygon,
            "display_polygon":   _to_list(d.get("display_polygon") or polygon),
            "analysis_polygon":  _to_list(d.get("analysis_polygon") or polygon),
            "area":              d.get("area", 0.0),
            "center":            _to_list(d.get("center", [0, 0])),
            "area_ratio_percent": area_ratio_percent,
            "overlap_ratio":     d.get("overlap_ratio", 0.0),
            "area_inside_panel": d.get("area_inside_panel", 0.0),
            "relative_position": d.get("relative_position", {"u": 0, "v": 0}),
            "location_in_panel": d.get("location_in_panel", ""),
            "severity":          d.get("severity", ""),
            "recommendation":    d.get("recommendation", ""),
            # Backward compat
            "type": class_name,
            "loss": area_ratio_percent,
            # ── Thermal Validation fields (Relative Spatial-Thermal Contrast) ──
            "thermal_validation_status":     d.get("thermal_validation_status", "not_run"),
            "thermal_validation_score":      d.get("thermal_validation_score"),
            "rule_class":                    d.get("rule_class"),
            "final_class_suggestion":        d.get("final_class_suggestion"),
            "relative_hot_delta":            d.get("relative_hot_delta"),
            "relative_dark_delta":           d.get("relative_dark_delta"),
            "blue_suppression":              d.get("blue_suppression"),
            "tni":                           d.get("tni"),
            "area_ratio_inner":              d.get("area_ratio_inner"),
            "severity_by_relative_contrast": d.get("severity_by_relative_contrast"),
            "suggested_review_status":       d.get("suggested_review_status"),
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
# --- KHỐI 7: LẤY ẢNH CẮT THEO TỌA ĐỘ (PANEL IMAGE) ---
# ================================
@app.get("/api/v1/panel-image")
def get_panel_image(
    filename: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    polygon: str = None
):
    import cv2
    import os
    from fastapi.responses import Response

    path_thermal = os.path.join("data/results", filename)
    path_raw = os.path.join("data/raw", filename)
    
    img_path = None
    is_rgb = False
    
    if os.path.exists(path_thermal):
        img_path = path_thermal
    elif os.path.exists(path_raw):
        img_path = path_raw
        is_rgb = True
    else:
        return Response(status_code=404)
        
    img = cv2.imread(img_path)
    if img is None:
        return Response(status_code=404)
        
    h_img, w_img = img.shape[:2]
    
    w_t, h_t = 640, 512
    
    scale_x = w_img / w_t if is_rgb else 1.0
    scale_y = h_img / h_t if is_rgb else 1.0
    
    crop_w_t = int(w_t * 0.5)
    crop_h_t = int(h_t * 0.5)
    
    cx_t = (x1 + x2) / 2
    cy_t = (y1 + y2) / 2
    
    if is_rgb:
        cx_r = int(cx_t * scale_x)
        cy_r = int(cy_t * scale_y)
        crop_w_r = int(crop_w_t * scale_x)
        crop_h_r = int(crop_h_t * scale_y)
        
        x1_crop = max(0, cx_r - crop_w_r // 2)
        y1_crop = max(0, cy_r - crop_h_r // 2)
        x2_crop = min(w_img, cx_r + crop_w_r // 2)
        y2_crop = min(h_img, cy_r + crop_h_r // 2)
        
        crop_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if crop_img is not None and crop_img.size > 0:
            x1_r = int(x1 * scale_x)
            y1_r = int(y1 * scale_y)
            x2_r = int(x2 * scale_x)
            y2_r = int(y2 * scale_y)
            
            box_x1 = x1_r - x1_crop
            box_y1 = y1_r - y1_crop
            box_x2 = x2_r - x1_crop
            box_y2 = y2_r - y1_crop
            
            cv2.rectangle(crop_img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
    else:
        x1_crop = max(0, int(cx_t - crop_w_t // 2))
        y1_crop = max(0, int(cy_t - crop_h_t // 2))
        x2_crop = min(w_img, int(cx_t + crop_w_t // 2))
        y2_crop = min(h_img, int(cy_t + crop_h_t // 2))
        
        crop_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
        
    if crop_img is None or crop_img.size == 0:
        return Response(status_code=404)
        
    ret, buf = cv2.imencode('.jpg', crop_img)
    if not ret:
        return Response(status_code=500)
        
    return Response(content=buf.tobytes(), media_type="image/jpeg")


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
    ai_results = []
    if batch_id > 0:
        ai_results = db.query(models.AiResult).join(models.Image).filter(
            models.Image.batch_id == batch_id
        ).all()
    
    report_name = f"Report_Batch_{batch_id}.pdf"
    report_path = os.path.join("data", report_name)
    
    ReportGenerator.generate_inspection_report(batch_id, ai_results, report_path)

    if batch_id > 0:
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
    system_capacity: Optional[str] = None
    supervisor: Optional[str] = None
    notes: Optional[str] = None
    data_type: Optional[str] = None
    ai_model: Optional[str] = None
    system_version: Optional[str] = None

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
    
    # Gộp scope và các trường extra thành JSON để lưu vào cột scope (255 chars max)
    import json
    scope_data = {
        "s": req.scope or "",
        "sc": req.system_capacity or "",
        "sv": req.supervisor or "",
        "nt": req.notes or "",
        "dt": req.data_type or "",
        "am": req.ai_model or "",
        "sys": req.system_version or ""
    }
    batch.scope = json.dumps(scope_data)
    
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
        "batch_id": batch.id,
        "project_name": batch.project_name,
        "location": batch.location,
        "scan_time": batch.scan_time,
        "operator": batch.operator,
        "device": batch.device,
        "scope": req.scope or "",
        "panel_power": batch.panel_power,
        "system_capacity": req.system_capacity or "",
        "supervisor": req.supervisor or "",
        "notes": req.notes or "",
        "data_type": req.data_type or "",
        "ai_model": req.ai_model or "",
        "system_version": req.system_version or ""
    }


# ================================
# --- ANOMALY DEFECT MANUAL REVIEW ---
# ================================
class ReviewItemRequest(BaseModel):
    review_status: str
    review_note: Optional[str] = ""
    maintenance_priority: Optional[str] = "medium"
    reviewer_name: Optional[str] = ""
    reviewed_at: Optional[str] = ""

@app.get("/api/v1/review/items")
def get_review_items(batch_id: Optional[int] = None, db: Session = Depends(get_db)):
    if batch_id is None:
        latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
        if not latest_batch:
            return {"batch_id": None, "total": 0, "items": []}
        batch_id = latest_batch.id
    else:
        batch = db.query(models.UploadBatch).filter(models.UploadBatch.id == batch_id).first()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

    # Get images of the batch, sorted by filename
    images = db.query(models.Image).filter(models.Image.batch_id == batch_id).order_by(models.Image.filename).all()
    image_to_index = {img.id: idx + 1 for idx, img in enumerate(images)}

    ai_results = db.query(models.AiResult).join(models.Image).filter(models.Image.batch_id == batch_id).all()
    
    items = []
    import json
    import re

    def parse_local_id(local_id):
        if not local_id:
            return (0, 0, 0)
        b = re.search(r"B(\d+)", local_id)
        r = re.search(r"R(\d+)", local_id)
        c = re.search(r"C(\d+)", local_id)
        block = int(b.group(1)) if b else 0
        row = int(r.group(1)) if r else 0
        col = int(c.group(1)) if c else 0
        return (block, row, col)

    for r in ai_results:
        if not r.defect_type or not r.defect_type.startswith("{"):
            continue
        try:
            panel_detail = json.loads(r.defect_type)
        except Exception:
            continue
        
        defects = panel_detail.get("defects", [])
        if not defects:
            continue
            
        panel_info = db.query(models.Panel).filter(models.Panel.id == r.panel_id).first()
        if not panel_info:
            continue
            
        img = r.image
        img_idx = image_to_index.get(img.id, 1)

        row_val, col_val = 0, 0
        if panel_info.local_id:
            m = re.match(r"R(\d+)_C(\d+)", panel_info.local_id)
            if m:
                row_val = int(m.group(1))
                col_val = int(m.group(2))
                
        raw_polygon = panel_detail.get("polygon", [])
        outer_polygon = panel_detail.get("outer_polygon") or raw_polygon
        inner_polygon = panel_detail.get("inner_polygon", [])
        calc_polygon = panel_detail.get("calc_polygon") or inner_polygon
        
        review_status = panel_detail.get("review_status", "unreviewed")
        if review_status == "false_positive":
            include_in_report = False
        elif review_status in ["confirmed_defect", "needs_review", "unreviewed"]:
            include_in_report = True
        else:
            include_in_report = panel_detail.get("include_in_report", True)

        review_label = {
            "unreviewed": "Chưa duyệt",
            "confirmed_defect": "Đã xác nhận",
            "needs_review": "Cần kiểm tra lại",
            "false_positive": "Bỏ qua"
        }.get(review_status, "Chưa duyệt")
        
        items.append({
            "review_item_id": f"{img.filename}::{panel_info.local_id}",
            "filename": img.filename,
            "image_index": img_idx,
            "image_url": f"/data/precalib/{img.filename}",
            "annotated_image_url": f"/data/results/{img.filename}",
            "local_id": panel_info.local_id,
            "row": panel_detail.get("row", row_val),
            "col": panel_detail.get("col", col_val),
            "panel": {
                "polygon": outer_polygon,
                "outer_polygon": outer_polygon,
                "inner_polygon": inner_polygon,
                "calc_polygon": calc_polygon,
                "bbox": panel_detail.get("bbox", []),
                "geometry_source": panel_detail.get("geometry_source", "unknown"),
            },
            "defects": _serialize_defects(defects),
            "review_status": review_status,
            "review_label": review_label,
            "review_note": panel_detail.get("review_note", ""),
            "include_in_report": include_in_report,
            "maintenance_priority": panel_detail.get("maintenance_priority", "medium"),
            "reviewer_name": panel_detail.get("reviewer_name", ""),
            "reviewed_at": panel_detail.get("reviewed_at", ""),
        })

    items.sort(key=lambda x: (x["image_index"], x["filename"], parse_local_id(x["local_id"])))
    
    return {
        "batch_id": batch_id,
        "total": len(items),
        "items": items
    }

@app.post("/api/v1/review/items/{review_item_id}")
def update_review_item(
    review_item_id: str,
    req: ReviewItemRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    if "::" not in review_item_id:
        raise HTTPException(status_code=400, detail="Invalid review_item_id format. Expected 'filename::local_id'")
        
    filename, local_id = review_item_id.split("::", 1)
    
    latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
    if not latest_batch:
        raise HTTPException(status_code=404, detail="No batches found")
        
    db_image = db.query(models.Image).filter(
        models.Image.batch_id == latest_batch.id,
        models.Image.filename == filename
    ).first()
    
    if not db_image:
        raise HTTPException(status_code=404, detail=f"Image {filename} not found in latest batch")
        
    db_panel = db.query(models.Panel).filter(models.Panel.local_id == local_id).first()
    if not db_panel:
        raise HTTPException(status_code=404, detail=f"Panel {local_id} not found")
        
    db_result = db.query(models.AiResult).filter(
        models.AiResult.image_id == db_image.id,
        models.AiResult.panel_id == db_panel.id
    ).first()
    
    if not db_result:
        raise HTTPException(status_code=404, detail=f"AI result not found for panel {local_id} in image {filename}")
        
    if not db_result.defect_type or not db_result.defect_type.startswith("{"):
        raise HTTPException(status_code=400, detail="AI result is not in JSON format")
        
    import json
    try:
        panel_detail = json.loads(db_result.defect_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI result JSON: {e}")
        
    status = req.review_status
    if status not in ["unreviewed", "confirmed_defect", "needs_review", "false_positive"]:
        raise HTTPException(status_code=400, detail="Invalid review_status value")
        
    labels = {
        "unreviewed": "Chưa duyệt",
        "confirmed_defect": "Đã xác nhận",
        "needs_review": "Cần kiểm tra lại",
        "false_positive": "Bỏ qua"
    }
    
    panel_detail["review_status"] = status
    panel_detail["review_label"] = labels[status]
    panel_detail["review_note"] = req.review_note or ""
    panel_detail["maintenance_priority"] = req.maintenance_priority or "medium"
    panel_detail["reviewer_name"] = req.reviewer_name or ""
    panel_detail["reviewed_at"] = req.reviewed_at or ""
    panel_detail["include_in_report"] = (status != "false_positive")
    
    db_result.defect_type = json.dumps(panel_detail)
    db.commit()
    
    # Construct updated_panel object for response
    import re
    row_val, col_val = 0, 0
    if db_panel.local_id:
        m = re.match(r"R(\d+)_C(\d+)", db_panel.local_id)
        if m:
            row_val = int(m.group(1))
            col_val = int(m.group(2))
            
    raw_polygon = panel_detail.get("polygon", [])
    outer_polygon = panel_detail.get("outer_polygon") or raw_polygon
    inner_polygon = panel_detail.get("inner_polygon", [])
    calc_polygon = panel_detail.get("calc_polygon") or inner_polygon
    
    if not outer_polygon or len(outer_polygon) < 3:
        outer_polygon = raw_polygon

    geometry_source = panel_detail.get("geometry_source", "unknown")
    snap_source = panel_detail.get("snap_source", "")
    outer_area = panel_detail.get("outer_area", 0.0)
    inner_area = panel_detail.get("inner_area", panel_detail.get("area", 0.0))

    if status == "false_positive":
        status_val = "healthy"
    else:
        status_val = "faulty" if panel_detail.get("defects", []) else "healthy"

    updated_panel = {
        "local_id": db_panel.local_id,
        "row": panel_detail.get("row", row_val),
        "col": panel_detail.get("col", col_val),
        "x": db_panel.x_coord,
        "y": db_panel.y_coord,
        "center": [db_panel.x_coord, db_panel.y_coord],
        "bbox": panel_detail.get("bbox", []),
        "box": panel_detail.get("bbox", []),
        "polygon": outer_polygon,
        "outer_polygon": outer_polygon,
        "inner_polygon": inner_polygon,
        "calc_polygon": calc_polygon,
        "geometry_source": geometry_source,
        "snap_source": snap_source,
        "outer_area": outer_area,
        "inner_area": inner_area,
        "total_panel_loss": db_result.loss_pct,
        "total_defect_area_ratio_percent": db_result.loss_pct,
        "confidence": db_result.confidence,
        "defects": _serialize_defects(panel_detail.get("defects", [])),
        "status": status_val,
        "worst_severity": panel_detail.get("worst_severity", "healthy"),
        "recommendation": panel_detail.get("recommendation", "Không cần xử lý"),
        "gps_lat": panel_detail.get("gps_lat"),
        "gps_lng": panel_detail.get("gps_lng"),
        "main_defect_class": panel_detail.get("main_defect_class"),
        "review_status": status,
        "review_label": labels[status],
        "review_note": panel_detail["review_note"],
        "include_in_report": panel_detail["include_in_report"],
        "maintenance_priority": panel_detail["maintenance_priority"],
        "reviewer_name": panel_detail["reviewer_name"],
        "reviewed_at": panel_detail["reviewed_at"],
    }
    
    ai_results = db.query(models.AiResult).join(models.Image).filter(
        models.Image.batch_id == latest_batch.id
    ).all()
    
    if ai_results:
        report_name = f"Report_Batch_{latest_batch.id}.pdf"
        report_path = os.path.join("data", report_name)
        
        background_tasks.add_task(ReportGenerator.generate_inspection_report, latest_batch.id, ai_results, report_path)
        
        db_report = db.query(models.Report).filter(models.Report.batch_id == latest_batch.id).first()
        if not db_report:
            db_report = models.Report(batch_id=latest_batch.id, file_path=report_path)
            db.add(db_report)
        else:
            db_report.file_path = report_path
        db.commit()
        
    return {
        "ok": True,
        "review_item_id": review_item_id,
        "review_status": status,
        "review_label": labels[status],
        "review_note": panel_detail["review_note"],
        "maintenance_priority": panel_detail["maintenance_priority"],
        "reviewer_name": panel_detail["reviewer_name"],
        "reviewed_at": panel_detail["reviewed_at"],
        "include_in_report": panel_detail["include_in_report"],
        "batch_id": latest_batch.id,
        "updated_panel": updated_panel
    }

# ================================
# --- REVIEW SYNC ENDPOINT ---
# ================================
class ReviewSyncRequest(BaseModel):
    batch_id: Optional[int] = None

@app.post("/api/v1/review/sync")
def sync_review(req: ReviewSyncRequest, db: Session = Depends(get_db)):
    """
    Đồng bộ review status:
    1. Chuẩn hóa include_in_report theo rule:
       - false_positive → include_in_report = False
       - confirmed_defect / needs_review / unreviewed → include_in_report = True
    2. Tính lại summary cho batch.
    3. Commit DB.
    """
    import json as _json

    if req.batch_id is None:
        latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
        if not latest_batch:
            raise HTTPException(status_code=404, detail="No batch found")
        batch_id = latest_batch.id
    else:
        batch_id = req.batch_id
        latest_batch = db.query(models.UploadBatch).filter(models.UploadBatch.id == batch_id).first()
        if not latest_batch:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    ai_results = db.query(models.AiResult).join(models.Image).filter(
        models.Image.batch_id == batch_id
    ).all()

    summary = {
        "total_ai_detected": 0,
        "confirmed_defect": 0,
        "needs_review": 0,
        "false_positive": 0,
        "unreviewed": 0,
        "included_in_report": 0,
        "excluded_from_report": 0,
    }

    for r in ai_results:
        if not r.defect_type or not r.defect_type.startswith("{"):
            continue
        try:
            panel_detail = _json.loads(r.defect_type)
        except Exception:
            continue

        defects = panel_detail.get("defects", [])
        if not defects:
            continue

        summary["total_ai_detected"] += 1

        status = panel_detail.get("review_status", "unreviewed")

        # Chuẩn hóa include_in_report
        if status == "false_positive":
            include = False
        else:
            include = True

        panel_detail["include_in_report"] = include
        r.defect_type = _json.dumps(panel_detail)

        # Đếm summary
        if status in ("confirmed_defect", "needs_review", "unreviewed", "false_positive"):
            summary[status] += 1
        else:
            summary["unreviewed"] += 1

        if include:
            summary["included_in_report"] += 1
        else:
            summary["excluded_from_report"] += 1

    db.commit()
    logger.info(f"[review/sync] Batch {batch_id}: {summary}")

    return {
        "ok": True,
        "batch_id": batch_id,
        "summary": summary,
        "message": "Review synchronized"
    }


@app.get("/api/v1/review/summary")
def get_review_summary(batch_id: Optional[int] = None, db: Session = Depends(get_db)):
    if batch_id is None:
        latest_batch = db.query(models.UploadBatch).order_by(models.UploadBatch.id.desc()).first()
        if not latest_batch:
            return {
                "batch_id": None,
                "total_review_items": 0,
                "unreviewed": 0,
                "confirmed_defect": 0,
                "needs_review": 0,
                "false_positive": 0
            }
        batch_id = latest_batch.id
    else:
        batch = db.query(models.UploadBatch).filter(models.UploadBatch.id == batch_id).first()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

    ai_results = db.query(models.AiResult).join(models.Image).filter(models.Image.batch_id == batch_id).all()
    
    counts = {
        "unreviewed": 0,
        "confirmed_defect": 0,
        "needs_review": 0,
        "false_positive": 0
    }
    
    import json
    for r in ai_results:
        if not r.defect_type or not r.defect_type.startswith("{"):
            continue
        try:
            panel_detail = json.loads(r.defect_type)
        except Exception:
            continue
            
        defects = panel_detail.get("defects", [])
        if not defects:
            continue
            
        status = panel_detail.get("review_status", "unreviewed")
        if status in counts:
            counts[status] += 1
        else:
            counts["unreviewed"] += 1
            
    total = sum(counts.values())
    return {
        "batch_id": batch_id,
        "total_review_items": total,
        **counts
    }