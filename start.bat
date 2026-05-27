@echo off
title Solar Inspection Project Launcher
color 0B
echo =====================================================================
echo  [SOLAR AI INSPECTION] TRINH KHOI DONG HE THONG TU DONG (WINDOWS)
echo =====================================================================
echo.
echo  [*] Huong dan: Vui long khong dong cac cua so cmd duoc mo len.
echo                 Chung la cac may chu dang chay ngam (Backend ^& Frontend).
echo.
echo ---------------------------------------------------------------------
echo  [1/3] Dang khoi dong Backend Server (FastAPI / Uvicorn)...
echo ---------------------------------------------------------------------
start "Solar AI Backend" cmd /k "cd /d datn_be && uvicorn app.main:app --reload"

echo  [2/3] Dang khoi dong Frontend Server (React / Vite / npm dev)...
echo ---------------------------------------------------------------------
start "Solar AI Frontend" cmd /k "cd /d datn_fe && npm run dev"

echo  [3/3] Dang doi may chu phan hoi de mo trinh duyet...
echo ---------------------------------------------------------------------
timeout /t 4 >nul

echo  [+] Dang mo Trinh duyet den: http://localhost:5173
start http://localhost:5173

echo.
echo =====================================================================
echo  [OK] He thong da duoc khoi dong song song thanh cong!
echo  [+] Ban co the bat dau trai nghiem ung dung tren trinh duyet.
echo =====================================================================
echo.
pause
