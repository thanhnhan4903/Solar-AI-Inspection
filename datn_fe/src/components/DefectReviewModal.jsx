import React, { useState, useEffect, useRef } from 'react';
import {
    X, ChevronLeft, ChevronRight, Check, AlertTriangle, XCircle, Save, Eye, Info
} from 'lucide-react';
import api, { fetchReviewItems, updateReviewItem, syncReview } from '../api';
import { PanelCropCanvas } from './PanelCropCanvas';

// ────────────────────────────────────────────────────────────────
// CONSTANTS
// ────────────────────────────────────────────────────────────────
const BASE_URL = 'http://127.0.0.1:8000';

const SEVERITY_OPTIONS = [
    { value: 'level_1_monitoring', label: 'Mức 1 – Theo dõi' },
    { value: 'level_2_inspection', label: 'Mức 2 – Cần kiểm tra' },
    { value: 'level_3_priority',   label: 'Mức 3 – Ưu tiên xử lý' },
    { value: 'recheck_required',   label: 'Cần chụp lại' },
    { value: 'healthy',            label: 'Bình thường' },
];

const DEFECT_CLASS_OPTIONS = [
    { value: 'hotspot_single_cell', label: 'Hotspot single-cell' },
    { value: 'hotspot_multi_cell',  label: 'Hotspot multi-cell' },
    { value: 'crack',               label: 'Nứt / crack' },
    { value: 'shading',             label: 'Che bóng / Cần chụp lại' },
];

const DEFECT_CLASS_LABELS = {
    hotspot_single_cell: 'Hotspot single-cell',
    hotspot_multi_cell:  'Hotspot multi-cell',
    crack:               'Nứt / crack',
    shading:             'Che bóng / Cần chụp lại',
};

const SEVERITY_LABELS = {
    healthy:             'Bình thường',
    level_1_monitoring:  'Mức 1 – Theo dõi',
    level_2_inspection:  'Mức 2 – Cần kiểm tra',
    level_3_priority:    'Mức 3 – Ưu tiên xử lý',
    recheck_required:    'Cần chụp lại',
};

// ────────────────────────────────────────────────────────────────
// HELPERS
// ────────────────────────────────────────────────────────────────
function translateSeverity(value) {
    return SEVERITY_LABELS[String(value || '').toLowerCase()] || 'Chưa phân loại';
}

function translateLocation(loc) {
    if (!loc) return 'Không xác định';
    const dict = {
        'upper-left': 'Góc trên trái',
        'upper-right': 'Góc trên phải',
        'lower-left': 'Góc dưới trái',
        'lower-right': 'Góc dưới phải',
        'center': 'Ở giữa',
        'top': 'Phía trên',
        'bottom': 'Phía dưới',
        'left': 'Bên trái',
        'right': 'Bên phải',
        'substring_1': 'Chuỗi diode 1 (Trái)',
        'substring_2': 'Chuỗi diode 2 (Giữa)',
        'substring_3': 'Chuỗi diode 3 (Phải)',
        'diode_1': 'Chuỗi diode 1 (Trái)',
        'diode_2': 'Chuỗi diode 2 (Giữa)',
        'diode_3': 'Chuỗi diode 3 (Phải)'
    };
    return dict[loc] || loc;
}

function roundValue(value, decimals) {
    if (value == null) return '0';
    return Number(value).toFixed(decimals);
}

function getConfidenceBadge(confidence) {
    const pct = (confidence || 0) * 100;
    if (pct < 60) return { label: 'Độ tin cậy thấp – cần kiểm tra kỹ', color: '#ef4444', bg: 'rgba(239,68,68,0.12)', icon: '⚠' };
    if (pct < 80) return { label: 'Độ tin cậy trung bình', color: '#f59e0b', bg: 'rgba(245,158,11,0.10)', icon: '●' };
    return { label: 'Độ tin cậy cao', color: '#10b981', bg: 'rgba(16,185,129,0.10)', icon: '✓' };
}


// ────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ────────────────────────────────────────────────────────────────
export default function DefectReviewModal({ isOpen, onClose, batchId, onRefresh, panelPower = 600 }) {
    const [items, setItems]               = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [loading, setLoading]           = useState(false);
    const [saving, setSaving]             = useState(false);
    const [hasFetchError, setHasFetchError] = useState(false);

    // Form state
    const [selectedStatus, setSelectedStatus]         = useState('unreviewed');
    const [reviewNote, setReviewNote]                 = useState('');
    const [maintenancePriority, setMaintenancePriority] = useState('level_1_monitoring');
    const [reviewerName, setReviewerName]             = useState('');
    const [reviewedAt, setReviewedAt]                 = useState('');
    const [correctedClass, setCorrectedClass]         = useState('');
    const [showStatusWarning, setShowStatusWarning]   = useState(false);

    // Sync state
    const [syncing, setSyncing]     = useState(false);
    const [syncError, setSyncError] = useState('');

    // ── Fetch ──────────────────────────────────────────────────
    useEffect(() => {
        if (isOpen && batchId) {
            setLoading(true);
            setHasFetchError(false);
            fetchReviewItems(batchId)
                .then(res => {
                    const fetched = res.data.items || [];
                    setItems(fetched);
                    setCurrentIndex(0);
                    if (fetched.length > 0) initFormFromItem(fetched[0]);
                    setLoading(false);
                })
                .catch(err => {
                    console.error('Failed to fetch review items:', err);
                    setHasFetchError(true);
                    setLoading(false);
                });
        }
    }, [isOpen, batchId]);

    // ── Sync form on index change ──────────────────────────────
    useEffect(() => {
        if (items.length > 0 && currentIndex < items.length) {
            initFormFromItem(items[currentIndex]);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentIndex]);

    function initFormFromItem(item) {
        const existingStatus = item.review_status;
        if (existingStatus && existingStatus !== 'unreviewed') {
            setSelectedStatus(existingStatus);
        } else {
            const d0 = (item.defects || [])[0];
            const suggested = d0?.suggested_review_status;
            setSelectedStatus(suggested === 'confirmed_defect' ? 'confirmed_defect'
                : suggested === 'needs_review' ? 'needs_review' : 'unreviewed');
        }
        setReviewNote(item.review_note || '');
        // Map old medium/low/high values to new severity system if needed
        const mp = item.maintenance_priority || '';
        const mapped = {
            low: 'level_1_monitoring', medium: 'level_2_inspection',
            high: 'level_3_priority', urgent: 'level_3_priority'
        }[mp] || (SEVERITY_OPTIONS.find(o => o.value === mp) ? mp : 'level_1_monitoring');
        setMaintenancePriority(mapped);
        setReviewerName(item.reviewer_name || '');
        setReviewedAt(item.reviewed_at || '');
        const d0 = (item.defects || [])[0];
        setCorrectedClass(item.corrected_class_name || d0?.class_name || '');
        setShowStatusWarning(false);
    }

    if (!isOpen) return null;

    // ── Current item ──────────────────────────────────────────
    const currentItem = (items.length > 0 && currentIndex < items.length)
        ? items[currentIndex]
        : { local_id: '---', row: '—', col: '—', filename: '—', defects: [{ type: '—', confidence: 0, area_ratio_percent: 0, class_name: 'healthy', thermal_validation_status: 'not_run' }], annotated_image_url: '', image_url: '', panel: { geometry_source: 'N/A' } };

    const d0 = currentItem.defects?.[0] || {};
    const hasItems = items.length > 0;
    const reviewedCount = items.filter(it => it.review_status && it.review_status !== 'unreviewed').length;
    const allReviewed = hasItems && reviewedCount === items.length;
    const confidenceBadge = getConfidenceBadge(d0.confidence);
    const isShading = (d0.class_name || '').toLowerCase().includes('shading');

    // ── Save ─────────────────────────────────────────────────
    const handleSave = async (silent = false) => {
        if (!hasItems) return;
        const item = items[currentIndex];
        let ts = reviewedAt;
        if (selectedStatus !== 'unreviewed' && !ts) {
            ts = new Date().toLocaleString('vi-VN');
        } else if (selectedStatus === 'unreviewed') {
            ts = '';
        }
        setSaving(true);
        try {
            await updateReviewItem(item.review_item_id, {
                review_status: selectedStatus,
                review_note: reviewNote,
                maintenance_priority: maintenancePriority,
                reviewer_name: reviewerName || 'Chưa cập nhật',
                reviewed_at: ts
                // NOTE: corrected_class_name is stored in frontend state only.
                // Backend schema does not yet support this field.
                // When backend adds corrected_class_name, include it here.
            });
            const updated = [...items];
            updated[currentIndex] = { ...item, review_status: selectedStatus, review_note: reviewNote, maintenance_priority: maintenancePriority, reviewer_name: reviewerName || 'Chưa cập nhật', reviewed_at: ts };
            setItems(updated);
            setReviewedAt(ts);
            if (!silent && onRefresh) onRefresh();
        } catch (err) {
            console.error('Failed to save review:', err);
            alert('Không thể lưu kết quả duyệt lỗi!');
        } finally {
            setSaving(false);
        }
    };

    const handleNext = async () => {
        if (selectedStatus === 'unreviewed') {
            setShowStatusWarning(true);
            return;
        }
        await handleSave(true);
        if (currentIndex < items.length - 1) setCurrentIndex(currentIndex + 1);
        else { if (onRefresh) onRefresh(); alert('Đã duyệt xong lỗi cuối cùng!'); }
    };

    const handlePrev = async () => {
        await handleSave(true);
        if (currentIndex > 0) setCurrentIndex(currentIndex - 1);
    };

    const handleFinishReviewAndSync = async () => {
        setSyncing(true);
        setSyncError('');
        try {
            await handleSave(true);
            await syncReview(batchId);
            window.dispatchEvent(new Event('review-sync-completed'));
            if (onRefresh) onRefresh();
            onClose();
        } catch (err) {
            console.error('Sync failed:', err);
            setSyncError('Đồng bộ thất bại. Vui lòng thử lại.');
        } finally {
            setSyncing(false);
        }
    };

    // ── Status card config ───────────────────────────────────
    const statusCards = [
        {
            value: 'confirmed_defect',
            label: 'Đã xác nhận',
            desc: 'Xác nhận tấm pin có bất thường nhiệt này',
            icon: <Check size={16} />,
            activeColor: '#10b981',
            activeBg: 'rgba(16,185,129,0.14)',
            activeBorder: '#10b981',
            iconColor: '#10b981',
        },
        {
            value: 'needs_review',
            label: 'Cần kiểm tra lại',
            desc: 'Cần khảo sát hiện trường để xác nhận',
            icon: <AlertTriangle size={16} />,
            activeColor: '#f59e0b',
            activeBg: 'rgba(245,158,11,0.14)',
            activeBorder: '#f59e0b',
            iconColor: '#f59e0b',
        },
        {
            value: 'false_positive',
            label: 'Bỏ qua (False Positive)',
            desc: 'AI nhận nhầm – phản xạ nhiệt, bụi, bóng che...',
            icon: <XCircle size={16} />,
            activeColor: '#94a3b8',
            activeBg: 'rgba(148,163,184,0.14)',
            activeBorder: '#94a3b8',
            iconColor: '#64748b',
        },
    ];

    // ── Thermal validation meta ──────────────────────────────
    const tvStatus = d0.thermal_validation_status;
    const tvMeta = (() => {
        const map = {
            hotter_than_panel_background:         { label: 'Vùng lỗi nóng hơn nền panel',            color: '#f87171', bg: 'rgba(248,113,113,0.10)', icon: '🔥' },
            weak_relative_contrast:               { label: 'Tương phản nhiệt chưa rõ',               color: '#f59e0b', bg: 'rgba(245,158,11,0.10)', icon: '⚠️' },
            thermal_mismatch_need_review:         { label: 'Không khớp đặc trưng nhiệt, cần xem xét', color: '#f97316', bg: 'rgba(249,115,22,0.10)', icon: '⚠️' },
            cooler_than_panel_background_recheck: { label: 'Vùng tối hơn nền panel, cần chụp lại',   color: '#38bdf8', bg: 'rgba(56,189,248,0.10)', icon: '❄️' },
            shading_need_recheck:                 { label: 'Che bóng, cần chụp lại',                 color: '#818cf8', bg: 'rgba(129,140,248,0.10)', icon: '☁️' },
            shape_based_detection_review:         { label: 'Lỗi hình thái, cần kiểm duyệt',          color: '#c084fc', bg: 'rgba(192,132,252,0.10)', icon: '🔍' },
        };
        return map[tvStatus] || { label: 'Chưa kiểm chứng', color: '#64748b', bg: 'rgba(100,116,139,0.08)', icon: '—' };
    })();

    const fmtDelta = (val) => {
        if (val == null) return 'Chưa có dữ liệu';
        const n = Number(val);
        return n > 0 ? `+${n.toFixed(3)}` : n.toFixed(3);
    };

    // ── Styles helpers ───────────────────────────────────────
    const inputStyle = {
        width: '100%', backgroundColor: '#1e293b',
        border: '1px solid #334155', borderRadius: 10, padding: '9px 13px',
        color: hasItems ? '#cbd5e1' : '#64748b', fontSize: 13, outline: 'none',
        boxSizing: 'border-box',
        opacity: hasItems ? 1 : 0.5,
    };
    const labelStyle = { display: 'block', color: '#94a3b8', fontSize: 11, fontWeight: 700, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' };

    // ── Render ────────────────────────────────────────────────
    return (
        <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(3,7,18,0.88)', backdropFilter: 'blur(14px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999, padding: 16 }}>
            <div style={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', width: '100%', maxWidth: 1120, height: '92vh', borderRadius: 22, display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: '0 30px 60px -10px rgba(0,0,0,0.6)' }}>

                {/* ── Header ── */}
                <div style={{ padding: '14px 24px', borderBottom: '1px solid #1e293b', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'linear-gradient(90deg,#1e293b 0%,#0f172a 100%)', flexShrink: 0 }}>
                    <div>
                        <h2 style={{ margin: 0, color: '#f8fafc', fontSize: 18, fontWeight: 700 }}>Duyệt lỗi phát hiện – Manual Review</h2>
                        <p style={{ margin: '3px 0 0', color: '#94a3b8', fontSize: 12 }}>
                            Đợt #{batchId || '---'} &nbsp;•&nbsp; Đã duyệt <strong style={{ color: '#38bdf8' }}>{reviewedCount}/{items.length}</strong> lỗi
                        </p>
                    </div>
                    <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 8, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <X size={20} />
                    </button>
                </div>

                {/* ── Body ── */}
                <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
                    {loading ? (
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
                            <div className="spinner" style={{ marginRight: 12 }} />Đang tải danh sách lỗi...
                        </div>
                    ) : hasFetchError ? (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', padding: 40 }}>
                            <AlertTriangle size={48} style={{ color: '#ef4444', marginBottom: 16 }} />
                            <h3 style={{ color: '#f8fafc', margin: '0 0 8px' }}>Lỗi tải dữ liệu</h3>
                            <p style={{ margin: 0, textAlign: 'center', maxWidth: 480, lineHeight: 1.6 }}>
                                Đợt #{batchId} không tìm thấy. Hãy <strong>nhấn F5</strong> để tải lại.
                            </p>
                        </div>
                    ) : (
                        <>
                            {/* ── LEFT: Image crop ── */}
                            <div style={{ flex: '0 0 500px', padding: '18px 20px', borderRight: '1px solid #1e293b', display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', backgroundColor: '#020617' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ color: '#38bdf8', fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                        <Eye size={13} style={{ marginRight: 5, verticalAlign: 'middle' }} />
                                        Vùng lỗi đang duyệt
                                    </span>
                                    <span style={{ color: '#64748b', fontSize: 11 }}>{currentItem.filename}</span>
                                </div>

                                {hasItems ? (
                                    <PanelCropCanvas
                                        item={currentItem}
                                        imgWidth={currentItem.image_width || 640}
                                        imgHeight={currentItem.image_height || 512}
                                    />
                                ) : (
                                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', borderRadius: 12, border: '1px dashed #1e293b', padding: 32 }}>
                                        <Check size={40} style={{ color: '#10b981', marginBottom: 12 }} />
                                        <div style={{ color: '#f8fafc', fontWeight: 600 }}>Không có lỗi nào cần duyệt</div>
                                        <div style={{ fontSize: 12, marginTop: 6, textAlign: 'center', lineHeight: 1.6 }}>Hệ thống không phát hiện dị thường hoặc đợt kiểm tra chưa hoàn tất.</div>
                                    </div>
                                )}

                                {/* Defect info table */}
                                {hasItems && (
                                    <div style={{ backgroundColor: '#0d1828', border: '1px solid #1e293b', borderRadius: 12, padding: '12px 14px', fontSize: 12 }}>
                                        <div style={{ color: '#64748b', fontSize: 11, fontWeight: 700, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                            <Info size={11} style={{ marginRight: 4, verticalAlign: 'middle' }} />Thông tin lỗi
                                        </div>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '5px 12px' }}>
                                            {[
                                                ['Panel ID', currentItem.local_id],
                                                ['Ảnh nguồn', currentItem.filename],
                                                ['Loại lỗi AI', DEFECT_CLASS_LABELS[d0.class_name] || d0.class_name || '—'],
                                                ['Loại lỗi sau duyệt', <span style={{ color: '#eab308' }}>{DEFECT_CLASS_LABELS[correctedClass] || correctedClass || '—'}</span>],
                                                ['Mức ưu tiên AI', translateSeverity(d0.severity)],
                                                ['Mức ưu tiên sau duyệt', <span style={{ color: '#eab308' }}>{translateSeverity(maintenancePriority)}</span>],
                                                ['Độ tin cậy AI', `${roundValue(d0.confidence * 100, 1)}%`],
                                                ['Vị trí trên tấm', translateLocation(d0.location_in_panel)],
                                                ['Tương phản nhiệt', fmtDelta(d0.relative_thermal_delta)],
                                                ['Kết luận kiểm chứng', tvMeta.label],
                                                ['Diện tích lỗi', `${roundValue(d0.area_ratio_percent, 2)}%`],
                                                ['Công suất hao hụt', isShading ? 'Không tính (bóng che)' : `${roundValue(currentItem.power_loss_w, 1)} W`],
                                                ['Khuyến nghị', (() => {
                                                    const p = maintenancePriority;
                                                    if (p === 'level_1_monitoring') return 'Theo dõi ở lần kiểm tra định kỳ tiếp theo.';
                                                    if (p === 'level_2_inspection') return 'Cần cử kỹ sư đến kiểm tra thực tế.';
                                                    if (p === 'level_3_priority') return 'Ưu tiên xử lý/thay thế sớm để tránh rủi ro.';
                                                    if (p === 'recheck_required') return 'Chụp lại ảnh bằng drone để xác minh.';
                                                    if (p === 'healthy') return 'Bình thường, không cần xử lý.';
                                                    return d0.recommendation || '—';
                                                })()],
                                            ].map(([k, v]) => (
                                                <React.Fragment key={k}>
                                                    <span style={{ color: '#64748b', whiteSpace: 'nowrap' }}>{k}</span>
                                                    <span style={{ color: '#cbd5e1', fontWeight: 500 }}>{v}</span>
                                                </React.Fragment>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* ── RIGHT: Form ── */}
                            <div style={{ flex: 1, padding: '18px 22px', display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto', backgroundColor: '#0f172a' }}>

                                {/* Panel + class header */}
                                <div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                                        <span style={{ backgroundColor: '#38bdf8', color: '#0f172a', padding: '2px 10px', borderRadius: 6, fontSize: 11, fontWeight: 700 }}>
                                            TẤM PIN {currentItem.local_id}
                                        </span>
                                        <span style={{ color: '#64748b', fontSize: 12 }}>Hàng {currentItem.row} • Cột {currentItem.col}</span>
                                    </div>
                                    <h3 style={{ margin: 0, color: '#f8fafc', fontSize: 19, fontWeight: 700 }}>
                                        {DEFECT_CLASS_LABELS[d0.class_name] || currentItem.defects.map(d => d.type).join(', ')}
                                    </h3>
                                </div>

                                {/* Thermal validation (Removed as requested) */}

                                {/* ── Status cards ── */}
                                <div>
                                    <div style={labelStyle}>Đánh giá thủ công (chọn 1 trạng thái)</div>
                                    {showStatusWarning && (
                                        <div style={{ color: '#f59e0b', fontSize: 12, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                                            <AlertTriangle size={13} /> Vui lòng chọn trạng thái duyệt cho lỗi hiện tại.
                                        </div>
                                    )}
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                        {statusCards.map(card => {
                                            const isActive = selectedStatus === card.value;
                                            return (
                                                <button
                                                    key={card.value}
                                                    disabled={!hasItems}
                                                    onClick={() => { setSelectedStatus(card.value); setShowStatusWarning(false); }}
                                                    style={{
                                                        padding: '11px 14px', borderRadius: 12,
                                                        border: `1.5px solid ${isActive ? card.activeBorder : '#334155'}`,
                                                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                                        cursor: hasItems ? 'pointer' : 'not-allowed',
                                                        backgroundColor: isActive ? card.activeBg : '#1a2540',
                                                        color: isActive ? card.activeColor : '#94a3b8',
                                                        opacity: hasItems ? 1 : 0.5,
                                                        transition: 'all 0.15s',
                                                        textAlign: 'left',
                                                    }}
                                                >
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                        <span style={{ color: isActive ? card.activeColor : '#475569' }}>{card.icon}</span>
                                                        <div>
                                                            <div style={{ fontWeight: 600, fontSize: 13 }}>{card.label}</div>
                                                            <div style={{ fontSize: 11, opacity: 0.75 }}>{card.desc}</div>
                                                        </div>
                                                    </div>
                                                    {isActive && <Check size={15} style={{ color: card.activeColor, flexShrink: 0 }} />}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* ── Corrected class dropdown ── */}
                                <div>
                                    <label style={labelStyle}>Loại lỗi sau duyệt</label>
                                    <div style={{ color: '#64748b', fontSize: 11, marginBottom: 6 }}>
                                        Loại lỗi AI: <strong style={{ color: '#94a3b8' }}>{DEFECT_CLASS_LABELS[d0.class_name] || d0.class_name || '—'}</strong>
                                    </div>
                                    <select
                                        disabled={!hasItems}
                                        value={correctedClass}
                                        onChange={e => setCorrectedClass(e.target.value)}
                                        style={{ ...inputStyle }}
                                    >
                                        {DEFECT_CLASS_OPTIONS.map(o => (
                                            <option key={o.value} value={o.value}>{o.label}</option>
                                        ))}
                                    </select>
                                </div>

                                {/* ── Priority & Reviewer ── */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                                    <div>
                                        <label style={labelStyle}>Mức ưu tiên bảo trì</label>
                                        <select
                                            disabled={!hasItems}
                                            value={maintenancePriority}
                                            onChange={e => setMaintenancePriority(e.target.value)}
                                            style={{ ...inputStyle }}
                                        >
                                            {SEVERITY_OPTIONS.map(o => (
                                                <option key={o.value} value={o.value}>{o.label}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div>
                                        <label style={labelStyle}>Người duyệt</label>
                                        <input
                                            type="text"
                                            disabled={!hasItems}
                                            value={reviewerName}
                                            onChange={e => setReviewerName(e.target.value)}
                                            placeholder="Tên kỹ sư duyệt"
                                            style={{ ...inputStyle }}
                                        />
                                    </div>
                                </div>

                                {/* ── Review note ── */}
                                <div>
                                    <label style={labelStyle}>Ghi chú kiểm tra</label>
                                    <textarea
                                        disabled={!hasItems}
                                        value={reviewNote}
                                        onChange={e => setReviewNote(e.target.value)}
                                        placeholder="Nhập nhận xét kiểm tra, lý do bỏ qua, yêu cầu chụp lại hoặc đề xuất bảo trì..."
                                        style={{ ...inputStyle, height: 80, resize: 'none' }}
                                    />
                                </div>
                            </div>
                        </>
                    )}
                </div>

                {/* ── Footer ── */}
                {hasItems && (
                    <>
                        <div style={{ padding: '13px 24px', borderTop: '1px solid #1e293b', display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: '#0f172a', flexShrink: 0 }}>
                            <div style={{ color: '#94a3b8', fontSize: 13 }}>
                                Đang duyệt: <strong style={{ color: '#f8fafc' }}>{currentIndex + 1} / {items.length}</strong> lỗi
                                &nbsp;
                                <span style={{ color: '#38bdf8' }}>({reviewedCount} đã duyệt)</span>
                            </div>
                            <div style={{ display: 'flex', gap: 10 }}>
                                <button onClick={handlePrev} disabled={currentIndex === 0 || saving}
                                    style={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#cbd5e1', borderRadius: 10, padding: '8px 16px', fontSize: 13, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, opacity: currentIndex === 0 ? 0.45 : 1 }}>
                                    <ChevronLeft size={15} /> Trước
                                </button>
                                <button onClick={handleNext} disabled={currentIndex === items.length - 1 || saving}
                                    style={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#cbd5e1', borderRadius: 10, padding: '8px 16px', fontSize: 13, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, opacity: currentIndex === items.length - 1 ? 0.45 : 1 }}>
                                    Tiếp theo <ChevronRight size={15} />
                                </button>
                                <div style={{ width: 1, height: 30, backgroundColor: '#334155', margin: '0 2px', alignSelf: 'center' }} />
                                <button
                                    onClick={handleFinishReviewAndSync}
                                    disabled={saving || syncing}
                                    style={{ backgroundColor: allReviewed ? '#10b981' : '#0ea5e9', border: 'none', color: '#0f172a', borderRadius: 10, padding: '8px 20px', fontSize: 13, fontWeight: 700, cursor: (saving || syncing) ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: 6, opacity: (saving || syncing) ? 0.8 : 1 }}>
                                    <Save size={15} />
                                    {syncing ? 'Đang đồng bộ...' : allReviewed ? 'Hoàn tất review và đồng bộ' : 'Lưu tiến độ review'}
                                </button>
                            </div>
                        </div>
                        {syncError && (
                            <div style={{ padding: '8px 24px', backgroundColor: 'rgba(239,68,68,0.14)', borderTop: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5', fontSize: 13, textAlign: 'center' }}>
                                ⚠️ {syncError}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
