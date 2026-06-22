import React, { useState, useEffect } from 'react';
import { 
    X, ChevronLeft, ChevronRight, Check, AlertTriangle, AlertCircle, Trash2, Save
} from 'lucide-react';
import api, { fetchReviewItems, updateReviewItem, syncReview } from '../api';

export default function DefectReviewModal({ isOpen, onClose, batchId, onRefresh }) {
    const [items, setItems] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [hasFetchError, setHasFetchError] = useState(false);
    
    // Form fields for current item
    const [selectedStatus, setSelectedStatus] = useState('unreviewed');
    const [reviewNote, setReviewNote] = useState('');
    const [maintenancePriority, setMaintenancePriority] = useState('medium');
    const [reviewerName, setReviewerName] = useState('');
    const [reviewedAt, setReviewedAt] = useState('');

    // Sync state
    const [syncing, setSyncing] = useState(false);
    const [syncError, setSyncError] = useState('');

    // Fetch review items
    useEffect(() => {
        if (isOpen && batchId) {
            setLoading(true);
            setHasFetchError(false);
            fetchReviewItems(batchId)
                .then(res => {
                    const fetchedItems = res.data.items || [];
                    setItems(fetchedItems);
                    setCurrentIndex(0);
                    if (fetchedItems.length > 0) {
                        setSelectedStatus(fetchedItems[0].review_status || 'unreviewed');
                        setReviewNote(fetchedItems[0].review_note || '');
                        setMaintenancePriority(fetchedItems[0].maintenance_priority || 'medium');
                        setReviewerName(fetchedItems[0].reviewer_name || '');
                        setReviewedAt(fetchedItems[0].reviewed_at || '');
                    }
                    setLoading(false);
                })
                .catch(err => {
                    console.error("Failed to fetch review items:", err);
                    setHasFetchError(true);
                    setLoading(false);
                });
        }
    }, [isOpen, batchId]);

    // Update form when index changes — preselect dựa vào thermal validator nếu chưa review thủ công
    useEffect(() => {
        if (items.length > 0 && currentIndex < items.length) {
            const currentItem = items[currentIndex];
            const existingStatus = currentItem.review_status;
            if (existingStatus && existingStatus !== 'unreviewed') {
                // Đã review thủ công → giữ nguyên
                setSelectedStatus(existingStatus);
            } else {
                // Chưa review → preselect theo thermal validator suggestion
                const d0 = (currentItem.defects || [])[0];
                const suggested = d0?.suggested_review_status;
                // confirmed_defect → preselect "confirmed_defect"; needs_review → "needs_review"; khác → "unreviewed"
                if (suggested === 'confirmed_defect') {
                    setSelectedStatus('confirmed_defect');
                } else if (suggested === 'needs_review') {
                    setSelectedStatus('needs_review');
                } else {
                    setSelectedStatus('unreviewed');
                }
            }
            setReviewNote(currentItem.review_note || '');
            setMaintenancePriority(currentItem.maintenance_priority || 'medium');
            setReviewerName(currentItem.reviewer_name || '');
            setReviewedAt(currentItem.reviewed_at || '');
        }
    }, [currentIndex, items]);

    if (!isOpen) return null;

    const handleSave = async (silent = false) => {
        if (items.length === 0) return;
        const currentItem = items[currentIndex];
        
        let currentTimestamp = reviewedAt;
        if (selectedStatus !== 'unreviewed' && !currentTimestamp) {
            currentTimestamp = new Date().toLocaleString('vi-VN');
        } else if (selectedStatus === 'unreviewed') {
            currentTimestamp = '';
        }
        
        setSaving(true);
        try {
            await updateReviewItem(currentItem.review_item_id, {
                review_status: selectedStatus,
                review_note: reviewNote,
                maintenance_priority: maintenancePriority,
                reviewer_name: reviewerName || 'Chưa cập nhật',
                reviewed_at: currentTimestamp
            });
            
            // Update local items array
            const updatedItems = [...items];
            updatedItems[currentIndex] = {
                ...currentItem,
                review_status: selectedStatus,
                review_note: reviewNote,
                maintenance_priority: maintenancePriority,
                reviewer_name: reviewerName || 'Chưa cập nhật',
                reviewed_at: currentTimestamp
            };
            setItems(updatedItems);
            setReviewedAt(currentTimestamp);
            
            if (!silent) {
                // Trigger global refresh so map & sidebar counters update
                if (onRefresh) onRefresh();
            }
        } catch (err) {
            console.error("Failed to save review:", err);
            alert("Không thể lưu kết quả duyệt lỗi!");
        } finally {
            setSaving(false);
        }
    };

    const handleNext = async () => {
        await handleSave(true);
        if (currentIndex < items.length - 1) {
            setCurrentIndex(currentIndex + 1);
        } else {
            // Last item
            if (onRefresh) onRefresh();
            alert("Đã duyệt xong lỗi cuối cùng!");
        }
    };

    const handlePrev = async () => {
        await handleSave(true);
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
        }
    };

    const handleComplete = async () => {
        await handleSave(false);
        onClose();
    };

    const handleFinishReviewAndSync = async () => {
        setSyncing(true);
        setSyncError('');
        try {
            // 1. Lưu item hiện tại nếu có thay đổi chưa lưu
            await handleSave(true);

            // 2. Gọi backend sync
            await syncReview(batchId);

            // 3. Dispatch event để refresh tất cả trang
            window.dispatchEvent(new Event('review-sync-completed'));

            // 4. Trigger onRefresh callback ngay (để Bản đồ update)
            if (onRefresh) onRefresh();

            // 5. Đóng modal
            onClose();
        } catch (err) {
            console.error('Sync failed:', err);
            setSyncError('Đồng bộ thất bại. Vui lòng thử lại.');
        } finally {
            setSyncing(false);
        }
    };

    const currentItem = (items && items.length > 0 && currentIndex < items.length) ? items[currentIndex] : {
        local_id: '---',
        row: '—',
        col: '—',
        filename: 'Không có dữ liệu',
        defects: [{
            type: 'Không có lỗi',
            confidence: 0,
            area_ratio_percent: 0,
            class_name: 'healthy',
            thermal_validation_status: 'not_run'
        }],
        annotated_image_url: '',
        image_url: '',
        panel: {
            geometry_source: 'N/A'
        }
    };

    return (
        <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(3, 7, 18, 0.85)', backdropFilter: 'blur(12px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 9999, padding: 24, animation: 'fadeIn 0.2s ease-out'
        }}>
            <div style={{
                backgroundColor: '#0f172a', border: '1px solid #334155',
                width: '100%', maxWidth: 1000, height: '85vh', borderRadius: 24,
                display: 'flex', flexDirection: 'column', overflow: 'hidden',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
            }}>
                {/* Header */}
                <div style={{
                    padding: '16px 28px', borderBottom: '1px solid #1e293b',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    background: 'linear-gradient(90deg, #1e293b 0%, #0f172a 100%)'
                }}>
                    <div>
                        <h2 style={{ margin: 0, color: '#f8fafc', fontSize: 20, fontWeight: 700 }}>
                            Duyệt lỗi phát hiện (Manual Review)
                        </h2>
                        <p style={{ margin: '4px 0 0 0', color: '#94a3b8', fontSize: 13 }}>
                            Đợt kiểm tra #{batchId || '---'} • Xem xét và xác nhận các dị thường nhiệt phát hiện bởi AI
                        </p>
                    </div>
                    <button 
                        onClick={onClose}
                        style={{
                            background: 'transparent', border: 'none', color: '#94a3b8',
                            cursor: 'pointer', padding: 8, borderRadius: '50%',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#1e293b'; e.currentTarget.style.color = '#f8fafc'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; e.currentTarget.style.color = '#94a3b8'; }}
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Body Content */}
                <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
                    {loading ? (
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
                            <div className="spinner" style={{ marginRight: 12 }}></div>
                            Đang tải danh sách lỗi...
                        </div>
                    ) : hasFetchError ? (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', padding: 40 }}>
                            <AlertTriangle size={48} style={{ color: '#ef4444', marginBottom: 16 }} />
                            <h3 style={{ color: '#f8fafc', margin: '0 0 8px 0' }}>Lỗi tải dữ liệu kiểm tra</h3>
                            <p style={{ margin: 0, textAlign: 'center', maxWidth: 500, lineHeight: 1.5 }}>
                                Đợt kiểm tra #{batchId || '---'} không tìm thấy trên hệ thống (có thể DB đã được thiết lập lại hoặc cập nhật đợt mới).<br/>
                                Hãy <strong>nhấn F5 (Tải lại trang)</strong> để cập nhật phiên làm việc mới nhất.
                            </p>
                        </div>
                    ) : (
                        <>
                            {/* Left Area: Images and Highlights */}
                            <div style={{
                                flex: 1.3, padding: 24, borderRight: '1px solid #1e293b',
                                display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto',
                                backgroundColor: '#020617'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ color: '#38bdf8', fontSize: 13, fontWeight: 600 }}>
                                        ẢNH NHIỆT (THERMAL HIGHLIGHT)
                                    </span>
                                    <span style={{ color: '#94a3b8', fontSize: 12 }}>
                                        {currentItem.filename}
                                    </span>
                                </div>
                                
                                <div style={{
                                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                                    borderRadius: 16, overflow: 'hidden', border: '1px solid #1e293b',
                                    backgroundColor: '#090d16', position: 'relative', minHeight: 300, padding: 24
                                }}>
                                    {items && items.length > 0 ? (
                                        <>
                                            {/* Mặc định hiển thị ảnh thermal custom vẽ */}
                                            <img 
                                                src={`http://127.0.0.1:8000${currentItem.annotated_image_url}`} 
                                                alt="Thermal Highlight" 
                                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                                onError={(e) => {
                                                    // Fallback to precalib image
                                                    e.target.src = `http://127.0.0.1:8000${currentItem.image_url}`;
                                                }}
                                            />
                                            {currentItem.panel && (
                                                <div style={{
                                                    position: 'absolute', bottom: 12, left: 12,
                                                    backgroundColor: 'rgba(15, 23, 42, 0.8)',
                                                    padding: '4px 10px', borderRadius: 8, fontSize: 11, color: '#cbd5e1'
                                                }}>
                                                    AI Source: {currentItem.panel.geometry_source}
                                                </div>
                                            )}
                                        </>
                                    ) : (
                                        <div style={{ textAlign: 'center', color: '#94a3b8' }}>
                                            <Check size={48} style={{ color: '#10b981', marginBottom: 16, display: 'inline-block' }} />
                                            <h3 style={{ color: '#f8fafc', margin: '0 0 8px 0', fontSize: 16 }}>Không tìm thấy lỗi nào cần duyệt</h3>
                                            <p style={{ margin: 0, fontSize: 12, lineHeight: 1.5, maxWidth: 300 }}>
                                                Hệ thống không phát hiện tấm pin bị lỗi hoặc đợt kiểm tra chưa hoàn tất.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Right Area: Form & Metadata */}
                            <div style={{
                                flex: 1, padding: 24, display: 'flex', flexDirection: 'column',
                                gap: 20, overflowY: 'auto', backgroundColor: '#0f172a'
                            }}>
                                {/* Panel Details */}
                                <div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                                        <span style={{
                                            backgroundColor: '#38bdf8', color: '#0f172a',
                                            padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 700
                                        }}>
                                            TẤM PIN {currentItem.local_id}
                                        </span>
                                        <span style={{ color: '#64748b', fontSize: 12 }}>
                                            Hàng {currentItem.row} • Cột {currentItem.col}
                                        </span>
                                    </div>
                                    <h3 style={{ margin: 0, color: '#f8fafc', fontSize: 22, fontWeight: 700 }}>
                                        {currentItem.defects.map(d => d.type).join(', ')}
                                    </h3>
                                </div>

                                {/* AI Metrics */}
                                <div style={{
                                    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12,
                                    padding: 16, backgroundColor: '#1e293b', borderRadius: 16, border: '1px solid #334155'
                                }}>
                                    <div>
                                        <div style={{ color: '#94a3b8', fontSize: 11 }}>ĐỘ TIN CẬY AI</div>
                                        <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 600 }}>
                                            {currentItem.defects[0] ? `${roundValue(currentItem.defects[0].confidence * 100, 1)}%` : 'N/A'}
                                        </div>
                                    </div>
                                    <div>
                                        <div style={{ color: '#94a3b8', fontSize: 11 }}>HAO HỤT CÔNG SUẤT</div>
                                        <div style={{ color: '#f59e0b', fontSize: 15, fontWeight: 600 }}>
                                            {currentItem.defects[0] ? `${currentItem.defects[0].area_ratio_percent || 0.0} W` : '0.0 W'}
                                        </div>
                                    </div>
                                </div>

                                {/* Thermal Validation Section */}
                                {(() => {
                                    const d0 = currentItem.defects[0];
                                    if (!d0) return null;
                                    const tvStatus = d0.thermal_validation_status;
                                    if (!tvStatus || tvStatus === 'not_run') return (
                                        <div style={{
                                            padding: '10px 14px', backgroundColor: 'rgba(148,163,184,0.06)',
                                            border: '1px solid #1e293b', borderRadius: 12
                                        }}>
                                            <div style={{ color: '#64748b', fontSize: 11, fontWeight: 700, marginBottom: 2 }}>KIỂM CHỨNG NHIỆT TƯƠNG ĐỐI</div>
                                            <div style={{ color: '#475569', fontSize: 12 }}>Chưa chạy — chạy lại phân tích để có kết quả.</div>
                                        </div>
                                    );

                                    const statusMeta = {
                                        confirmed_by_relative_thermal: { label: 'Đã xác nhận bằng tương phản nhiệt', color: '#10b981', bg: 'rgba(16,185,129,0.10)', icon: '✔' },
                                        needs_review:                  { label: 'Cần xem xét thêm',                  color: '#f59e0b', bg: 'rgba(245,158,11,0.10)',  icon: '⚠' },
                                        class_mismatch:                { label: 'Sai loại lỗi (class mismatch)',    color: '#f97316', bg: 'rgba(249,115,22,0.10)',  icon: '⚠' },
                                        suspect_false_positive:        { label: 'Nghi ngờ nhận nhầm',               color: '#ef4444', bg: 'rgba(239,68,68,0.10)',   icon: '✗' },
                                        insufficient_pixels:           { label: 'Không đủ dữ liệu ảnh',             color: '#64748b', bg: 'rgba(100,116,139,0.10)', icon: '?' },
                                        not_run:                       { label: 'Chưa chạy',                        color: '#64748b', bg: 'rgba(100,116,139,0.10)', icon: '-' },
                                    };
                                    const sm = statusMeta[tvStatus] || statusMeta.not_run;
                                    const fmt = (v) => v != null ? Number(v).toFixed(3) : 'N/A';

                                    return (
                                        <div style={{
                                            padding: '12px 14px', backgroundColor: sm.bg,
                                            border: `1px solid ${sm.color}30`, borderRadius: 12
                                        }}>
                                            <div style={{ color: '#94a3b8', fontSize: 11, fontWeight: 700, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                                                KIỂM CHỨNG NHIỆT TƯƠNG ĐỐI
                                                <span style={{ color: sm.color, fontSize: 12 }}>{sm.icon} {sm.label}</span>
                                            </div>

                                            {tvStatus === 'suspect_false_positive' && (
                                                <div style={{
                                                    backgroundColor: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.35)',
                                                    borderRadius: 8, padding: '6px 10px', marginBottom: 8,
                                                    color: '#fca5a5', fontSize: 11, lineHeight: 1.5
                                                }}>
                                                    ⚠ Thuật toán tương phản nhiệt <strong>không đủ bằng chứng</strong> xác nhận lỗi này.<br/>
                                                    Đề xuất: <strong>Xem xét thủ công</strong> trước khi xác nhận.
                                                </div>
                                            )}

                                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px', fontSize: 12 }}>
                                                <div style={{ color: '#64748b' }}>YOLO class</div>
                                                <div style={{ color: '#cbd5e1', fontWeight: 600 }}>{d0.class_name || '—'}</div>

                                                <div style={{ color: '#64748b' }}>Rule class</div>
                                                <div style={{ color: d0.rule_class !== d0.class_name ? '#f97316' : '#cbd5e1', fontWeight: 600 }}>
                                                    {d0.rule_class || d0.class_name || '—'}
                                                    {d0.rule_class && d0.rule_class !== d0.class_name && ' ⚠'}
                                                </div>

                                                <div style={{ color: '#64748b' }}>Validation score</div>
                                                <div style={{ color: sm.color, fontWeight: 700 }}>
                                                    {d0.thermal_validation_score != null ? (d0.thermal_validation_score * 100).toFixed(0) + '%' : 'N/A'}
                                                </div>

                                                <div style={{ color: '#64748b' }}>Tương phản nóng (Δhot)</div>
                                                <div style={{ color: d0.relative_hot_delta >= 0.12 ? '#f87171' : '#94a3b8', fontWeight: 600 }}>
                                                    {fmt(d0.relative_hot_delta)}
                                                </div>

                                                <div style={{ color: '#64748b' }}>Blue suppression</div>
                                                <div style={{ color: d0.blue_suppression >= 0.06 ? '#fb923c' : '#94a3b8', fontWeight: 600 }}>
                                                    {fmt(d0.blue_suppression)}
                                                </div>

                                                <div style={{ color: '#64748b' }}>TNI (non-uniformity)</div>
                                                <div style={{ color: d0.tni >= 0.12 ? '#facc15' : '#94a3b8', fontWeight: 600 }}>
                                                    {fmt(d0.tni)}
                                                </div>

                                                <div style={{ color: '#64748b' }}>Mức tương phản nhiệt</div>
                                                <div style={{ color: d0.severity_by_relative_contrast === 'high' ? '#ef4444' : d0.severity_by_relative_contrast === 'medium' ? '#f59e0b' : '#94a3b8', fontWeight: 600 }}>
                                                    {d0.severity_by_relative_contrast ? { high: 'Cao', medium: 'Trung bình', low: 'Thấp' }[d0.severity_by_relative_contrast] : 'N/A'}
                                                </div>

                                                <div style={{ color: '#64748b' }}>Đề xuất tự động</div>
                                                <div style={{ color: '#38bdf8', fontWeight: 600 }}>
                                                    {d0.suggested_review_status === 'confirmed_defect' ? 'Đúng có lỗi' :
                                                     d0.suggested_review_status === 'needs_review' ? 'Xem xét' : 'N/A'}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })()}

                                {/* Status Options */}
                                <div>
                                    <div style={{ color: '#94a3b8', fontSize: 12, fontWeight: 600, marginBottom: 10 }}>
                                        ĐÁNH GIÁ THỦ CÔNG (CHỌN 1 TRẠNG THÁI)
                                    </div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                        {/* Đã xác nhận */}
                                        <button 
                                            disabled={items.length === 0}
                                            onClick={() => setSelectedStatus('confirmed_defect')}
                                            style={{
                                                padding: '12px 16px', borderRadius: 12, border: '1px solid',
                                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                                cursor: items.length === 0 ? 'not-allowed' : 'pointer',
                                                borderColor: items.length === 0 ? '#334155' : (selectedStatus === 'confirmed_defect' ? '#ef4444' : '#334155'),
                                                backgroundColor: items.length === 0 ? '#1e293b' : (selectedStatus === 'confirmed_defect' ? 'rgba(239, 68, 68, 0.15)' : '#1e293b'),
                                                color: items.length === 0 ? '#64748b' : (selectedStatus === 'confirmed_defect' ? '#fca5a5' : '#94a3b8'),
                                                opacity: items.length === 0 ? 0.4 : 1
                                            }}
                                        >
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                <AlertCircle size={16} style={{ color: items.length === 0 ? '#64748b' : '#ef4444' }} />
                                                <div>
                                                    <div style={{ fontWeight: 600, fontSize: 14 }}>Đã xác nhận</div>
                                                    <div style={{ fontSize: 11, opacity: 0.8 }}>Xác nhận tấm pin có bất thường nhiệt này</div>
                                                </div>
                                            </div>
                                            {(items.length > 0 && selectedStatus === 'confirmed_defect') && <Check size={16} style={{ color: '#ef4444' }} />}
                                        </button>

                                        {/* Cần kiểm tra lại */}
                                        <button 
                                            disabled={items.length === 0}
                                            onClick={() => setSelectedStatus('needs_review')}
                                            style={{
                                                padding: '12px 16px', borderRadius: 12, border: '1px solid',
                                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                                cursor: items.length === 0 ? 'not-allowed' : 'pointer',
                                                borderColor: items.length === 0 ? '#334155' : (selectedStatus === 'needs_review' ? '#f59e0b' : '#334155'),
                                                backgroundColor: items.length === 0 ? '#1e293b' : (selectedStatus === 'needs_review' ? 'rgba(245, 158, 11, 0.15)' : '#1e293b'),
                                                color: items.length === 0 ? '#64748b' : (selectedStatus === 'needs_review' ? '#fde047' : '#94a3b8'),
                                                opacity: items.length === 0 ? 0.4 : 1
                                            }}
                                        >
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                <AlertTriangle size={16} style={{ color: items.length === 0 ? '#64748b' : '#f59e0b' }} />
                                                <div>
                                                    <div style={{ fontWeight: 600, fontSize: 14 }}>Cần kiểm tra lại</div>
                                                    <div style={{ fontSize: 11, opacity: 0.8 }}>Cần khảo sát hiện trường thực tế để xác nhận</div>
                                                </div>
                                            </div>
                                            {(items.length > 0 && selectedStatus === 'needs_review') && <Check size={16} style={{ color: '#f59e0b' }} />}
                                        </button>

                                        {/* Bỏ qua */}
                                        <button 
                                            disabled={items.length === 0}
                                            onClick={() => setSelectedStatus('false_positive')}
                                            style={{
                                                padding: '12px 16px', borderRadius: 12, border: '1px solid',
                                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                                cursor: items.length === 0 ? 'not-allowed' : 'pointer',
                                                borderColor: items.length === 0 ? '#334155' : (selectedStatus === 'false_positive' ? '#94a3b8' : '#334155'),
                                                backgroundColor: items.length === 0 ? '#1e293b' : (selectedStatus === 'false_positive' ? 'rgba(148, 163, 184, 0.15)' : '#1e293b'),
                                                color: items.length === 0 ? '#64748b' : (selectedStatus === 'false_positive' ? '#e2e8f0' : '#94a3b8'),
                                                opacity: items.length === 0 ? 0.4 : 1
                                            }}
                                        >
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                                <Trash2 size={16} style={{ color: items.length === 0 ? '#64748b' : '#94a3b8' }} />
                                                <div>
                                                    <div style={{ fontWeight: 600, fontSize: 14 }}>Bỏ qua (False Positive)</div>
                                                    <div style={{ fontSize: 11, opacity: 0.8 }}>AI nhận nhầm (phản xạ nhiệt, bụi bẩn, bóng che...)</div>
                                                </div>
                                            </div>
                                            {(items.length > 0 && selectedStatus === 'false_positive') && <Check size={16} style={{ color: '#94a3b8' }} />}
                                        </button>
                                    </div>
                                </div>

                                {/* O&M fields: Priority & Reviewer */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                                    <div>
                                        <label style={{ display: 'block', color: '#94a3b8', fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                                            ƯU TIÊN XỬ LÝ
                                        </label>
                                        <select
                                            disabled={items.length === 0}
                                            value={maintenancePriority}
                                            onChange={(e) => setMaintenancePriority(e.target.value)}
                                            style={{
                                                width: '100%', backgroundColor: '#1e293b',
                                                border: '1px solid #334155', borderRadius: 12, padding: '10px 14px',
                                                color: items.length === 0 ? '#64748b' : '#cbd5e1', fontSize: 13, outline: 'none',
                                                cursor: items.length === 0 ? 'not-allowed' : 'default',
                                                opacity: items.length === 0 ? 0.5 : 1
                                            }}
                                        >
                                            <option value="low">Thấp</option>
                                            <option value="medium">Trung bình</option>
                                            <option value="high">Cao</option>
                                            <option value="urgent">Khẩn cấp</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', color: '#94a3b8', fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                                            NGƯỜI DUYỆT
                                        </label>
                                        <input 
                                            type="text"
                                            disabled={items.length === 0}
                                            value={reviewerName}
                                            onChange={(e) => setReviewerName(e.target.value)}
                                            placeholder="Tên kỹ sư duyệt"
                                            style={{
                                                width: '100%', backgroundColor: '#1e293b',
                                                border: '1px solid #334155', borderRadius: 12, padding: '10px 14px',
                                                color: items.length === 0 ? '#64748b' : '#cbd5e1', fontSize: 13, outline: 'none', boxSizing: 'border-box',
                                                cursor: items.length === 0 ? 'not-allowed' : 'text',
                                                opacity: items.length === 0 ? 0.5 : 1
                                            }}
                                        />
                                    </div>
                                </div>

                                {/* Review Note */}
                                <div>
                                    <label style={{ display: 'block', color: '#94a3b8', fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                                        GHI CHÚ KIỂM TRA (TÙY CHỌN)
                                    </label>
                                    <textarea 
                                        disabled={items.length === 0}
                                        value={reviewNote}
                                        onChange={(e) => setReviewNote(e.target.value)}
                                        placeholder="Nhập lý do phản xạ, yêu cầu kiểm tra kỹ hơn, hoặc ghi chú bảo trì..."
                                        style={{
                                            width: '100%', height: 75, backgroundColor: '#1e293b',
                                            border: '1px solid #334155', borderRadius: 12, padding: '10px 14px',
                                            color: items.length === 0 ? '#64748b' : '#cbd5e1', fontSize: 13, resize: 'none', outline: 'none',
                                            cursor: items.length === 0 ? 'not-allowed' : 'text',
                                            opacity: items.length === 0 ? 0.5 : 1
                                        }}
                                    />
                                </div>
                            </div>
                        </>
                    )}
                </div>

                {/* Footer Navigation */}
                {items.length > 0 && (
                    <>
                    <div style={{
                        padding: '16px 28px', borderTop: '1px solid #1e293b',
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        backgroundColor: '#0f172a'
                    }}>
                        {/* Progress */}
                        <div style={{ color: '#94a3b8', fontSize: 13 }}>
                            Đang duyệt: <strong style={{ color: '#f8fafc' }}>{currentIndex + 1} / {items.length}</strong> lỗi
                        </div>

                        {/* Navigation controls */}
                        <div style={{ display: 'flex', gap: 12 }}>
                            <button
                                onClick={handlePrev}
                                disabled={currentIndex === 0 || saving}
                                style={{
                                    backgroundColor: '#1e293b', border: '1px solid #334155', color: '#cbd5e1',
                                    borderRadius: 12, padding: '8px 16px', fontSize: 13, cursor: 'pointer',
                                    display: 'flex', alignItems: 'center', gap: 6, opacity: currentIndex === 0 ? 0.5 : 1
                                }}
                            >
                                <ChevronLeft size={16} /> Trước
                            </button>
                            <button
                                onClick={handleNext}
                                disabled={currentIndex === items.length - 1 || saving}
                                style={{
                                    backgroundColor: '#1e293b', border: '1px solid #334155', color: '#cbd5e1',
                                    borderRadius: 12, padding: '8px 16px', fontSize: 13, cursor: 'pointer',
                                    display: 'flex', alignItems: 'center', gap: 6, opacity: currentIndex === items.length - 1 ? 0.5 : 1
                                }}
                            >
                                Tiếp theo <ChevronRight size={16} />
                            </button>
                            
                            <div style={{ width: 1, height: 28, backgroundColor: '#334155', margin: '0 4px' }} />

                            <button
                                onClick={handleFinishReviewAndSync}
                                disabled={saving || syncing}
                                style={{
                                    backgroundColor: syncing ? '#059669' : '#10b981',
                                    border: 'none', color: '#0f172a',
                                    borderRadius: 12, padding: '8px 20px', fontSize: 13, fontWeight: 700,
                                    cursor: (saving || syncing) ? 'not-allowed' : 'pointer',
                                    display: 'flex', alignItems: 'center', gap: 6,
                                    opacity: (saving || syncing) ? 0.85 : 1,
                                    transition: 'all 0.2s'
                                }}
                            >
                                <Save size={16} />
                                {syncing ? 'Đang đồng bộ...' : 'Hoàn tất review và đồng bộ'}
                            </button>
                        </div>
                    </div>
                    {syncError && (
                        <div style={{
                            padding: '8px 28px', backgroundColor: 'rgba(239,68,68,0.15)',
                            borderTop: '1px solid rgba(239,68,68,0.3)',
                            color: '#fca5a5', fontSize: 13, textAlign: 'center'
                        }}>
                            ⚠️ {syncError}
                        </div>
                    )}
                    </>
                )}
            </div>
        </div>
    );
}

function roundValue(value, decimals) {
    if (value === undefined || value === null) return '0';
    return Number(value).toFixed(decimals);
}
