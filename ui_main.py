"""
ui_main.py - PyQt6 GUI for LDRM (LocalDoc Research Manager).
Provides:
  - Drag-and-drop import zone for .docx files
  - Document list with title + category
  - Search bar
  - Re-cluster button
  - Document detail viewer with keywords
  - Keyword statistics chart (matplotlib)
  - CSV export
"""

import os
import csv

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QLineEdit, QSplitter, QDialog, QTextEdit, QFileDialog,
    QMessageBox, QFrame, QScrollArea, QApplication, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QFont, QColor, QPalette, QDragEnterEvent, QDropEvent

import db
import doc_parser
import nlp
import classifier

# ── Optional: matplotlib for keyword chart ───────────────────────────────────
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Background worker thread for long-running operations
# ─────────────────────────────────────────────────────────────────────────────

class WorkerThread(QThread):
    """Generic worker thread to keep the GUI responsive."""
    finished = pyqtSignal(object)   # emits the result
    error    = pyqtSignal(str)      # emits an error message

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Drop Zone widget
# ─────────────────────────────────────────────────────────────────────────────

class DropZone(QLabel):
    """A label that accepts dragged .docx files and emits their paths."""
    files_dropped = pyqtSignal(list)  # list of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setText("📂 拖放 .docx 文件到此处\n（或点击浏览）")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(90)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #5a9fd4;
                border-radius: 8px;
                background: #1e2b3a;
                color: #7eb8e0;
                font-size: 13px;
                padding: 10px;
            }
            QLabel:hover {
                background: #253447;
                border-color: #7eb8e0;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        """Allow clicking the zone to open a file dialog."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .docx Files", "", "Word Documents (*.docx)"
        )
        if paths:
            self.files_dropped.emit(paths)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            docx_present = any(
                u.toLocalFile().lower().endswith(".docx")
                for u in event.mimeData().urls()
            )
            if docx_present:
                event.acceptProposedAction()
                self.setStyleSheet(self.styleSheet().replace("#1e2b3a", "#1a3a5a"))
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace("#1a3a5a", "#1e2b3a"))

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self.styleSheet().replace("#1a3a5a", "#1e2b3a"))
        paths = [
            u.toLocalFile()
            for u in event.mimeData().urls()
            if u.toLocalFile().lower().endswith(".docx")
        ]
        if paths:
            self.files_dropped.emit(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Document Detail Dialog
# ─────────────────────────────────────────────────────────────────────────────

class DocDetailDialog(QDialog):
    """Modal dialog showing document content and extracted keywords."""

    def __init__(self, doc: dict, keywords: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"📄 {doc['title']}")
        self.resize(720, 560)
        self.setStyleSheet("""
            QDialog { background: #131c27; color: #d0dde8; }
            QTextEdit { background: #1a2535; color: #d0dde8; border: 1px solid #2e4460;
                        border-radius: 4px; font-size: 13px; }
            QLabel { color: #7eb8e0; font-size: 12px; }
            QPushButton { background: #2e5c8e; color: white; border: none;
                          border-radius: 4px; padding: 6px 16px; }
            QPushButton:hover { background: #3a72b0; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Meta info row
        meta_row = QHBoxLayout()
        meta_row.addWidget(QLabel(f"📁 {doc['file_path']}"))
        meta_row.addStretch()
        meta_row.addWidget(QLabel(f"🗂 {doc['category']}"))
        meta_row.addWidget(QLabel(f"🕐 {doc['created_at'][:19]}"))
        layout.addLayout(meta_row)

        # Keywords
        kw_label = QLabel("🔑 关键词：" + ("  ·  ".join(keywords) if keywords else "无"))
        kw_label.setWordWrap(True)
        kw_label.setStyleSheet("color: #a8d8a8; font-size: 12px; background: #1a3020;"
                               "border-radius: 4px; padding: 6px;")
        layout.addWidget(kw_label)

        # Content
        content_edit = QTextEdit()
        content_edit.setPlainText(doc.get("content", "") or "(未提取到内容)")
        content_edit.setReadOnly(True)
        layout.addWidget(content_edit)

        # Close button
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)

        # Optional keyword chart
        if MATPLOTLIB_AVAILABLE and keywords:
            self._add_chart(layout, keywords)

    def _add_chart(self, layout, keywords: list[str]):
        """Add a horizontal bar chart of the top keywords."""
        chart_label = QLabel("📊 关键词频率排名 (TF-IDF 得分)")
        chart_label.setStyleSheet("color: #7eb8e0; font-weight: bold; margin-top: 8px;")
        layout.addWidget(chart_label)

        fig = Figure(figsize=(6, 2.5), facecolor="#131c27")
        ax = fig.add_subplot(111, facecolor="#1a2535")
        top = keywords[:10]
        scores = list(range(len(top), 0, -1))  # proxy: rank as score
        ax.barh(top, scores, color="#3a72b0", edgecolor="#5a9fd4")
        ax.tick_params(colors="#d0dde8", labelsize=9)
        ax.spines[:].set_color("#2e4460")
        ax.set_xlabel("Relative Rank", color="#7eb8e0", fontsize=9)
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(200)
        layout.addWidget(canvas)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Primary application window for LDRM."""

    DARK_STYLE = """
        QMainWindow, QWidget { background: #0f1923; color: #c8d8e8; }
        QListWidget { background: #131c27; border: 1px solid #2e4460;
                      border-radius: 4px; font-size: 13px; outline: none; }
        QListWidget::item { padding: 8px 10px; border-bottom: 1px solid #1e2e40; }
        QListWidget::item:selected { background: #1e4060; color: #ffffff; }
        QListWidget::item:hover:!selected { background: #1a2f45; }
        QLineEdit { background: #131c27; border: 1px solid #2e4460;
                    border-radius: 4px; padding: 6px 10px; color: #d0dde8; font-size: 13px; }
        QLineEdit:focus { border-color: #5a9fd4; }
        QPushButton {
            background: #1e4060; color: #c8dff0; border: 1px solid #2e5c8e;
            border-radius: 4px; padding: 7px 16px; font-size: 13px;
        }
        QPushButton:hover  { background: #2a5580; border-color: #5a9fd4; }
        QPushButton:pressed { background: #163050; }
        QPushButton#btnImport { background: #1a4a2e; border-color: #2e7a4e; color: #a8d8b8; }
        QPushButton#btnImport:hover { background: #226040; }
        QPushButton#btnExport { background: #3a2a0a; border-color: #7a5a1a; color: #d8b870; }
        QPushButton#btnExport:hover { background: #4a3a12; }
        QLabel#statusBar { color: #5a7a9a; font-size: 11px; padding: 3px 0; }
        QSplitter::handle { background: #2e4460; width: 1px; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalDoc 研究管理器 (LDRM)")
        self.resize(1100, 700)
        self.setStyleSheet(self.DARK_STYLE)
        self._docs: list[dict] = []       # currently displayed documents
        self._worker: WorkerThread | None = None
        self._setup_ui()
        self._load_documents()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        # ── Title bar ──
        title_row = QHBoxLayout()
        title_lbl = QLabel("📚  LocalDoc 研究管理器")
        title_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #7eb8e0;")
        title_row.addWidget(title_lbl)
        title_row.addStretch()

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("statusBar")
        title_row.addWidget(self.lbl_status)
        root.addLayout(title_row)

        # ── Drop zone ──
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        root.addWidget(self.drop_zone)

        # ── Toolbar ──
        toolbar = QHBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍  按标题或内容搜索…")
        self.search_box.returnPressed.connect(self._on_search)
        toolbar.addWidget(self.search_box, stretch=3)

        btn_search = QPushButton("搜索")
        btn_search.clicked.connect(self._on_search)
        toolbar.addWidget(btn_search)

        btn_refresh = QPushButton("↻  刷新 / 重新聚类")
        btn_refresh.clicked.connect(self._on_recluster)
        toolbar.addWidget(btn_refresh)

        btn_export = QPushButton("⬇  导出 CSV")
        btn_export.setObjectName("btnExport")
        btn_export.clicked.connect(self._on_export_csv)
        toolbar.addWidget(btn_export)

        btn_clear = QPushButton("✕  清除搜索")
        btn_clear.clicked.connect(self._on_clear_search)
        toolbar.addWidget(btn_clear)

        root.addLayout(toolbar)

        # ── Splitter: list + info panel ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: document list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        count_row = QHBoxLayout()
        self.lbl_count = QLabel("Documents: 0")
        self.lbl_count.setStyleSheet("color: #5a8aaa; font-size: 11px;")
        count_row.addWidget(self.lbl_count)
        count_row.addStretch()
        left_layout.addLayout(count_row)

        self.doc_list = QListWidget()
        self.doc_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.doc_list.itemDoubleClicked.connect(self._on_doc_double_clicked)
        self.doc_list.itemClicked.connect(self._on_doc_clicked)
        left_layout.addWidget(self.doc_list)

        splitter.addWidget(left_panel)

        # Right: quick preview panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)

        self.lbl_preview_title = QLabel("请选择文档进行预览")
        self.lbl_preview_title.setStyleSheet("color: #7eb8e0; font-size: 13px; font-weight: bold;")
        self.lbl_preview_title.setWordWrap(True)
        right_layout.addWidget(self.lbl_preview_title)

        self.lbl_preview_meta = QLabel("")
        self.lbl_preview_meta.setStyleSheet("color: #5a8aaa; font-size: 11px;")
        self.lbl_preview_meta.setWordWrap(True)
        right_layout.addWidget(self.lbl_preview_meta)

        self.lbl_keywords = QLabel("")
        self.lbl_keywords.setWordWrap(True)
        self.lbl_keywords.setStyleSheet("color: #a8d8a8; font-size: 12px; "
                                        "background: #1a3020; border-radius: 4px; padding: 6px;")
        right_layout.addWidget(self.lbl_keywords)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("内容预览将显示在此处…")
        right_layout.addWidget(self.preview_text)

        btn_open = QPushButton("🔎  打开全文视图")
        btn_open.clicked.connect(self._on_open_full_view)
        right_layout.addWidget(btn_open)

        splitter.addWidget(right_panel)
        splitter.setSizes([380, 680])
        root.addWidget(splitter)

        # Status bar footer
        self.lbl_footer = QLabel("")
        self.lbl_footer.setObjectName("statusBar")
        root.addWidget(self.lbl_footer)

        self._current_doc: dict | None = None

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_documents(self, docs: list[dict] | None = None):
        """Populate the list widget. Uses all DB docs if docs is None."""
        if docs is None:
            docs = db.get_all_documents()
        self._docs = docs
        self.doc_list.clear()
        for doc in docs:
            item = QListWidgetItem(f"  {doc['title']}\n  [{doc['category']}]")
            item.setData(Qt.ItemDataRole.UserRole, doc["id"])
            # Color-code by cluster
            cluster_colors = {
                "簇 1": "#2e5070",
                "簇 2": "#2e5040",
                "簇 3": "#503020",
                "簇 4": "#502050",
                "簇 5": "#305040",
            }
            bg = cluster_colors.get(doc["category"], "#1e2e3a")
            item.setBackground(QColor(bg))
            self.doc_list.addItem(item)

        self.lbl_count.setText(f"文档：{len(docs)}")
        self.lbl_footer.setText(f"已加载 {len(docs)} 条文档")

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _on_files_dropped(self, paths: list[str]):
        """Import dropped/selected .docx files into the system."""
        self._set_status("Importing…")
        imported = 0
        skipped = 0
        errors = []

        for path in paths:
            try:
                title, local_path, content = doc_parser.import_docx(path)
                row_id = db.insert_document(title, local_path, content)
                if row_id == -1:
                    skipped += 1
                else:
                    imported += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        msg_parts = [f"已导入：{imported}", f"已跳过（重复）：{skipped}"]
        if errors:
            msg_parts.append(f"错误：{len(errors)}")
        self._set_status("  |  ".join(msg_parts))
        self._load_documents()

        if errors:
            QMessageBox.warning(self, "导入错误",
                                "部分文件导入失败：\n" + "\n".join(errors))

    def _on_search(self):
        keyword = self.search_box.text().strip()
        if not keyword:
            self._load_documents()
            return
        results = db.search_documents(keyword)
        self._load_documents(results)
        self._set_status(f"搜索 '{keyword}'：{len(results)} 个结果")

    def _on_clear_search(self):
        self.search_box.clear()
        self._load_documents()
        self._set_status("搜索已清除")

    def _on_recluster(self):
        """Run KMeans clustering in a background thread."""
        docs = db.get_all_documents()
        if not docs:
            QMessageBox.information(self, "暂无文档",
                                    "请先导入文档。")
            return
        self._set_status("正在重新聚类…")

        def do_cluster():
            return classifier.cluster_documents()

        self._worker = WorkerThread(do_cluster)
        self._worker.finished.connect(self._on_cluster_done)
        self._worker.error.connect(lambda e: self._set_status(f"Error: {e}"))
        self._worker.start()

    def _on_cluster_done(self, result: dict):
        count = len(result)
        self._set_status(f"聚类完成 — {count} 个文档已分类")
        self._load_documents()

    def _on_doc_clicked(self, item: QListWidgetItem):
        """Show a quick preview in the right panel."""
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        doc = db.get_document_by_id(doc_id)
        if not doc:
            return
        self._current_doc = doc
        self.lbl_preview_title.setText(doc["title"])
        self.lbl_preview_meta.setText(
            f"类别：{doc['category']}   |   导入时间：{doc['created_at'][:19]}"
        )
        keywords = nlp.extract_keywords(doc.get("content", "") or "", top_n=10)
        kw_text = "关键词：" + ("  ·  ".join(keywords) if keywords else "无")
        self.lbl_keywords.setText(kw_text)
        preview = (doc.get("content") or "")[:800]
        self.preview_text.setPlainText(preview + ("…" if len(doc.get("content", "")) > 800 else ""))

    def _on_doc_double_clicked(self, item: QListWidgetItem):
        self._on_doc_clicked(item)
        self._on_open_full_view()

    def _on_open_full_view(self):
        if not self._current_doc:
            QMessageBox.information(self, "未选择文档", "请先点击一个文档。")
            return
        doc = self._current_doc
        keywords = nlp.extract_keywords(doc.get("content", "") or "", top_n=15)
        dialog = DocDetailDialog(doc, keywords, self)
        dialog.exec()

    def _on_export_csv(self):
        """Export all currently displayed documents to a CSV file."""
        if not self._docs:
            QMessageBox.information(self, "无可导出项",
                                    "当前没有可导出的文档，请先导入文档。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", "ldrm_export.csv", "CSV 文件 (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["id", "title", "category", "file_path", "created_at", "content"]
                )
                writer.writeheader()
                for doc in self._docs:
                    writer.writerow({k: doc.get(k, "") for k in writer.fieldnames})
            self._set_status(f"已导出 {len(self._docs)} 条记录 → {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)
        QApplication.processEvents()
