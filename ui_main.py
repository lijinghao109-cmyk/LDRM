"""
ui_main.py - PyQt6 GUI for LDRM (LocalDoc Research Manager).
Provides:
  - Drag-and-drop import zone for .docx files
  - Document list with title + category
  - Search bar with keyword highlighting
  - Re-cluster button
  - Document detail viewer with keywords
  - High-frequency term analysis
  - Automatic text summarization
"""

import os

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QLineEdit, QSplitter, QDialog, QTextEdit, QFileDialog,
    QMessageBox, QFrame, QScrollArea, QApplication, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QDragEnterEvent, QDropEvent,
    QTextCursor, QTextCharFormat,
)

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
# Similarity Comparison Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityDialog(QDialog):
    """Dialog for comparing similarity between two documents."""

    def __init__(self, all_docs: list[dict], current_doc: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚖️  文献相似度对比")
        self.resize(600, 400)
        self.setStyleSheet("""
            QDialog { background: #131c27; color: #d0dde8; }
            QComboBox { background: #1a2535; color: #d0dde8; border: 1px solid #2e4460;
                        border-radius: 4px; padding: 4px; }
            QLabel { color: #7eb8e0; font-size: 12px; }
            QTextEdit { background: #1a2535; color: #d0dde8; border: 1px solid #2e4460;
                        border-radius: 4px; font-size: 12px; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("当前文档：" + current_doc["title"]))

        layout.addWidget(QLabel("对比文档："))
        self.combo_docs = QListWidget()
        self.combo_docs.setMaximumHeight(150)
        for doc in all_docs:
            if doc["id"] != current_doc["id"]:
                item = QListWidgetItem(f"{doc['title']} [{doc['category']}]")
                item.setData(Qt.ItemDataRole.UserRole, doc["id"])
                self.combo_docs.addItem(item)
        layout.addWidget(self.combo_docs)

        btn_analyze = QPushButton("🔍  分析相似度")
        btn_analyze.clicked.connect(self._on_analyze)
        layout.addWidget(btn_analyze)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("相似度分析结果将显示在此…")
        layout.addWidget(self.result_text)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)

        self.current_doc = current_doc
        self.all_docs = all_docs

    def _on_analyze(self):
        if not self.combo_docs.currentItem():
            QMessageBox.information(self, "未选择", "请先选择一篇文献进行对比。")
            return

        doc_id = self.combo_docs.currentItem().data(Qt.ItemDataRole.UserRole)
        compare_doc = next((d for d in self.all_docs if d["id"] == doc_id), None)

        if not compare_doc:
            return

        content1 = self.current_doc.get("content", "") or ""
        content2 = compare_doc.get("content", "") or ""

        similarity = nlp.calculate_doc_similarity(content1, content2)
        similarity_percent = round(similarity * 100, 2)

        result = f"""
【对比结果】
当前文献：{self.current_doc['title']}
对比文献：{compare_doc['title']}

相似度：{similarity_percent}%

【相似度说明】
0-20%：完全不同
20-40%：关联度较低
40-60%：有一定关联
60-80%：关联度较高
80-100%：高度相似
"""
        self.result_text.setPlainText(result)


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
        self._search_term: str = ""
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

        btn_topics = QPushButton("🔬  主题分析")
        btn_topics.clicked.connect(self._on_analyze_topics)
        toolbar.addWidget(btn_topics)

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

        self.lbl_highfreq = QLabel("")
        self.lbl_highfreq.setWordWrap(True)
        self.lbl_highfreq.setStyleSheet("color: #d0d8c8; font-size: 12px; background: #182a34; border-radius: 4px; padding: 6px;")
        right_layout.addWidget(self.lbl_highfreq)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFixedHeight(148)
        self.summary_text.setStyleSheet("background: #171f2b; color: #d0dde8; border: 1px solid #2e4460; border-radius: 4px; font-size: 12px;")
        self.summary_text.setPlaceholderText("自动归纳要义将在此显示…")
        right_layout.addWidget(self.summary_text)

        self.lbl_topics = QLabel("")
        self.lbl_topics.setWordWrap(True)
        self.lbl_topics.setStyleSheet("color: #d8b8d8; font-size: 11px; background: #2a1a3a; border-radius: 4px; padding: 6px;")
        right_layout.addWidget(self.lbl_topics)

        btn_compare = QPushButton("⚖  文献相似度对比")
        btn_compare.clicked.connect(self._on_compare_similarity)
        right_layout.addWidget(btn_compare)

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

    def _load_documents(self, docs: list[dict] | None = None, select_first: bool = False):
        """Populate the list widget. Uses all DB docs if docs is None."""
        try:
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

            if select_first and docs:
                item = self.doc_list.item(0)
                if item:
                    self.doc_list.setCurrentItem(item)
                    self._on_doc_clicked(item)
            elif not docs:
                self._current_doc = None
                self.lbl_preview_title.setText("请选择文档进行预览")
                self.lbl_preview_meta.setText("")
                self.lbl_keywords.setText("")
                self.lbl_highfreq.setText("")
                self.lbl_topics.setText("")
                self.summary_text.clear()
                self.preview_text.clear()
        except Exception as e:
            self._set_status(f"Load docs error: {str(e)}")
            import traceback
            traceback.print_exc()

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
        self._search_term = keyword
        if not keyword:
            self._load_documents()
            self._set_status("搜索已清除")
            return
        try:
            print(f"[DEBUG] Searching for: {keyword}")
            results = db.search_documents(keyword)
            print(f"[DEBUG] Found {len(results)} results")
            self._load_documents(results, select_first=True)
            self._set_status(f"搜索 '{keyword}'：{len(results)} 个结果")
        except Exception as e:
            self._set_status(f"Search error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _on_clear_search(self):
        self._search_term = ""
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
        try:
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
            self._show_preview(doc, self._search_term)
        except Exception as e:
            self._set_status(f"Error loading doc: {str(e)}")
            import traceback
            traceback.print_exc()

    def _show_preview(self, doc: dict, search_term: str = ""):
        try:
            content = doc.get("content", "") or ""
            preview = content[:800] if not search_term else content
            self.preview_text.setPlainText(preview + ("…" if not search_term and len(content) > 800 else ""))
            self._render_doc_analysis(doc)
            self._highlight_search_matches(search_term)
        except Exception as e:
            self._set_status(f"Preview error: {str(e)}")
            self.preview_text.setPlainText("(无法加载内容)")
            import traceback
            traceback.print_exc()

    def _render_doc_analysis(self, doc: dict):
        try:
            content = doc.get("content", "") or ""
            highfreq = nlp.get_term_frequencies(content, top_n=10)
            if highfreq:
                terms = [f"{term}({count})" for term, count in highfreq]
                self.lbl_highfreq.setText("高频词：" + "  ·  ".join(terms))
            else:
                self.lbl_highfreq.setText("高频词：无可用数据")

            summary = nlp.summarize_text(content, max_sentences=3)
            self.summary_text.setPlainText(summary or "(未能自动归纳要义)")

            self.lbl_topics.setText("📌 (点击'主题分析'按钮查看主题)")
        except Exception as e:
            self._set_status(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _clear_preview_highlight(self):
        """Clear any previous highlighting by resetting the text."""
        pass  # No need to clear, just moving cursor is sufficient

    def _highlight_search_matches(self, term: str):
        """Highlight all occurrences of search term in preview text."""
        if not term:
            return
        try:
            document = self.preview_text.document()
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.MoveOperation.Start)

            position = 0
            first_found = False
            highlight_format = QTextCharFormat()
            highlight_format.setBackground(QColor("#ffe066"))
            highlight_format.setForeground(QColor("#000000"))

            while True:
                cursor = document.find(term, position)
                if cursor.isNull():
                    break
                cursor.mergeCharFormat(highlight_format)
                if not first_found:
                    self.preview_text.setTextCursor(cursor)
                    self.preview_text.ensureCursorVisible()
                    first_found = True
                position = cursor.position()

            if not first_found:
                self.preview_text.moveCursor(QTextCursor.MoveOperation.Start)
        except Exception as e:
            self._set_status(f"Highlight error: {str(e)}")

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

    def _on_analyze_topics(self):
        """Extract topics from all documents using LDA."""
        docs = db.get_all_documents()
        if not docs:
            QMessageBox.information(self, "暂无文档", "请先导入文档。")
            return

        self._set_status("正在分析主题…")

        def do_lda():
            texts = [doc.get("content", "") or "" for doc in docs]
            return nlp.extract_topics_lda(texts, num_topics=3)

        self._worker = WorkerThread(do_lda)
        self._worker.finished.connect(self._on_topics_analyzed)
        self._worker.error.connect(lambda e: self._set_status(f"Error: {e}"))
        self._worker.start()

    def _on_topics_analyzed(self, result: dict):
        if not result:
            self._set_status("主题分析失败（可能需要安装 gensim）")
            return

        topics = result.get("topics", [])
        if topics:
            topics_text = "\n".join(topics)
            QMessageBox.information(self, "📊 LDA 主题分析", topics_text)
            self._set_status(f"发现 {len(topics)} 个主题")

    def _on_compare_similarity(self):
        """Show a dialog to compare similarity between two documents."""
        if not self._current_doc:
            QMessageBox.information(self, "未选择文档", "请先点击一个文档。")
            return

        dialog = SimilarityDialog(self._docs, self._current_doc, self)
        dialog.exec()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)
        QApplication.processEvents()
