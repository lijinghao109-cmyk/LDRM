# LocalDoc 研究管理器 (LDRM)

这是一个本地桌面应用，用于管理、搜索、分类和分析 `.docx` 研究文档。基于 Python + PyQt6 开发，全部数据保存在本机，不依赖云服务、账号或网络。

---

## 📁 项目结构

```
LDRM/
├── main.py          ← 应用入口
├── ui_main.py       ← PyQt6 界面实现
├── db.py            ← SQLite 数据库层
├── doc_parser.py    ← .docx 导入与文本提取
├── nlp.py           ← TF-IDF 关键词提取 (jieba)
├── classifier.py    ← KMeans 文档聚类
├── requirements.txt ← Python 依赖
├── README.md        ← 使用说明
├── build_mac.sh     ← macOS 打包脚本
├── build_windows.bat← Windows 打包脚本
└── data/
    ├── docs/        ← 导入的 .docx 文件保存目录
    └── app.db       ← SQLite 数据库（首次运行自动创建）
```

---

## ⚙️ 运行指南

### 1. 安装 Python 3.10+

请先安装 Python 3.10 及以上版本。macOS 推荐使用官方安装包或 Homebrew，Windows 推荐使用官方安装器并勾选“Add Python to PATH”。

### 2. 创建并激活虚拟环境（推荐）

macOS/Linux：
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows：
```bat
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动应用

```bash
python main.py
```

---

## 🚀 功能说明

- **导入文档**：将 `.docx` 文件拖放到导入区域，或点击导入区域进行选择。
- **查看文档**：单击文档列表查看预览，双击打开全文视图并显示关键词。
- **搜索**：在搜索框输入关键词，回车或点击“搜索”按钮。
- **重新聚类**：点击“↻ 刷新 / 重新聚类”按钮，自动对当前文档进行 KMeans 聚类。
- **导出 CSV**：点击“⬇ 导出 CSV”将当前文档列表保存为表格文件。
- **关键词图表**：在全文视图中显示关键词统计图（需要 matplotlib）。

---

## 🧭 macOS 独立应用

已打包生成 macOS 独立应用：

- `dist/LDRM.app`

直接双击 `dist/LDRM.app` 即可打开。

如果需要重新打包，请运行：

```bash
./build_mac.sh
```

---

## 🪟 Windows 独立应用

当前仓库包含 Windows 打包脚本：

- `build_windows.bat`

在 Windows 环境中执行该脚本即可生成 `dist\LDRM.exe`。

打包前请先在 Windows 上安装 Python、虚拟环境，并运行：

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt pyinstaller
build_windows.bat
```

---

## 🔧 常见问题

**无法启动，提示缺少 `PyQt6`**

请执行：

```bash
pip install PyQt6
```

**导入 `.docx` 失败，或出现重复导入**

请确认文件路径是否已存在。重复路径的文件会自动跳过。

**jieba 首次运行打印编译信息**

这是正常现象，jieba 首次运行时会生成词典，后续运行即可静默。

**数据库损坏或想重置**

删除 `data/app.db` 后重启应用即可重新建立数据库。原始 `.docx` 文件保存在 `data/docs/` 下，不会被删除。

---

## 📦 依赖说明

- `PyQt6`：图形用户界面框架
- `python-docx`：读取 `.docx` 文件
- `scikit-learn`：TF-IDF 向量化与 KMeans 聚类
- `jieba`：中文分词与关键词提取
- `matplotlib`：关键词图表显示（可选）
- `numpy`：数值计算库

---

## 📝 说明

- 本应用所有数据仅保存在本地。
- 文档内容、聚类结果和导出文件都不会上传到任何服务器。
- 默认聚类数量为 `k=3`，如需调整请修改 `classifier.py` 中的 `DEFAULT_K`。
