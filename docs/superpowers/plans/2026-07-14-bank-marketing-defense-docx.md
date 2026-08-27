# Bank Marketing Defense DOCX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份适合机器学习初学者练习和现场答辩的详细 Word 答辩稿。

**Architecture:** 使用 python-docx 从已确认设计、实际脚本和模型指标生成单一 DOCX。文档采用 compact_reference_guide 预设与 editorial_cover 首页，正文由逐字稿、现场提示、结果表、问答和速查页组成；生成后进行结构审计、渲染和逐页视觉检查。

**Tech Stack:** Python、python-docx、OOXML、LibreOffice、Poppler。

## Global Constraints

- 最终文件：`D:\UniFiles\Python_ML\OtherFiles\银行营销分类实验答辩稿.docx`。
- 不修改实验代码、数据集或实验报告。
- 所有实验数字必须与 `model_metrics.csv` 和实际终端输出一致。
- 第三个实验场景必须表述为“保留 duration 的随机森林”，不能称为第三种模型。
- 采用第一人称可朗读正文，并将现场操作提示与朗读文字在视觉上区分。
- 最终必须渲染并检查全部页面。

---

### Task 1: 创建生成脚本与正式内容

**Files:**
- Create: `D:\UniFiles\Python_ML\.codex_tmp\build_defense_docx.py`
- Create: `D:\UniFiles\Python_ML\OtherFiles\银行营销分类实验答辩稿.docx`

**Interfaces:**
- Consumes: 已确认设计、实际脚本结构、模型指标和现有图表。
- Produces: 含逐字稿、提示、问答和速查页的 DOCX。

- [ ] **Step 1:** 编写完整正文、样式、页眉页脚、表格和页面分隔逻辑。
- [ ] **Step 2:** 使用工作区自带 Python 运行生成脚本。
- [ ] **Step 3:** 检查 DOCX 存在、可打开且没有占位符。

### Task 2: 渲染并逐页检查

**Files:**
- Inspect: `D:\UniFiles\Python_ML\OtherFiles\银行营销分类实验答辩稿.docx`
- Create: `D:\UniFiles\Python_ML\.codex_tmp\defense_docx_render\page-*.png`

**Interfaces:**
- Consumes: Task 1 的 DOCX。
- Produces: 页面 PNG、页数和视觉检查结论。

- [ ] **Step 1:** 使用文档渲染器生成全部页面 PNG。
- [ ] **Step 2:** 逐页检查文字、表格、标题、页眉页脚和分页。
- [ ] **Step 3:** 若发现缺陷，修改生成脚本并重新生成、重新渲染。

### Task 3: 最终验收

**Files:**
- Inspect: `D:\UniFiles\Python_ML\OtherFiles\银行营销分类实验答辩稿.docx`

**Interfaces:**
- Consumes: 通过视觉检查的 DOCX。
- Produces: 可交付的正式答辩稿。

- [ ] **Step 1:** 核对章节、问答数量、关键数字和输出路径。
- [ ] **Step 2:** 确认无 TODO、TBD、内部引用标记或未替换占位符。
