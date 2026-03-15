# 自适应教学 Agent + QBank 题库系统

## 目录结构

```
Teaching_Agent/Teaching_Agent/
├── src/
│   ├── qbank_agent/              ← QBank-agent 子包（软链接）
│   ├── qbank_pipeline.py         ← 全自动题库 orchestrator（核心入口）
│   ├── watch_slide_adaptive_new.py  ← 实时自适应教学 Agent
│   ├── build_qbank_index_openai.py  ← ChromaDB 索引构建（已集成到 pipeline）
│   ├── chroma_qbank_openai/      ← ChromaDB 向量数据库
│   ├── data/
│   │   ├── student_events.jsonl  ← 学生答题记录
│   │   └── student_profile.json  ← 学生画像
│   └── *.json                    ← 当前使用的题库 JSON
└── requirements.txt
```

## 环境准备

```bash
# 1. 进入项目目录
cd Teaching_Agent/Teaching_Agent

# 2. 安装依赖（使用你当前 conda 环境的 pip）
pip install -r requirements.txt

# 3. 确保 .env 文件中有 OpenAI API Key
#    在 src/ 目录下创建 .env 文件（如果没有的话）
echo "OPENAI_API_KEY=你的key" > src/.env

# 4. 同步在 Teaching_Agent/QBank-agent/qbank_agent/config.py 中设置 OPENAI_API_KEY（与 .env 保持一致）
```

> **注意**：所有命令都必须在 `src/` 目录下执行：
> ```bash
> cd Teaching_Agent/Teaching_Agent/src
> ```

---

## 命令一览

| 命令 | 功能 | 耗时 | 是否调用 LLM |
|---|---|---|---|
| `python qbank_pipeline.py generate` | 全流程：PDF→出题→模拟评测→部署 | **很长**（每个主题约1-2分钟） | ✅ 大量调用 |
| `python qbank_pipeline.py maintain` | 分析学生数据→自动演化劣质题 | 中等 | ✅ 仅对flagged题 |
| `python qbank_pipeline.py reindex` | 仅重建 ChromaDB 向量索引 | **几秒** | ✅ Embedding |
| `python watch_slide_adaptive_new.py` | 启动学生自适应学习会话 | 持续运行 | ✅ 实时 |

---

## 详细使用说明

### 1. 全流程生成题库（`generate`）

从 PDF 课件一键生成题库并部署到 ChromaDB。

```bash
cd Teaching_Agent/Teaching_Agent/src
python qbank_pipeline.py generate
```

**执行流程（6个阶段）：**
1. 解析 PDF → JSON（跳过已解析的）
2. 生成教学大纲（跳过已有的）
3. 为每个主题生成 5 道 MCQ（**最耗时**，每题需 LLM 调用）
4. 打乱选项顺序
5. 认知模拟评测（模拟3类学生答题）
6. 部署到 `src/` 并重建 ChromaDB 索引

**中断与恢复：**
- 随时按 `Ctrl+C` 中断
- 再次运行同一命令会**自动跳过已完成的步骤**
- PDF 放在 `QBank-agent/data/input_slides/` 中

---

### 2. 质量维护（`maintain`）

分析真实学生答题数据，自动找出质量差的题目并演化替换。

```bash
cd Teaching_Agent/Teaching_Agent/src
python qbank_pipeline.py maintain
```

**前提条件：** `data/student_events.jsonl` 中需有学生答题记录（至少3条/题才触发分析）

**执行流程：**
1. 读取 `student_events.jsonl`
2. 计算每题的 P-value（难度）和区分度
3. 标记问题题目：`too_easy` / `too_hard` / `low_discrimination`
4. 自动用 LLM 重新出题替换
5. 旧题标记 `deprecated`，新题追加
6. 自动重建 ChromaDB 索引

**输出示例：**
```
  🟢 bayes_theorem:q1  p=0.67  disc=0.94  attempts=6  flag=ok
  🔴 mle_basics:q3     p=0.95  disc=0.10  attempts=8  flag=too_easy
  ⚠️  1 question(s) flagged. Auto-evolving...
```

---

### 3. 重建索引（`reindex`）

当你手动修改了题库 JSON 文件后，用这个命令重建 ChromaDB。

```bash
cd Teaching_Agent/Teaching_Agent/src
python qbank_pipeline.py reindex
```

**速度最快**，通常几秒完成。

---

### 4. 启动学生学习会话

```bash
cd Teaching_Agent/Teaching_Agent/src
python watch_slide_adaptive_new.py --student alice
```

**工作方式：**
- 监听 `current_slide.txt` 文件内容变化
- 当幻灯片切换时，自动从 ChromaDB 检索相关题目
- 通过 LLM 根据学生画像定制题目（Agentic RAG）
- 记录答题数据到 `data/student_events.jsonl`
- 每 5 题自动调整难度

**退出：** 按 `Ctrl+C`，退出时会**自动触发质量维护**（相当于自动运行 `maintain`）

---

## 常见问题

### Q: `generate` 太慢了怎么办？
生成速度取决于 LLM API 响应速度。9 套 PDF × 每套 5-8 个主题 × 每主题 5 题 = 约 200-350 次 LLM 调用。可以 `Ctrl+C` 中断，下次会从断点继续。

### Q: 如何只给新加的 PDF 生成题库？
把新 PDF 放入 `QBank-agent/data/input_slides/`，然后运行 `generate`。已有的会自动跳过。

### Q: `ModuleNotFoundError` 怎么办？
确保用 conda 环境的 pip 安装依赖：
```bash
~/miniconda3/bin/pip install -r requirements.txt
```

### Q: 如何查看当前题库有多少题？
```bash
cd Teaching_Agent/Teaching_Agent/src
python -c "
import json
from pathlib import Path
for f in sorted(Path('.').glob('*.mcq_by_topic_*.json')):
    d = json.load(open(f))
    total = sum(len(t.get('questions',[])) for t in d.get('topics',[]))
    print(f'{f.name}: {total} questions')
"
```
