# Task Shield 论文复现

> 复现论文《The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents》

## 环境搭建

### 1. 创建 Conda 虚拟环境

```bash
conda env create -f environment.yml
conda activate task_shield
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxx
```

### 3. 验证安装

```bash
python -c "import agentdojo; print('AgentDojo OK')"
python -c "from task_shield import TaskShieldDefense; print('Task Shield OK')"
```

## 运行实验

### 快速测试（单个场景单个任务）

```bash
python scripts/run_single_test.py --suite workspace --user-task user_task_0
```

### 运行完整基准测试

```bash
# 运行所有场景所有攻击
python scripts/run_benchmark.py --all

# 运行指定场景
python scripts/run_benchmark.py --suite workspace --attack important_instructions

# 只测试 Utility（无攻击）
python scripts/run_benchmark.py --suite workspace --no-attack
```

### 分析结果

```bash
python scripts/analyze_results.py --results-dir results/
```

## 项目结构

```
├── environment.yml              # Conda 环境配置
├── .env.example                 # API Key 模板
├── task_shield/                 # Task Shield 核心实现
│   ├── prompts.py               # 论文中的所有 Prompt
│   ├── llm_client.py            # DeepSeek API 客户端
│   ├── instruction_extractor.py # 指令提取模块
│   ├── alignment_checker.py     # 对齐检查模块
│   ├── feedback_generator.py    # 反馈生成模块
│   ├── task_shield_defense.py   # 主防御类
│   └── task_shield_pipeline.py  # 自定义 AgentDojo Pipeline
├── scripts/
│   ├── run_benchmark.py         # 运行实验
│   ├── run_single_test.py       # 单个测试
│   └── analyze_results.py       # 结果分析
└── results/                     # 实验结果
```

## 论文对应关系

| 论文内容 | 代码位置 |
|---------|---------|
| Algorithm 1 (附录 B.2) | `task_shield/task_shield_defense.py` |
| Task Extraction Prompt (附录 E) | `task_shield/prompts.py` → `TASK_EXTRACTION_*` |
| Content Checker Prompt (附录 E) | `task_shield/prompts.py` → `CONTENT_CHECKER_*` |
| Tool Call Checker Prompt (附录 E) | `task_shield/prompts.py` → `TOOL_CALL_CHECKER_*` |
| Feedback Generation (附录 E) | `task_shield/prompts.py` → `FEEDBACK_*` |
| AgentDojo 集成 | `task_shield/task_shield_pipeline.py` |

## 预期结果

参考论文 Table 1，使用 Task Shield 后：
- 攻击成功率 (ASR) 应大幅下降（论文报告 ~2.07%）
- 任务效用 (Utility) 应保持较高水平（论文报告 ~69.79%）

> 注：由于使用 DeepSeek-v4 替代 GPT-4o，结果可能与论文原始数据存在差异。
