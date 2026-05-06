# Agent 运行日志质量检测器 - 设计文档

## 概述

实现一个最小可用的 Agent 运行日志检测器，用于模型/RAG升级后通过运行日志发现质量退化信号，为上线/回滚提供依据。

**输入**：两份Claude Code导出的JSONL日志目录（baseline vs upgraded）
**输出**：一份详细分析风格的HTML报告，含指标数据、对比、退化风险判断和处理建议

## 项目结构

```
agent_log_check/
├── main.py                  # CLI入口
├── prompts/
│   └── judge_prompt.md      # LLM判断提示词模板（$variable语法，string.Template）
├── src/
│   ├── parser.py            # JSONL日志解析器
│   ├── pipeline.py          # MetricPipeline 流水线
│   ├── reporter.py          # HTML报告生成器
│   ├── llm_judge.py         # claude -p 调用模块
│   └── metrics/
│       ├── __init__.py
│       ├── base.py          # BaseMetric 基类
│       ├── resource.py      # 资源消耗类指标
│       ├── tool_usage.py    # 工具调用类指标
│       ├── thinking.py      # 深度思考类指标
│       └── content_quality.py # 内容质量类指标
└── sample/                  # 示例日志数据
```

## 数据模型

### LogSession
一份日志（主agent或子agent）的解析结果：
- `name`: 日志标识名
- `log_path`: 原始日志路径
- `messages`: 所有Message列表
- `first_timestamp` / `last_timestamp`: 首末时间戳

### Message
单条消息：
- `type`: user / assistant / system 等
- `timestamp`: ISO时间戳
- `role`: user / assistant
- `content`: ContentBlock列表
- `usage`: TokenUsage

### ContentBlock
内容块：
- `type`: text / thinking / tool_use / tool_result
- `text`: 文本内容（text/thinking）
- `tool_name`: 工具名（tool_use）
- `tool_input`: 工具输入参数（tool_use）
- `tool_use_id`: 关联ID（tool_result）
- `is_error`: 是否出错（tool_result）
- `error_content`: 错误内容

### TokenUsage
- `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`

### LogGroup
- `main`: LogSession（主agent）
- `subagents`: list[(meta_info, LogSession)]（子agent列表）

## 日志解析 (parser.py)

`parse_log_dir(log_dir) -> LogGroup`:
1. 扫描目录根下的 `.jsonl` 文件 → 主agent日志
2. 扫描 `*/subagents/` 下的 `.jsonl` + `.meta.json` → 各子agent
3. 每行JSONL解析为Message，只保留type=user/assistant的消息
4. content为字符串时包装为ContentBlock列表
5. 缺失字段给默认值

## 指标检测器

### 基类 BaseMetric

```python
class BaseMetric(ABC):
    name: str
    category: str  # resource / tool / thinking / content

    @abstractmethod
    def compute(self, session: LogSession) -> dict:
        """返回 {metric_key: value} 字典"""
```

### ResourceMetric (category: resource)

| 指标key | 含义 | 计算方式 |
|---------|------|---------|
| total_input_tokens | 输入token总量 | 累加 input_tokens + cache_read_input_tokens |
| total_output_tokens | 输出token总量 | 累加 output_tokens |
| session_duration_sec | 会话时长(秒) | last_timestamp - first_timestamp |
| tokens_per_second | 每秒输出token数 | total_output_tokens / duration |
| total_steps | 执行总步骤数 | assistant消息数量 |

### ToolUsageMetric (category: tool)

| 指标key | 含义 | 计算方式 |
|---------|------|---------|
| tool_call_count | 工具调用次数 | type=tool_use的content块数 |
| tool_error_count | 工具调用失败次数 | type=tool_result且is_error=True的块数 |
| tool_error_rate | 失败占比 | error_count / call_count |
| subagent_count | 子agent调用数 | tool_use中name=Agent的块数（仅主agent有意义） |
| tool_name_distribution | 各工具调用次数 | {tool_name: count} |

### ThinkingMetric (category: thinking)

| 指标key | 含义 | 计算方式 |
|---------|------|---------|
| thinking_count | think次数 | type=thinking的content块数 |
| thinking_total_chars | think总字符数 | 累加thinking块字符长度 |
| thinking_avg_chars | think平均字符数 | total_chars / count |

### ContentQualityMetric (category: content)

| 指标key | 含义 | 计算方式 |
|---------|------|---------|
| vague_word_hits | 不确定性词汇命中次数 | 在text块中匹配关键词列表 |
| sentence_repeat_3plus | 重复3次以上句子数 | 按句号/问号/感叹号分句，统计频次>=3的句子 |

不确定性词汇列表：可能、大概、或许、建议、疑似、也许、似乎、好像、应该、不确定、不确定是否、无法确定、未必、不一定

## 流水线 (pipeline.py)

```python
class MetricPipeline:
    def __init__(self):
        self.metrics: list[BaseMetric] = []

    def register(self, metric: BaseMetric): ...

    def run(self, session: LogSession) -> dict:
        """运行所有检测器，返回 {category: {metric_key: value}} """
```

## LLM判断 (src/llm_judge.py)

1. 从 `prompts/judge_prompt.md` 加载提示词模板（使用 `string.Template` 的 `$variable` 语法）
2. 填充模板变量：baseline_name、baseline_metrics、upgraded_name、upgraded_metrics、baseline_log_path、upgraded_log_path
3. 通过 stdin 将提示词传给 `claude -p --output-format json`（避免 Windows 命令行长度限制）
4. 解析 JSON 结果：风险等级(高/中/低)、各指标风险判断、处理建议

HTML报告中只展示解析后的结构化内容（风险等级、各指标判断、建议），不展示LLM原始输出。

## HTML报告 (reporter.py)

自包含HTML文件，详细分析报告风格：

1. **概览区**：两份日志基本信息、整体风险等级
2. **主Agent指标对比区**：四类指标表格，行=指标，列=指标名|baseline值|upgraded值|变化幅度，颜色编码（红=退化，绿=改善，灰=持平）
3. **子Agent指标区**：每个子agent一个section，含meta信息+四类表格
4. **LLM分析区**：结构化的退化风险判断和处理建议（不含原始输出）

## CLI用法

```bash
python main.py \
  --baseline sample/log/log_glm5.1_20260505 \
  --upgraded sample/log/log_minimax2.7_20260505 \
  --output report.html
```

## 边界处理

- 跳过非user/assistant类型的消息
- content为字符串时自动包装
- 缺失字段给默认值（0或None）
- session_duration_sec为0时tokens_per_second返回0
- tool_call_count为0时tool_error_rate返回0
- thinking_count为0时thinking_avg_chars返回0
- claude -p调用失败时，LLM分析区显示"LLM判断调用失败"并保留指标数据
