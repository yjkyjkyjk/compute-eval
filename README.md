# SparseBench

**SparseBench** 是基于 [ComputeEval](README.md) 构建的针对 LLM 生成**稀疏矩阵算子**代码的专项评测体系，聚焦于 NVIDIA cuSPARSE 库的正确性、性能与 API 规范性三个维度。

> 本项目在 `mathlibs` 组的 101 道题目基础上，筛选并扩展了全部 29 道 cuSPARSE 问题，同时设计了一套面向稀疏计算场景的评测方法论。

---

## 目录

- [背景与动机](#背景与动机)
- [cuSPARSE 问题集](#cusparse-问题集)
  - [完整列表](#完整列表)
  - [按算子族分类](#按算子族分类)
- [评测体系设计](#评测体系设计)
  - [评测维度](#评测维度)
  - [评测流水线](#评测流水线)
  - [评分规则](#评分规则)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装](#安装)
  - [运行 cuSPARSE 子集评测](#运行-cusparse-子集评测)
- [实验设计：LLM 提示策略对比](#实验设计llm-提示策略对比)
- [技术路线与扩展计划](#技术路线与扩展计划)

---

## 背景与动机

稀疏矩阵计算是科学计算、图神经网络、大模型推理等领域的核心瓶颈。cuSPARSE 是 NVIDIA 提供的工业级稀疏线性代数库，其 Generic API 设计复杂（两阶段 buffer 模式、多种存储格式、descriptor 生命周期管理），对 LLM 的代码生成能力提出了独特挑战：

- LLM 是否能正确选择稀疏存储格式（CSR / COO / BSR / ELLPACK）？
- LLM 能否处理 cuSPARSE 的两阶段调用模式（`bufferSize` → `execute`）？
- LLM 生成的稀疏算子能达到接近 roofline 的性能吗？
- LLM 能否正确管理 handle / descriptor 的生命周期？

SparseBench 旨在系统性地回答这些问题。

---

## cuSPARSE 问题集

### 完整列表

compute-eval `2026-1` 版本中，`mathlibs` 组共有 101 道题目，其中使用 cuSPARSE 的题目共 **29 道**（14 道标注为 `perf-sensitive`，含性能基准测试）。

| Task ID | 难度 | 核心 cuSPARSE API | 算子类型 | 性能测试 |
|---------|------|-------------------|----------|----------|
| `cusparse/0` | easy | `cusparseAxpby` | 稀疏向量线性组合 (axpy) | — |
| `cusparse/1` | medium | `cusparseXcoosortByRow` | COO 矩阵行排序 | ✓ |
| `cusparse/2` | medium | `cusparseDenseToSparse_convert` | Dense → Blocked-ELL 格式转换 | — |
| `cusparse/3` | medium | `cusparseDenseToSparse_convert` | Dense → CSR 格式转换 | — |
| `cusparse/4` | medium | `cusparseGather` | 稀疏向量 gather | ✓ |
| `cusparse/5` | medium | `cusparseSgpsvInterleavedBatch` | 批量三对角系统求解 | — |
| `cusparse/6` | medium | `cusparseSpVV` + CUDA Graph | SpVV + CUDA Graph Capture | ✓ |
| `cusparse/7` | medium | `cusparseRot` | Givens 旋转（稀疏向量） | ✓ |
| `cusparse/8` | medium | `cusparseScatter` | 稀疏向量 scatter | — |
| `cusparse/9` | medium | `cusparseSDDMM` | SDDMM — BSR 格式 | ✓ |
| `cusparse/10` | medium | `cusparseSDDMM` | 批量 SDDMM — CSR 格式 | — |
| `cusparse/11` | medium | `cusparseSDDMM` | SDDMM — CSR 格式 | — |
| `cusparse/12` | medium | `cusparseSparseToDense` | Sparse → Dense 格式转换 | ✓ |
| `cusparse/13` | medium | `cusparseSpGEMM` | SpGEMM（含 workspace 管理） | ✓ |
| `cusparse/14` | medium | `cusparseSpGEMM` | SpGEMM — CSR 格式 | — |
| `cusparse/15` | medium | `cusparseSpMM` | SpMM — Blocked-ELL 格式 | ✓ |
| `cusparse/16` | medium | `cusparseSpMM` | 批量 SpMM — COO 格式 | ✓ |
| `cusparse/17` | medium | `cusparseSpMM` | SpMM — COO 格式 | — |
| `cusparse/18` | medium | `cusparseSpMM` | 批量 SpMM — CSR 格式 | ✓ |
| `cusparse/19` | medium | `cusparseSpMM` | SpMM — CSR 格式 | — |
| `cusparse/20` | medium | `cusparseSpMV` | SpMV — COO 格式 | — |
| `cusparse/21` | medium | `cusparseSpMV` | SpMV — CSR 格式 | — |
| `cusparse/22` | medium | `cusparseSpMV` | SpMV — Sliced ELLPACK 格式 | — |
| `cusparse/23` | medium | `cusparseSpSM` | 稀疏三角求解（矩阵 RHS）| ✓ |
| `cusparse/24` | medium | `cusparseSpSM` | 稀疏三角求解 A×X=B | ✓ |
| `cusparse/25` | medium | `cusparseSpSV` | 稀疏三角求解（向量 RHS）| ✓ |
| `cusparse/26` | medium | `cusparseSpSV` | SpSV — CSR 格式 | — |
| `cusparse/27` | medium | `cusparseSpSV` | SpSV — Sliced ELLPACK 格式 | ✓ |
| `cusparse/28` | medium | `cusparseSpVV` | 稀疏-稠密向量点积 | — |

### 按算子族分类

| 算子族 | 问题数 | Task IDs | 说明 |
|--------|--------|----------|------|
| **SpMM**（稀疏矩阵 × 稠密矩阵） | 5 | 15, 16, 17, 18, 19 | 支持 CSR / COO / Blocked-ELL，含批量版本 |
| **SpMV**（稀疏矩阵 × 向量） | 3 | 20, 21, 22 | 支持 CSR / COO / Sliced-ELLPACK |
| **SpSV**（稀疏三角求解，向量 RHS） | 3 | 25, 26, 27 | 支持 CSR / Sliced-ELLPACK |
| **SpSM**（稀疏三角求解，矩阵 RHS） | 2 | 23, 24 | 下三角矩阵，多 RHS |
| **SpGEMM**（稀疏矩阵 × 稀疏矩阵） | 2 | 13, 14 | 含 workspace 两阶段管理 |
| **SDDMM**（Sampled Dense-Dense MM） | 3 | 9, 10, 11 | 支持 CSR / BSR，含批量版本 |
| **格式转换**（Dense ↔ Sparse） | 3 | 2, 3, 12 | Dense→CSR / Dense→Blocked-ELL / Sparse→Dense |
| **稀疏向量运算** | 6 | 0, 4, 6, 7, 8, 28 | axpy / gather / scatter / Givens旋转 / SpVV |
| **矩阵排序** | 1 | 1 | COO 按行排序 |
| **批量三对角求解** | 1 | 5 | `GpsvInterleavedBatch`，cuBLAS 混用 |

---

## 评测体系设计

### 评测维度

SparseBench 从三个正交维度评测 LLM 生成代码的质量：

```
综合得分 = 正确性 (50%) + 性能 (30%) + 代码质量 (20%)
```

对于非 `perf-sensitive` 题目，性能权重转入正确性，变为 `正确性 (80%) + 代码质量 (20%)`。

#### 1. 正确性（Functional Correctness）

测试输入覆盖以下稀疏矩阵场景：

| 场景 | 稀疏率 | 矩阵类型 |
|------|--------|----------|
| 轻度稀疏 | 50% | 随机 |
| 中度稀疏 | 90% | 随机 / 对角带状 |
| 高度稀疏 | 99% | 随机 / 分块结构 |
| 真实矩阵 | 变化 | SuiteSparse Matrix Collection |

数值验证：与 CPU 端 `scipy.sparse` 或 cuBLAS 稠密结果对比，容差 `1e-5`（FP32）/ `1e-10`（FP64）。

#### 2. 性能（Performance，仅 perf-sensitive 题目）

| 指标 | 定义 | 目标 |
|------|------|------|
| GFLOPS 效率 | 实测 GFLOPS / 理论峰值 GFLOPS | > 50% |
| 内存带宽效率 | 实测带宽 / GPU 峰值带宽 | > 60% |
| 相对基准加速比 | 生成方案时间 / 基准方案时间 | ≥ 0.8× |

性能分级：`>80% roofline` = 优秀，`50–80%` = 良好，`<50%` = 不及格。

#### 3. 代码质量（API 规范性）

静态检查（编译阶段）：
- Handle / Descriptor 是否配对创建与销毁（`cusparseCreate` ↔ `cusparseDestroy`）
- `CUSPARSE_CHECK` 或等价错误处理是否完整覆盖每次 API 调用
- `bufferSize` → `allocate` → `execute` 三段式调用是否完整

动态检查（运行阶段）：
- `cuda-sanitizer` 检测内存越界与未初始化访问
- Nsight Systems 检测不必要的 host-device 同步点

### 评测流水线

```
问题 Prompt
    │
    ▼
LLM 代码生成（支持 zero-shot / few-shot / CoT / RAG）
    │
    ▼
编译（nvcc + cuSPARSE 链接）
    │  ✗ 编译失败 → 记录 0 分
    ▼
正确性测试（多组稀疏矩阵输入）
    │  ✗ 测试失败 → 记录正确性分
    ▼
性能 Benchmark（仅 perf-sensitive）
    │
    ▼
API 规范性检查（静态 + 动态）
    │
    ▼
综合评分输出
```

与 compute-eval 原有框架的集成点：
- 一次性运行 `scripts/create_sparse_datapack.py` 生成 `sparse` 专属 datapack
- 使用 `--include=sparse` 直接定位稀疏算子题目，无需在大 mathlibs 组中二次过滤
- 使用 `--profile_mode=ncu` 开启 Nsight Compute 性能剖析
- 扩展 `profilers/` 模块添加稀疏矩阵专用 GFLOPS 计算

### 评分规则

```json
{
  "task_id": "cusparse/19",
  "correctness_score": 1.0,
  "performance_score": 0.82,
  "quality_score": 0.9,
  "final_score": 0.918,
  "details": {
    "test_cases_passed": "5/5",
    "gflops_efficiency": "82%",
    "handle_leak": false,
    "error_handling_complete": true
  }
}
```

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA Toolkit 12.0+（评测阶段必须）
- NVIDIA GPU，计算能力 ≥ 7.0（V100 / A100 / H100 推荐）
- Docker（推荐，用于隔离执行生成代码）

### 安装

```bash
git clone <repo-url>
cd compute-eval
uv sync
```

### 运行 cuSPARSE 子集评测

**第零步（一次性）：生成 sparse 专属 datapack**

```bash
# 从 mathlibs datapack 中提取 29 道 cuSPARSE 题目，写入独立的 sparse datapack
uv run python scripts/create_sparse_datapack.py
# 输出：data/releases/2026-1-sparse-problems.tar.gz
```

完成后，后续所有命令只需 `--include=sparse`，无需在结果里二次过滤。

**第一步：生成代码**

```bash
uv run compute_eval generate_samples \
  --release=2026-1 \
  --include=sparse \
  --problems_datapack_dir=data/releases/ \
  --model=claude-opus-4-6 \
  --solutions_per_problem=5 \
  --n_workers=8
```

**第二步：正确性评测**

```bash
uv run compute_eval evaluate_functional_correctness \
  --release=2026-1 \
  --solutions_datapack=2026-1-claude-opus-4-6-solutions.tar.gz \
  --problems_datapack_dir=data/releases/ \
  --mode=docker \
  --k='(1, 3, 5)' \
  --n_workers=4
```

**第三步：性能评测（可选，需要本地 GPU）**

```bash
uv run compute_eval evaluate_functional_correctness \
  --release=2026-1 \
  --solutions_datapack=2026-1-claude-opus-4-6-solutions.tar.gz \
  --problems_datapack_dir=data/releases/ \
  --mode=local \
  --profile_mode=ncu \
  --n_workers=2
```

评测结果已按 `sparse` 组自动聚合，JSON 输出中 `metrics_by_group.sparse` 即为 cuSPARSE 专项指标，无需额外过滤。

---

## 实验设计：LLM 提示策略对比

SparseBench 支持以下四种提示策略的对比实验，以研究哪种方式最有助于 LLM 正确生成 cuSPARSE 代码：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **Zero-shot** | 仅提供问题描述，无示例 | 评测模型基础能力 |
| **Few-shot** | 提供 1–3 个 cuSPARSE 正确示例 | 评测上下文学习效果 |
| **Chain-of-Thought** | 要求模型先规划 API 调用流程再写代码 | 评测推理链对复杂 API 的帮助 |
| **RAG 增强** | 自动检索 cuSPARSE 官方文档片段注入 Prompt | 评测外部知识对正确率的提升 |

使用 `--system_prompt` 参数或配置文件指定不同策略：

```yaml
# few_shot_config.yaml
release: 2026-1
include: [sparse]
problems_datapack_dir: data/releases/
model: claude-opus-4-6
solutions_per_problem: 5
system_prompt: |
  You are an expert CUDA programmer. Here is an example of correct cuSPARSE usage:

  // SpMV example
  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, rows, cols, nnz, ...);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecX, &beta, vecY,
                          CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);
  cusparseSpMV(handle, ...);
  cusparseDestroySpMat(matA);

  Now implement the following function:
```

**关键研究问题：**

1. **格式选择**：LLM 是否能根据矩阵结构自动选择最优存储格式？
2. **两阶段 API**：LLM 能否正确处理 `bufferSize` → `execute` 的调用模式？
3. **高稀疏度鲁棒性**：在稀疏率 > 99% 时，生成代码是否仍然正确？
4. **性能意识**：LLM 能否在不提示的情况下选择高效算法（如 `CUSPARSE_SPMM_ALG_DEFAULT` vs 手动选择）？
5. **错误处理**：LLM 是否会自发添加完整的 `CUSPARSE_CHECK` 宏？

---

## 技术路线与扩展计划

### 当前阶段（Phase 1）：基础评测

- [x] 筛选 29 道 cuSPARSE 题目，建立子集评测流程
- [x] 复用 compute-eval 正确性评测框架
- [x] 集成 Nsight Compute 性能剖析（`--profile_mode=ncu`）

### Phase 2：稀疏专用评测扩展

- [ ] 添加多种稀疏矩阵生成器（随机 / 带状 / 块对角 / 真实矩阵）
- [ ] 实现基于 GFLOPS 和带宽效率的稀疏性能评分
- [ ] 添加 API 规范性静态检查模块（handle 泄漏 / error handling 覆盖）

### Phase 3：题目类型扩展

在现有 **Generation** 类型基础上，新增：

| 题目类型 | 描述 | 示例 |
|----------|------|------|
| **Optimization** | 优化已有正确但低效的 cuSPARSE 代码 | 选择更优 SpMM algorithm，调整 workspace 策略 |
| **Repair** | 修复包含常见错误的代码 | 修复遗漏 `cusparseSpMM_preprocess` 调用 |
| **Translation** | 将 Legacy API 迁移到 Generic API | `cusparseDcsrmv` → `cusparseSpMV` |

### Phase 4：新库覆盖

参考 [DOMAIN_MAP.md](DOMAIN_MAP.md) 中的 gap，扩展至：

| 库 | 说明 |
|----|------|
| **cuSPARSELt** | 2:4 结构化稀疏（Ampere+ 专用，适合稀疏 Transformer） |
| **cuDSS** | 直接稀疏求解器（LU / Cholesky 分解） |
| **混合精度 SpMM** | FP16 / BF16 稀疏矩阵乘法 |

---

## 数据集与许可

问题集来自 compute-eval `2026-1` 发布版，遵循 [NVIDIA Evaluation Dataset License Agreement](data/LICENSE)。

- 数据集**仅可用于 AI 模型的评测与基准测试**，不可用于训练。
- 代码部分遵循 [Apache 2.0](LICENSE)。

---

## 引用

如果本项目对您的研究有帮助，请引用 compute-eval 原始项目，并注明使用了 SparseBench 稀疏矩阵评测子集。

---

*SparseBench 基于 [ComputeEval](README.md) 构建 | 当前版本：2026-1 | 问题数：29（cuSPARSE）*
