# 知途 · 知识追踪与自适应学习路径系统

> 用知识追踪模型估计每个知识点的掌握程度，据此规划下一步该学什么，并让每一次推荐都附带可解释的理由。

挑战杯参赛作品。第一版面向《数据结构》单门课程，28 个知识点、224 道题。

---

## 一、这个项目在做什么

现在的刷题平台只记录**行为量**（做了几道题、正确率多少），不估计**知识状态**（哪个知识点真的会了、
哪个只是碰巧蒙对）。知途把答题序列喂给知识追踪模型，算出每个知识点的掌握概率，
再结合知识点之间的前置依赖关系给出下一步该学什么，并把"为什么推荐这个"讲清楚。

**两点差异化**（也是答辩时主要被追问的地方）：

1. **多模型对比实验**。BKT / DKT / SAKT 三个模型加两个非学习基线（全局均值、逐知识点频率），
   在同一数据集、同一套评测口径下对比，报告 AUC 与置信区间。不是"我们搭了个系统"，
   而是"我们测了什么、发现了什么"。
2. **可解释推荐**。每次推荐都输出可核查的依据：
   > 推荐这道题，因为它是「二叉树的遍历」这个知识点的练习，而你在这个知识点上的掌握概率为 0.34，低于 0.70 的阈值。
   > （补强）。作答依据：已作答 4 次、答对 1 次；模型预测你答对这道题的概率约为 0.41。

---

## 二、快速开始

### 1. 后端环境

```bash
cd Qin-path

# 建虚拟环境（本机开发时用的是一份独立 venv，路径无所谓，重建一份即可）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
# 若 pip 装到的 torch 体积过大，用 CPU 轮子：
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 有显卡的机器请改用 `requirements-gpu.txt` 的步骤（先装 CUDA 版 torch 再装其余依赖）。

### 2. 建数据集

```bash
python -m ml.data.build_dataset          # 按当前设备自动选档位
python -m ml.data.build_dataset --profile cpu
```

产物在 `data/processed/`：知识点依赖图、题库 CSV、三个切分（train/valid/test）。

### 3. 跑对比实验（本项目的核心产出）

```bash
python -m ml.run_experiments                        # 全部模型
python -m ml.run_experiments --models BKT DKT SAKT  # 只跑三个模型
python -m ml.run_experiments --skip-bootstrap       # 快速模式
```

产出：
- `ml/artifacts/results_<数据集>.json` —— 指标原始数据
- `ml/artifacts/report_<数据集>.md` —— 可读报告（含自动生成的结论段）
- `ml/artifacts/models/*.pt|.json` —— 训练好的模型，后端要加载

**已跑出的正式结果**（ASSISTments 2009，测试集 370 名学生 / 16920 条交互，CPU 档，seed=42）：

| 模型 | AUC | AUC 95% 置信区间 | 准确率 | RMSE | NLL | 参数量 |
|------|------|------------------|--------|------|-----|--------|
| Mean（全局均值，最弱基线） | 0.5000 | [0.5000, 0.5000] | 0.6291 | 0.4830 | 0.6594 | 不适用 |
| KC-Mean（逐知识点频率，严格下限） | 0.6206 | [0.6019, 0.6393] | 0.6515 | 0.4716 | 0.6360 | 不适用 |
| BKT | 0.7177 | [0.6947, 0.7341] | 0.7005 | 0.4471 | 0.5858 | 不适用 |
| **DKT** | **0.8092** | [0.7917, 0.8255] | 0.7555 | 0.4051 | 0.4885 | 29233 |
| SAKT | 0.7977 | [0.7779, 0.8160] | 0.7427 | 0.4128 | 0.5068 | 48577 |

要点：DKT 相对最强非学习基线（KC-Mean）提升 **+0.1886 AUC**，说明知识状态建模确实有增益；
但 DKT 与 SAKT 的置信区间大幅重叠（`[0.7917, 0.8255]` vs `[0.7779, 0.8160]`），
**不能声称二者有实质差异**——这与设计文档引用的实证研究结论一致，也是报告里如实写出的结论。

> **两个数据集各管一段，不要混着讲**：
> - **产品演示**跑的是自拟的《数据结构》课程（28 个知识点、224 道题）+ 模拟学生作答，
>   用来演示学生端闭环与教师端薄弱点定位；
> - **答辩指标**跑的是公开数据集 ASSISTments 2009（116 个知识点、3694 名学生），
>   用它的知识点划分，与演示课程相互独立。
>
> 两者知识点数不同是刻意的，不是数据错乱。教师端实验表下方也写了这条口径说明。

### 4. 起后端

```bash
python -m backend.app.seed          # 初始化演示数据（班级 + 12 名学生 + 作答历史）
python -m uvicorn backend.app.main:app --reload --port 8000
```

接口文档：<http://127.0.0.1:8000/docs>

演示账号（密码均为 `123456`）：

| 账号 | 角色 |
|------|------|
| `teacher` | 教师 |
| `stu01` … `stu12` | 学生 |

### 5. 起前端

```bash
cd frontend
npm install
npm run dev
```

打开 <http://localhost:5173>（已配好 `/api` 代理到 8000 端口）。

### 6. 跑测试

```bash
python -m pytest backend/tests -q      # 后端接口闭环
```

---

## 三、目录结构

```
Qin-path/
├── ml/                              模型与实验（项目的技术内核）
│   ├── config.py                    设备探测 + 规模档位（cpu / gpu）
│   ├── data/
│   │   ├── knowledge_graph.py       《数据结构》28 个知识点 + 前置依赖
│   │   ├── question_bank.py         题库 224 道题
│   │   ├── synth.py                 合成答题序列生成器（BKT 生成过程 + 前置影响）
│   │   ├── adapters.py              公开数据集适配（ASSISTments CSV / pyKT pickle）
│   │   ├── dataset.py               按学生切分 + 持久化
│   │   └── build_dataset.py         数据集构建入口
│   ├── models/
│   │   ├── base.py                  统一接口 KnowledgeTracer
│   │   ├── baseline.py              全局均值 / 逐知识点频率（非学习基线）
│   │   ├── bkt.py                   贝叶斯知识追踪，EM 估计四参数
│   │   ├── dkt.py                   深度知识追踪（LSTM）
│   │   └── sakt.py                  自注意力知识追踪
│   ├── planner.py                   掌握度 → 路径（显式规则，无大模型参与）
│   ├── metrics.py                   AUC / ACC / RMSE / NLL + bootstrap 置信区间
│   ├── service.py                   在线推理服务（掌握度、推荐、班级统计、曲线）
│   ├── train.py                     单模型训练
│   └── run_experiments.py           对比实验入口
├── backend/                         FastAPI 服务
│   └── app/
│       ├── main.py                  应用入口
│       ├── models.py                ORM（只存会变的东西）
│       ├── security.py              bcrypt 口令 + HMAC 签名 token
│       ├── routers/                 auth / student / teacher / meta
│       ├── services/history.py      把作答记录还原成模型输入序列
│       └── seed.py                  演示数据
├── frontend/                        Vue 3 + Vite + ECharts
│   └── src/
│       ├── views/                   LoginView / StudentView / TeacherView
│       └── components/              KnowledgeDAG / LearningCurve / ClassHeatmap / ExperimentTable
├── data/                            raw（原始数据集）+ processed（构建产物）
└── ml/artifacts/                    实验指标与模型产物
```

---

## 四、系统架构

```
数据层    题库（题目 ↔ 知识点标注） + 作答记录（学生 × 知识点 × 对错 × 顺序）
            │
模型层    知识追踪模型（BKT / DKT / SAKT）→ 逐知识点掌握概率
          知识点依赖图（DAG）→ 路径规划的约束条件
            │
应用层    学生端：做题 → 掌握度实时更新 → 接收推荐 → 查看推荐理由
          教师端：全班掌握度分布 → 定位共性薄弱点
```

**一条重要的建模约定**：`mastery` 里 **未观测（还没做过）的知识点记 NaN，不记 0**。
把"没做过"当成"不会"会把一大批尚未学习的知识点误判成漏洞，推荐直接乱掉。
代码里 `BKT.mastery()` 与 `planner.plan_path()` 都严格区分"未知"与"薄弱"两种状态。

---

## 五、接入公开数据集（答辩指标必须走这条）

合成数据集的产物里写死了 `data_source='synthetic'`，界面上也会打标提示，
**它只用于验证实现与演示，不能作为答辩结论**。

### 方式一：ASSISTments 原始 CSV（本机已跑通）

```bash
# 1) 下载（USTC 镜像，无需 EduData）
curl -L -o data/raw/assist2009_skill_builder.zip \
    http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip
# 2) 解压得到 skill_builder_data_corrected.csv
python -c "import zipfile;zipfile.ZipFile('data/raw/assist2009_skill_builder.zip').extractall('data/raw/')"
# 3) 构建切分（--max-seq-len 必须与训练时 max_len 一致，CPU 档为 100）
python -m ml.data.build_dataset --source assistments \
    --csv data/raw/skill_builder_data_corrected.csv \
    --name assistments2009 --prefix assistments2009 --max-seq-len 100
# 4) 跑实验（--prefix 要与上一步一致）
python -m ml.run_experiments --prefix assistments2009
```

适配器会自动识别列名别名（`user_id` / `skill_id` / `problem_id` / `correct` / `order_id`
及其常见变体），并按 `--min-kc-count`（默认 30）过滤长尾知识点，按 `--max-seq-len`
截断每条学生序列。

两个容易踩的点：

- **编码**：ASSISTments 2009 的导出文件不是 UTF-8（实测是 cp1252/latin-1，
  题目文本里含 `0x80` 字节），按 utf-8 读会在第 2634 字节直接抛 `UnicodeDecodeError`。
  适配器按 `utf-8-sig → cp1252 → latin-1` 顺序自动回退。
- **序列长度必须对齐**：BKT 会在全长序列上评测，DKT/SAKT 只评测前 `max_len` 步。
  如果数据集长度大于模型的 `max_len`，五个模型其实不在同一批交互上比。
  所以 `--max-seq-len` 要和 `ml/config.py` 里对应档位的 `max_seq_len` 保持一致。

### 方式二：pyKT 预处理后的 pickle

```bash
pip install EduData
python -c "from EduData import get_data; get_data('assistment-2009-2010-skill', 'data/raw/')"
```

EduData 内置数据集：`assist2009` / `assist2012` / `assist2015` / `algebra2005` /
`bridge2006` / `statics2011` / `junyi` / `xes3g5m` / `ednet_kt1`。

拿到数据后：

```bash
python -m ml.data.build_dataset --source pykt --csv data/raw/<文件>.pkl \
    --name assist2009 --prefix assist2009 --max-seq-len 100
python -m ml.run_experiments --prefix assist2009
```

> EduData 的 USTC 镜像（`base.ustc.edu.cn`）实测会间歇性被网络拦截
> （表现为连接被 RST，或响应不带 `Content-Length` 导致下载器抛 `TypeError`）。
> 被拦时不用折腾下载器，直接按方式一用 `curl` 拉原始 zip 即可；
> 若 `curl -I` 能拿到 `HTTP/1.1 200` 且带 `Content-Length`，就说明镜像已恢复。

### 方式三：自采真实数据（第二版）

请任课老师提供一个班一学期的作业或小测记录，按 `question_bank.load_bank_from_csv()`
与 `adapters.load_assistments_csv()` 的列约定整理即可。

---

## 六、迁移到有显卡的机器

代码不需要改。`ml/config.py` 的 `resolve_device()` 会自动探测 CUDA，
并切到 `gpu` 档位（数据规模 ×5、轮次 ×2、模型加宽）。

```bash
# 目标机器上：装 CUDA 版 torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"      # 期望 True

set ZT_PROFILE=gpu
# 演示用合成数据
python -m ml.data.build_dataset --profile gpu
# 答辩用真实数据集（--max-seq-len 要对齐 gpu 档的 200）
python -m ml.data.build_dataset --source assistments \
    --csv data/raw/skill_builder_data_corrected.csv \
    --name assistments2009 --prefix assistments2009 --max-seq-len 200
python -m ml.run_experiments --profile gpu --prefix assistments2009
```

> **迁过去以后指标必须重跑**。GPU 档用的是另一组超参（序列 200 步、轮次 120、模型加宽），
> 和本机 CPU 档跑出来的数字不是同一组实验，**不能直接把 CPU 档的 AUC 拿去答辩**。
> 跑完记得把第二节的结果表和各处答辩材料一起换掉。

详细清单见 `requirements-gpu.txt`。

支持的环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `ZT_DEVICE` | `auto` | 强制 `cpu` / `cuda` / `mps` |
| `ZT_PROFILE` | `auto` | 规模档位 `cpu` / `gpu`（auto 按设备选） |
| `ZT_DATABASE_URL` | SQLite | 换成 PostgreSQL 时改这里 |
| `ZT_SECRET_KEY` | 演示值 | 生产环境必须覆盖 |
| `ZT_KT_MODEL` | `BKT` | 用哪个模型产出掌握度 |

---

## 七、踩过的坑（复现时先看这里）

**1. 深度模型的模型选择必须用验证集 AUC，不能用验证集 loss。**
答题数据噪声很大（失误/猜测），验证 loss 常常在第 2 个 epoch 就平台化，而模型其实还在
持续改善排序能力。最初按 loss 选 checkpoint，结果回滚到几乎没训起来的模型，线上 AUC ≈ 0.5。
改用 AUC 选型后恢复正常。见 `ml/models/torch_utils.py` 的 `train_loop()`。

**2. 训练步数不足会让 DKT 直接退化成"预测均值"。**
早期用 96 名学生 × 12 epoch × batch 32 = 36 个优化步，深度模型完全没训起来（AUC 0.49）。
CPU 档位把 epochs 提到 60 才够用。

**3. 合成数据的序列结构必须贴近真实刷题行为。**
第一版生成器每步在知识点池里均匀随机抽题，等于让模型学"无记忆"序列，
序列模型被不公平地压制。真实数据里学生是成串刷同一个知识点的，
所以加了 `repeat_prob`（默认 0.55）与"优先练薄弱点"的抽样权重。

**4. 未观测 ≠ 薄弱。** 见第四节。

**5. `SequenceDataset` 实现了序列协议。** 可以像 `list[dict]` 一样直接传给模型，
不用到处写 `.sequences`。

**6. ASSISTments 2009 的 CSV 不是 UTF-8，且有个 0x80 字节的雷。**
`skill_builder_data_corrected.csv` 是 cp1252/latin-1 编码，按 `utf-8` 读会在**第 2634 字节**
直接抛 `UnicodeDecodeError`（报错位置在文件极前面，容易误判成文件损坏）。
适配器按 `utf-8-sig → cp1252 → latin-1` 顺序流式嗅探编码再读。
注意别为了"顺手"把整个文件 `list(reader)` 物化：40 万行 × 30 列的 dict 列表要吃掉近 1GB，
改成惰性迭代后峰值内存只有 62MB。

**7. 按频次过滤知识点时，题目表会带上"孤儿题目"。**
知识点按 `min_kc_count` 过滤后，若 `question_ids` 仍由全部原始行构建，
其中属于已丢弃知识点的题目在拼 `question_to_kc` 时会让 `kc_index[...]` 直接 `KeyError`。
合成数据所有知识点都保留，所以这个坑只有接真实数据才会暴露。
修法：`item_to_kc` 只在保留下来的知识点里构建。回归测试见 `ml/tests/test_adapters.py`。

**8. 数据集截断长度必须与模型 `max_len` 对齐，否则五个模型不可比。**
BKT 会在每条序列的**全长**上评测，而 DKT/SAKT 只看前 `max_len` 步（`ml/config.py` 档位决定，
CPU 档 100、GPU 档 200）。若数据集允许的序列长度大于模型 `max_len`，
汇总表看着整齐，其实几个模型根本不在同一批交互上比。
所以 `build_dataset --max-seq-len` 要和对应档位的 `max_seq_len` 保持一致。

---

## 八、后续衍生路线

这个方向的可复用性很强，后续版本不需要重学技术栈：

| 版本 | 做什么 | 复用情况 |
|------|--------|----------|
| 第二版 | 换学科（高等数学 / 英语 / 编程课） | 模型、后端、前端全部复用，只换题库与知识点图 | 
| 第三版 | 拍照做题，引入文字识别与多模态 | 新增输入通道 |
| 第四版 | 错因分析（不只判断对错，判断错在哪一步） | 模型层扩展 |
| 第五版 | 跨学科迁移分析（一门课的掌握对另一门课的影响） | 数据层扩展 |

---

## 九、诚实声明（答辩前必读）

- **已有商业化先例**：松鼠 Ai 的认知诊断测评正是基于"知识空间理论 + 贝叶斯知识追踪"
  构建个体知识图谱，公开信息称已覆盖全国六万余所学校、服务五千万以上学生。
  被追问时的定位说明：本项目做的是**单学科、开源、可解释、带完整对比实验的轻量系统**，
  是研究性作品，重点是把方法讲清楚、把实验数据摆出来，不与商业产品竞争功能覆盖面。
- **模型之间差距可能很小**：已有实证研究（JEDM, 2022）在多个公开数据集上系统评测后指出，
  深度模型充分调参后整体优于传统模型，但**模型间相对差异很细微且随数据集变化，
  没有任何一个模型能在所有数据集上占优**；在部分数据集上连最简单的均值预测基线都可能
  在准确率上超过复杂模型。所以本项目把"多模型对比实验"设为核心产出，
  并用 bootstrap 置信区间判断"差距是否真实存在"，而不是照搬论文结论。
  报告里的自动生成结论段会明确写出这一点，不掩盖不利结果。
- **演示数据是模拟的，但实验指标不是**：`backend/app/seed.py` 预填的作答历史是按模拟掌握度
  生成的，不是真实学生数据，只能用来演示交互流程。而第二节表里的五个模型指标，
  是在公开数据集 ASSISTments 2009 上真实跑出来的（源文件、过滤参数、丢弃行数都记在
  `ml/artifacts/report_assistments2009.md` 的「数据来源与预处理」一节，可追溯）。
  被问"数据哪来的"直接指那一节，不要说"用了真实学校数据"。
- **合成数据不报结论**：`ml/artifacts/results_synthetic.json` 里的指标是自拟课程 + 模拟学生
  跑出来的（BKT 0.56 / DKT 0.54 那一档），只证明流水线通了，**不能当答辩结论**。
  界面上会自动打警示条；只要公开数据集的结果文件存在，界面就会自动切过去。
