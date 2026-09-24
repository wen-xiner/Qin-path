# BUDDYREADME —— 迁移到新机器后，先读这份

> **给接手的 AI 助手（buddy）看的。**
> 第一件事**不是改代码，是确认环境到底有没有让显卡生效**。
>
> 这份文件是「交接载体」。原来的开发工作记忆（`.workbuddy/memory/`）在仓库之外，
> **不会随代码迁移过来**，所以关键约定和现状都写在这里，不要凭空假设。
> 面向人类的完整说明在 `README.md`，本文件只补充「接手时要先做什么、不能碰什么」。

---

## 0. 三十秒了解现状

- 项目：**知途（Qin-path）**，知识追踪与自适应学习路径系统。挑战杯参赛作品。
- 仓库：<https://github.com/wen-xiner/Qin-path>（公开）
- 技术栈：PyTorch（BKT 自实现 / DKT / SAKT）+ FastAPI + SQLAlchemy + Vue3 + Vite + ECharts。
- **第一版已经做完，且指标是真的**：五个模型（Mean / KC-Mean / BKT / DKT / SAKT）在公开数据集
  **ASSISTments 2009** 上跑出的测试集 AUC —— DKT **0.8092** 最高。
- **但那是 CPU 档跑出来的**（`max_seq_len=100`、`epochs=60`、`d_model=48`）。
  这台新机器有显卡，**本次迁移的核心任务就是在 `gpu` 档重跑一遍，并把文档里的数字换掉**。
- 提交历史：`main` 分支。首次提交 87 文件 / 11300 行（本文件与 `scripts/check_env.py` 是之后补的）。

---

## 1. 第一件事：跑环境自检

```bash
cd Qin-path                 # 必须先 cd 到项目根，后面所有 python -m ml.xxx 都依赖工作目录
python scripts/check_env.py
```

它做 27 项只读检查（不训练、不写文件），**只用标准库**，所以依赖没装齐时也能跑起来、
并明确告诉你缺什么。输出最后会直接给一段「结论」。

### 怎么读结论

| 输出里的结论 | 含义 | 下一步 |
|---|---|---|
| `环境就绪，且已自动切到 gpu 档` | CUDA 生效了 | 直接进第 2 节 |
| `CUDA 可用，但当前档位仍是 cpu` | 被 `ZT_PROFILE` / `--profile` 固定住了 | 去掉环境变量，或每条命令显式加 `--profile gpu` |
| `没有可用的 CUDA` | 显卡没生效 | **先排查，别急着跑实验**，见下 |
| `环境不完整：必修依赖导不进来` | 依赖没装 | `pip install -r requirements.txt`（有显卡的机器用 `requirements-gpu.txt` 的步骤） |

### 「没有可用的 CUDA」时按顺序排查

1. **看自检输出里的「torch 编译的 CUDA 版本」**：如果是 `None`，那就是装成了 CPU 版 torch。
   重装（按驱动支持的版本选一条）：
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   # cu121 / cu126 同理
   ```
2. 显卡驱动是否正常：`nvidia-smi` 能不能列出显卡。
3. 驱动版本与 torch 要求的 CUDA 运行时是否匹配。

> **止步条件**：CUDA 没确认可用之前，不要跑对比实验。
> CPU 档和 GPU 档是两组不同的实验（数据量、序列长度、轮次、模型宽度都不同），
> 数字不能混用，混了答辩会被追问。

### 另外两个自检会提示、但容易忽略的点

- **`当前工作目录` 不是项目根** → `python -m ml.xxx` 会报 `No module named ml`。先 `cd`。
- **`切分（合成 / 演示）` 缺失** → 后端会起不来。原因见第 2 节第 2 步。

---

## 2. 迁移三步（CUDA 确认可用之后再动手）

先说清楚**为什么必须重建**：`.gitignore` 把 `data/raw/` 与 `data/processed/` 排除了
（原始 CSV 有 63 MB，不想塞进仓库）。所以**新机器 clone 下来是没有数据集的**，
而 `ml/service.py` 的 `KTService.create()` 第一句就是 `load_splits()`，
**没有切分文件后端直接起不来**（`main.py` 的 lifespan 只告警不阻塞，症状是接口逐个报错，
容易误判成"后端坏了"）。

### 第 1 步：拿到真实数据集（63,455,745 bytes）

```bash
# 方式 A：重新下载（USTC 镜像，直连原始 zip，不依赖 EduData）
curl -L -o data/raw/assist2009_skill_builder.zip \
    http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip
python -c "import zipfile;zipfile.ZipFile('data/raw/assist2009_skill_builder.zip').extractall('data/raw/')"

# 方式 B：直接从旧机器拷 data/raw/skill_builder_data_corrected.csv（更快更稳）
```

**校验**：解压出来的 `data/raw/skill_builder_data_corrected.csv` 应为 **63,455,745 字节**。
对不上就说明下载不完整，别往下走。

> 这个镜像**会间歇性被网络拦截**。判断方法：`curl -I` 能拿到带 `Content-Length: 9084422`
> 的 200 就是好的；拿不到就换网络环境或改用方式 B。

### 第 2 步：重建切分（`--max-seq-len` 必须与 gpu 档的 200 对齐）

```bash
# 2a) 真实数据集（答辩指标用）
python -m ml.data.build_dataset --source assistments \
    --csv data/raw/skill_builder_data_corrected.csv \
    --name assistments2009 --prefix assistments2009 --max-seq-len 200

# 2b) 合成数据集（演示用；后端启动依赖它）
python -m ml.data.build_dataset --profile gpu
```

**为什么 `--max-seq-len` 必须是 200**：它同时决定数据集截断长度和 DKT/SAKT 能看多少步。
BKT 会在每条序列**全长**上评测，而神经网络只看前 `max_len` 步。
两者不一致，就等于五个模型不在同一批交互上比，汇总表看着整齐但结论站不住。
CPU 档该值是 100，GPU 档是 200，见 `ml/config.py`。

**验收**（这几个数是本机用 `--max-seq-len 200` 实测的，对不上说明命令敲错了）：

```
n_kc=116    n_students=3694    n_questions=17712    n_interactions=219557
avg_seq_len=59.44             accuracy=0.6303
切分按学生 80/10/10 → 2955 / 369 / 370
```

> `n_interactions` 会随 `max_seq_len` 变化（cap=100 时是 161403，cap=200 时是 219557），
> 但 `n_kc`、`n_students` 应当恒定。如果 `n_kc` 不是 116，先看下面第 2 步的 `--min-kc-count`。

### 第 3 步：跑实验，然后更新数字

```bash
# 3a) 五个模型在真实数据上的正式指标（本次的核心产出）
python -m ml.run_experiments --profile gpu --prefix assistments2009

# 3b) 合成数据的产物也重训一遍，否则后端会加载到 CPU 档的旧模型
python -m ml.run_experiments --profile gpu
```

耗时参考（CPU 档实测，seed=42）：BKT 293s、DKT 84s、SAKT 263s，全套约 11 分钟。
BKT 是纯 numpy 单线程，随知识点数增加而变慢，换机器不会变快；GPU 只是让 DKT/SAKT 更快。

跑完**必须同步更新这些地方的数字**（漏一个就会出现两份不一致的指标）：

| 位置 | 说明 |
|---|---|
| `ml/artifacts/report_assistments2009.md` | 自动生成，会被覆盖，不用手改 |
| `ml/artifacts/results_assistments2009.json` | 自动生成 |
| `README.md` 第二节「已跑出的正式结果」表 | **手工改**，目前是 CPU 档数字 |
| `README.md` 第六节 | 若档位/耗时描述有变，一起改 |
| 前端教师端实验表 | 自动读取 mtime 最新的 `results_*.json`，**前端零改动** |
| `docs/screenshots/*.png` | 数字变了要重新截图（`python scripts/shot.py`，先起前后端） |
| 仓库外的答辩 PPT / 文档 | **最容易忘**，迁移后要人工核对一遍 |

### 验收：怎么确认这次是真跑成了

- 报告开头应显示 `运行设备: cuda`、`档位: gpu`、`每条序列截断长度: 200`。
- 报告「数据来源与预处理」一节应与 CPU 版**完全一致**（同一份 CSV）：
  原始行数 `401756`、丢弃 `kc_empty=63755`、知识点保留 `116 / 123`。
- 五个模型 AUC 都应明显高于 0.5，且 DKT/SAKT 明显高于最强基线 KC-Mean。
- **红旗**：如果 DKT 的 AUC 掉到 0.5 附近，说明踩到了「用验证集 loss 选 checkpoint」的老坑
  （答题数据噪声大，loss 早早平台化，会回滚到没训起来的模型）。见 `README.md` 第七节第 1 条。
- 测试全绿：`python -m pytest ml/tests backend/tests -q` → 应为 **33 passed**。

---

## 3. 迁移前这台机器上的基线（CPU 档，seed=42，供对照）

真实数据集 ASSISTments 2009：**116 个知识点 / 3694 名学生 / 161403 条交互**，
整体正确率 0.626。切分 2955 / 369 / 370 名学生。测试集 370 人 / 16920 条交互。

| 模型 | AUC | AUC 95% CI | 准确率 | 训练耗时 |
|---|---|---|---|---|
| Mean（全局均值，最弱基线） | 0.5000 | [0.5000, 0.5000] | 0.6291 | 0.0s |
| KC-Mean（逐知识点频率，严格下限） | 0.6206 | [0.6019, 0.6393] | 0.6515 | 0.03s |
| BKT | 0.7177 | [0.6947, 0.7341] | 0.7005 | 293s |
| **DKT** | **0.8092** | [0.7917, 0.8255] | 0.7555 | 84s |
| SAKT | 0.7977 | [0.7779, 0.8160] | 0.7427 | 263s |

两条已经写进报告的结论，**重跑后要重新验证而不是照抄**：

1. DKT 比最强非学习基线（KC-Mean）高 **+0.1886 AUC** —— 知识状态建模确实带来增益。
2. **DKT 与 SAKT 的置信区间大幅重叠，不能声称二者有实质差异。**
   这与设计文档引用的 JEDM 2022 实证结论一致：模型间相对差异很细微且随数据集变化。
   **GPU 档重跑后如果区间仍然重叠，这句话必须继续保留**，不要为了好看改成"SAKT 略逊于 DKT"。

---

## 4. 迁移时不要碰的东西

这些是刻意做的设计决定，不是遗留问题，**改之前先问人**：

- **三条建模约定**：
  1. 「未观测 ≠ 薄弱」—— `BKT.mastery()` 对没做过的知识点返回 `NaN` 而不是 0；
  2. 深度模型**用验证集 AUC 选 checkpoint，不用验证集 loss**；
  3. **合成数据只能验证实现与演示，不能当答辩结论**（产物里写死 `data_source='synthetic'`）。
- **`README.md` 第九节「诚实声明」不能删**：松鼠 Ai 已有商业化先例、模型间差距可能很小、
  演示数据是模拟的。这是有意写出来的，被追问时是加分项。
- **两个数据集各管一段，讲的时候不要混**：
  产品演示跑自拟《数据结构》课程（28 知识点 + 224 题 + 模拟学生）；
  答辩指标跑 ASSISTments 2009（116 知识点）。**知识点数不同是刻意的，不是数据错乱。**
- **路径规划不用大模型**，规则全显式写在 `ml/planner.py`。答辩时经得起追问，别换成 LLM。
- **认证不引入 JWT 库**（bcrypt + HMAC-SHA256 紧凑 token），为了"看得懂"，不是技术债。
- **这个仓库是公开的**：不要往里加真实学生数据、密钥、`.env`。演示账号密码 `123456`
  是刻意公开的演示数据。

---

## 5. 已知的坑（详细版在 `README.md` 第七节）

接手前扫一眼，能省几小时：

1. **深度模型必须用验证集 AUC 选 checkpoint**，用 loss 会回滚到没训起来的模型（AUC≈0.5）。
2. **训练步数不足会让 DKT 退化成"预测均值"**，CPU 档要 60 epoch 才够。
3. **合成数据必须有成串练习结构**（`repeat_prob=0.55`），否则序列模型被不公平压制。
4. **ASSISTments 2009 的 CSV 不是 UTF-8**（cp1252/latin-1），按 utf-8 读会在第 2634 字节炸；
   适配器会按 `utf-8-sig → cp1252 → latin-1` 自动回退。别用 `list(reader)` 物化全部行（近 1 GB）。
5. **按频次过滤知识点时题目表会带"孤儿题目"**，导致 `KeyError`；回归测试在 `ml/tests/test_adapters.py`。
6. **数据集截断长度必须与模型 `max_len` 对齐**（就是第 2 节第 2 步强调的那件事）。
7. **前端访问要用 `localhost` 而不是 `127.0.0.1`**：本机 `127.0.0.1:5173` 曾被别的进程占用，
   而 Vite 绑在 IPv6 的 `localhost`；uvicorn 要绑 `--host 0.0.0.0` 才能两边都通。

---

## 6. 当前待办（按优先级）

第一版范围（设计文档第七节）已完成六条，**只剩一条没做**：

1. **演示脚本**：学生连续作答 6-8 题，掌握度实时变化、推荐路径随之调整。
   仓库里目前**没有**任何 demo 脚本文件。纯前端+接口，零算力需求。
2. **第二个公开数据集**：USTC 镜像同级有 `2012-2013-data-with-predictions-4-final.zip`、
   `2015_100_skill_builders_main_problems.zip`、`anonymized_full_release_competition_dataset.zip`。
   多跑一个数据集能把「模型间差异随数据集变化」从**引用别人的结论**变成**我们自己的实验发现**
   （注意原始 CSV 列名与 2009 版不一致，可能要往 `ml/data/adapters.py` 的 `CSV_ALIASES` 补别名）。
3. **消融 / 敏感性实验**（如 DKT 改用题目 id、不同 `max_len` 的指标曲线）。
4. **前端体积优化**：`ClassHeatmap.vue` / `LearningCurve.vue` 都是
   `import * as echarts from 'echarts'` 全量引入，单 chunk 1.13 MB，改按需引入可降到约 400-500 KB。
5. **不要做**：`docker-compose.yml` 在本机从未实测过（没有装 docker），要么装 Docker Desktop
   验证，要么留到目标机器；设计文档第八节的第四版（错因分析）、第五版（跨学科迁移）需要
   公开数据集里没有的标注，现在做不了。

---

## 7. 常用命令

```bash
# 环境自检（第一件事）
python scripts/check_env.py

# 数据集
python -m ml.data.build_dataset --profile gpu                      # 合成（演示）
python -m ml.data.build_dataset --source assistments \
    --csv data/raw/skill_builder_data_corrected.csv \
    --name assistments2009 --prefix assistments2009 --max-seq-len 200

# 实验（核心产出）
python -m ml.run_experiments --profile gpu --prefix assistments2009

# 后端 + 前端
python -m backend.app.seed                 # 演示账号 teacher / stu01..stu12，密码 123456
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev                 # http://localhost:5173（别用 127.0.0.1）
# 若 npm 脚本因工具链路径报错，直接调 vite：
#   node ./node_modules/vite/bin/vite.js --port 5173
python scripts/shot.py                     # 无头截图验收（学生端 + 教师端）

# 测试
python -m pytest ml/tests backend/tests -q # 33 条
```

环境变量：`ZT_PROFILE`（cpu/gpu）、`ZT_DEVICE`（auto/cpu/cuda/mps）、
`ZT_DATABASE_URL`、`ZT_SECRET_KEY`、`ZT_KT_MODEL`。完整说明见 `README.md` 第六节。
