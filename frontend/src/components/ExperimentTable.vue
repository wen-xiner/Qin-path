<script setup>
/** 模型对比实验结果表。把实验数据和系统放在同一个界面，避免「实验是实验、系统是系统」。 */
const props = defineProps({
  experiment: { type: Object, default: null },
})

function fmt(v, digits = 4) {
  return v === null || v === undefined ? '—' : Number(v).toFixed(digits)
}

const SPLIT_LABELS = { train: '训练集', valid: '验证集', test: '测试集' }
const SPLIT_NOTES = {
  train: '训练模型参数',
  valid: '早停与选型（按 AUC）',
  test: '只用于报告最终指标，不参与任何训练决策',
}

function splitLabel(key) {
  return SPLIT_LABELS[key] ?? key
}

function splitNote(key) {
  return SPLIT_NOTES[key] ?? ''
}
</script>

<template>
  <div>
    <div v-if="!experiment" class="muted">
      还没有实验数据。运行 <span class="mono">python -m ml.run_experiments</span> 后这里会显示结果。
    </div>
    <template v-else>
      <div v-if="experiment.is_synthetic" class="warn-bar">
        当前展示的是<strong>合成数据集</strong>的结果，仅用于验证流水线是否跑通，
        不作为答辩结论。答辩指标请用公开学术数据集跑（见 README「接入公开数据集」一节）。
      </div>
      <table class="exp-table">
        <thead>
          <tr>
            <th>模型</th>
            <th>AUC</th>
            <th>95% 置信区间</th>
            <th>准确率</th>
            <th>RMSE</th>
            <th>NLL</th>
            <th>参数量</th>
            <th>训练耗时(s)</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in experiment.table" :key="row.model">
            <td>
              <strong>{{ row.model }}</strong>
            </td>
            <td class="mono hi">{{ fmt(row.auc) }}</td>
            <td class="mono muted">
              {{ row.auc_ci95 ? `[${fmt(row.auc_ci95[0])}, ${fmt(row.auc_ci95[1])}]` : '—' }}
            </td>
            <td class="mono">{{ fmt(row.acc) }}</td>
            <td class="mono">{{ fmt(row.rmse) }}</td>
            <td class="mono">{{ fmt(row.nll) }}</td>
            <td class="mono">{{ row.params ?? '不适用' }}</td>
            <td class="mono">{{ row.seconds ?? '—' }}</td>
          </tr>
        </tbody>
      </table>
      <p class="muted" style="margin-top: 8px">
        <strong>口径说明</strong>：本表是「三个知识追踪模型 + 两个非学习基线」在
        <strong>{{ experiment.data_source }}</strong>
        <template v-if="experiment.n_kc">
          （{{ experiment.n_kc }} 个知识点 / {{ experiment.n_students }} 名学生 /
          {{ experiment.n_interactions }} 条交互）
        </template>
        这份<strong>公开学术数据集</strong>上的对比结果，与上方的课程演示数据相互独立——
        演示课程是自拟的《数据结构》知识点体系（28 个），公开数据集用的是它自己的知识点划分。
        <template v-if="experiment.profile">
          训练设备 {{ experiment.device }}（档位 {{ experiment.profile }}）。</template>
        结果文件：{{ experiment.source_file }}。
      </p>
      <p class="muted" style="margin-top: 6px">
        AUC 为逐交互口径（含每条序列第一题）；置信区间按学生 bootstrap 得到，
        区间大幅重叠时不能声称模型之间有明显差异。
      </p>
      <table v-if="experiment.splits" class="exp-table" style="margin-top: 10px">
        <thead>
          <tr>
            <th>切分</th>
            <th>学生数</th>
            <th>交互数</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(s, key) in experiment.splits" :key="key">
            <td><strong>{{ splitLabel(key) }}</strong></td>
            <td class="mono">{{ s.n_students }}</td>
            <td class="mono">{{ s.n_interactions }}</td>
            <td class="muted">{{ splitNote(key) }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </div>
</template>

<style scoped>
.exp-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}

.exp-table th {
  text-align: left;
  color: var(--text-dim);
  font-weight: 500;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}

.exp-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #f1f5f9;
  white-space: nowrap;
}

.exp-table td.hi {
  color: var(--primary);
  font-weight: 600;
}

.warn-bar {
  background: var(--warn-dim);
  color: #92400e;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 12.5px;
  margin-bottom: 12px;
  line-height: 1.7;
}
</style>
