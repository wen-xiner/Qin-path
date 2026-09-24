<script setup>
/** 教师端：班级掌握度分布 + 结构性薄弱点定位 + 模型实验结论。 */
import { computed, onMounted, ref } from 'vue'
import { api } from '../api/client'
import ClassHeatmap from '../components/ClassHeatmap.vue'
import ExperimentTable from '../components/ExperimentTable.vue'

const loading = ref(true)
const error = ref('')
const data = ref(null)
const students = ref([])
const experiment = ref(null)

const kcNames = computed(() => (data.value?.kc || []).map((k) => k.name))
const threshold = computed(() => data.value?.threshold ?? 0.7)

const classStats = computed(() => {
  const kc = data.value?.kc || []
  const observed = kc.filter((k) => k.mean_mastery !== null)
  return {
    students: data.value?.n_students ?? 0,
    weakPoints: data.value?.class_weak_points?.length ?? 0,
    meanMastery: observed.length
      ? (observed.reduce((s, k) => s + k.mean_mastery, 0) / observed.length).toFixed(3)
      : '—',
    coverage: kc.length ? `${observed.length} / ${kc.length}` : '—',
  }
})

async function load() {
  loading.value = true
  error.value = ''
  try {
    const [cm, st, models] = await Promise.all([
      api.classMastery(),
      api.students(),
      api.models(),
    ])
    data.value = cm
    students.value = st
    experiment.value = models.experiments
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div v-if="loading" class="panel">正在加载…</div>

  <div v-else class="grid">
    <div v-if="error" class="panel err-panel">{{ error }}</div>

    <div class="stat-row">
      <div class="panel stat">
        <div class="stat-num">{{ classStats.students }}</div>
        <div class="muted">班级学生数</div>
      </div>
      <div class="panel stat">
        <div class="stat-num">{{ classStats.meanMastery }}</div>
        <div class="muted">班级平均掌握度</div>
      </div>
      <div class="panel stat">
        <div class="stat-num bad">{{ classStats.weakPoints }}</div>
        <div class="muted">共性薄弱知识点</div>
      </div>
      <div class="panel stat">
        <div class="stat-num">{{ classStats.coverage }}</div>
        <div class="muted">知识点覆盖（有数据 / 总数）</div>
      </div>
    </div>

    <!-- 结构性薄弱点 -->
    <div class="panel">
      <h3 class="panel-title">
        班级共性薄弱点
        <span class="tag bad">整体卡住的地方</span>
      </h3>
      <p class="panel-sub">
        判定规则：该知识点至少有 30% 的学生作答过，且班级平均掌握概率低于 {{ threshold }}。
        这是「看平均分看不到，但确实存在」的结构性问题。
      </p>
      <div v-if="!data.class_weak_points.length" class="muted">
        暂无满足条件的薄弱点（可能因为作答数据太少，或班级整体掌握良好）。
      </div>
      <ul v-else class="weak-list">
        <li v-for="w in data.class_weak_points" :key="w.kc_id">
          <div class="weak-head">
            <strong>{{ w.name }}</strong>
            <span class="tag gray">{{ w.chapter }}</span>
          </div>
          <div class="weak-meta">
            <span class="muted">
              平均掌握 <strong class="bad-text">{{ w.mean_mastery.toFixed(3) }}</strong>
            </span>
            <span class="muted">
              未达标 <strong>{{ w.n_below }}</strong> / {{ w.n_observed }} 人
              （{{ (w.weak_ratio * 100).toFixed(0) }}%）
            </span>
          </div>
          <div class="bar">
            <div class="bar-fill" :style="{ width: `${Math.min(100, w.weak_ratio * 100)}%` }"></div>
          </div>
        </li>
      </ul>
    </div>

    <!-- 热力图 -->
    <div class="panel">
      <h3 class="panel-title">全班掌握度热力图</h3>
      <p class="panel-sub">纵轴为学生，横轴为知识点（按学习层级顺序），颜色越红越薄弱、越绿越掌握</p>
      <ClassHeatmap :per-student="data.per_student" :kc-names="kcNames" :threshold="threshold" />
    </div>

    <!-- 学生列表 -->
    <div class="panel">
      <h3 class="panel-title">学生明细</h3>
      <p class="panel-sub">按薄弱知识点数量排序，方便优先关注</p>
      <table class="mini-table">
        <thead>
          <tr>
            <th>学生</th>
            <th>作答数</th>
            <th>正确率</th>
            <th>平均掌握度</th>
            <th>薄弱知识点</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in [...students].sort((a, b) => b.n_weak - a.n_weak)" :key="s.student_id">
            <td>{{ s.display_name || s.username }}</td>
            <td class="mono">{{ s.n_answered }}</td>
            <td class="mono">{{ s.accuracy === null ? '—' : (s.accuracy * 100).toFixed(1) + '%' }}</td>
            <td class="mono">{{ s.mean_mastery === null ? '—' : s.mean_mastery.toFixed(3) }}</td>
            <td>
              <span class="tag" :class="s.n_weak > 3 ? 'bad' : s.n_weak > 0 ? 'warn' : 'ok'">
                {{ s.n_weak }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 实验结论 -->
    <div class="panel">
      <h3 class="panel-title">模型对比实验结果</h3>
      <p class="panel-sub">
        三个知识追踪模型（BKT / DKT / SAKT）加两个非学习基线，在同一数据集、同一套评测下对比
      </p>
      <ExperimentTable :experiment="experiment" />
    </div>
  </div>
</template>

<style scoped>
.err-panel {
  border-color: var(--bad);
  background: var(--bad-dim);
  color: var(--bad);
}

.stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}

.stat {
  padding: 14px 18px;
}

.stat-num {
  font-size: 26px;
  font-weight: 700;
  color: var(--primary);
}

.stat-num.bad {
  color: var(--bad);
}

.weak-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px;
}

.weak-list li {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fff;
}

.weak-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.weak-meta {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  font-size: 12.5px;
  margin-bottom: 8px;
}

.bad-text {
  color: var(--bad);
}

.bar {
  height: 6px;
  background: #f1f5f9;
  border-radius: 3px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #fbbf24, #dc2626);
  border-radius: 3px;
}

.mini-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.mini-table th {
  text-align: left;
  font-weight: 500;
  color: var(--text-dim);
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}

.mini-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #f1f5f9;
}
</style>
