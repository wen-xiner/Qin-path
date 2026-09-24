<script setup>
/** 学生端：做题 → 掌握度更新 → 推荐 → 看理由，完整闭环。 */
import { computed, onMounted, ref } from 'vue'
import { api } from '../api/client'
import KnowledgeDAG from '../components/KnowledgeDAG.vue'
import LearningCurve from '../components/LearningCurve.vue'

const props = defineProps({
  meta: { type: Object, default: null },
})

const loading = ref(true)
const busy = ref(false)
const error = ref('')

const mastery = ref(null)
const graph = ref({ nodes: [], layers: [] })
const rec = ref(null)
const chosen = ref(null)
const result = ref(null)
const curve = ref({ points: [] })
const selectedKc = ref('')
const history = ref([])

const threshold = computed(() => mastery.value?.threshold ?? 0.7)

const masteryMap = computed(() => {
  const m = {}
  for (const it of mastery.value?.items || []) m[it.kc_id] = it.mastery
  return m
})

const stats = computed(() => {
  const items = mastery.value?.items || []
  return {
    answered: items.reduce((s, i) => s + i.attempts, 0),
    observed: items.filter((i) => i.state !== 'unobserved').length,
    weak: items.filter((i) => i.state === 'weak').length,
    mastered: items.filter((i) => i.state === 'mastered').length,
    total: items.length,
  }
})

const plan = computed(() => mastery.value?.plan || null)

async function loadAll() {
  loading.value = true
  error.value = ''
  try {
    const [m, g] = await Promise.all([api.mastery(), api.knowledgeGraph()])
    mastery.value = m
    graph.value = g
    await Promise.all([loadRecommendation(), loadCurve(), loadHistory()])
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function loadRecommendation() {
  rec.value = await api.nextQuestion()
  chosen.value = null
  result.value = null
}

async function loadCurve(kcId = '') {
  curve.value = await api.curve(kcId || undefined)
}

async function loadHistory() {
  history.value = await api.history(12)
}

async function submit() {
  if (chosen.value === null || !rec.value) return
  busy.value = true
  error.value = ''
  try {
    const body = await api.answer(rec.value.question.question_id, chosen.value)
    result.value = body
    const [m] = await Promise.all([api.mastery(), loadCurve(selectedKc.value), loadHistory()])
    mastery.value = m
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

async function nextOne() {
  busy.value = true
  try {
    rec.value = result.value?.next || (await api.nextQuestion())
    chosen.value = null
    result.value = null
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

async function onSelectKc(kcId) {
  selectedKc.value = selectedKc.value === kcId ? '' : kcId
  await loadCurve(selectedKc.value)
}

function stateTag(state) {
  return { mastered: 'tag ok', weak: 'tag bad', unobserved: 'tag gray' }[state] || 'tag'
}

function stateText(state) {
  return { mastered: '已掌握', weak: '薄弱', unobserved: '未观测' }[state] || state
}

function actionText(action) {
  return (
    {
      remedy: '补强',
      remedy_prerequisite: '先补前置',
      advance: '推进新知识点',
    }[action] || action
  )
}

onMounted(loadAll)
</script>

<template>
  <div v-if="loading" class="panel">正在加载…</div>

  <div v-else class="grid">
    <div v-if="error" class="panel err-panel">{{ error }}</div>

    <!-- 顶部统计 -->
    <div class="stat-row">
      <div class="panel stat">
        <div class="stat-num">{{ stats.answered }}</div>
        <div class="muted">累计作答</div>
      </div>
      <div class="panel stat">
        <div class="stat-num">{{ stats.observed }} <span class="of">/ {{ stats.total }}</span></div>
        <div class="muted">已观测知识点</div>
      </div>
      <div class="panel stat">
        <div class="stat-num bad">{{ stats.weak }}</div>
        <div class="muted">薄弱知识点</div>
      </div>
      <div class="panel stat">
        <div class="stat-num ok">{{ stats.mastered }}</div>
        <div class="muted">已掌握知识点</div>
      </div>
    </div>

    <div class="two-col">
      <!-- 左：做题 + 推荐理由 -->
      <div class="col">
        <div class="panel" v-if="rec">
          <h3 class="panel-title">
            推荐练习
            <span class="tag">{{ actionText(rec.action) }}</span>
            <span class="tag gray">{{ rec.question.kc_name }}</span>
            <span class="tag gray">难度 {{ rec.question.difficulty }}</span>
          </h3>
          <p class="panel-sub">按你的掌握度分布挑的题，理由见下方「为什么推荐这道题」</p>

          <div class="stem">{{ rec.question.stem }}</div>

          <div class="options">
            <button
              v-for="(opt, i) in rec.question.options"
              :key="i"
              class="option"
              :class="{
                picked: chosen === i && !result,
                right: result && i === result.correct_index,
                wrong: result && chosen === i && !result.is_correct,
              }"
              :disabled="!!result || busy"
              @click="chosen = i"
            >
              <span class="opt-label">{{ 'ABCD'[i] }}</span>
              <span>{{ opt }}</span>
            </button>
          </div>

          <div class="actions">
            <button
              v-if="!result"
              class="btn-primary"
              :disabled="chosen === null || busy"
              @click="submit"
            >
              {{ busy ? '提交中…' : '提交答案' }}
            </button>
            <template v-else>
              <span class="feedback" :class="result.is_correct ? 'ok-text' : 'bad-text'">
                {{ result.feedback }}
              </span>
              <span v-if="result.mastery_delta !== null" class="tag" :class="result.mastery_delta >= 0 ? 'ok' : 'bad'">
                掌握度 {{ result.mastery_delta >= 0 ? '+' : '' }}{{ result.mastery_delta }}
              </span>
              <button class="btn-primary" :disabled="busy" @click="nextOne">下一题</button>
            </template>
          </div>
        </div>

        <div class="panel" v-if="rec">
          <h3 class="panel-title">为什么推荐这道题</h3>
          <p class="panel-sub">推荐理由由规则与模型状态共同生成，每个数字都可核查</p>
          <blockquote class="reason">{{ rec.reason }}</blockquote>
          <div class="evidence">
            <div class="ev-item">
              <span class="muted">知识点掌握概率</span>
              <strong>{{ rec.evidence.mastery === null ? '未观测' : rec.evidence.mastery }}</strong>
            </div>
            <div class="ev-item">
              <span class="muted">判定阈值</span>
              <strong>{{ rec.evidence.threshold }}</strong>
            </div>
            <div class="ev-item">
              <span class="muted">该知识点作答/答对</span>
              <strong>{{ rec.evidence.attempts }} / {{ rec.evidence.correct }}</strong>
            </div>
            <div class="ev-item">
              <span class="muted">模型预测答对概率</span>
              <strong>
                {{
                  rec.evidence.predicted_correct_prob === null
                    ? '暂无历史'
                    : rec.evidence.predicted_correct_prob
                }}
              </strong>
            </div>
          </div>
        </div>

        <!-- 学习路径 -->
        <div class="panel" v-if="plan">
          <h3 class="panel-title">下一步学习路径</h3>
          <p class="panel-sub">
            规则显式：前置没达标就先补前置；前置都达标且自己薄弱就补强；从未学过且前置达标才推进新知识点
          </p>
          <ol class="plan">
            <li v-for="(item, i) in plan.queue" :key="item.kc_id">
              <span class="idx">{{ i + 1 }}</span>
              <div class="plan-body">
                <div class="plan-head">
                  <strong>{{ item.name }}</strong>
                  <span class="tag" :class="item.action === 'advance' ? '' : 'warn'">
                    {{ actionText(item.action) }}
                  </span>
                  <span class="muted">{{ item.chapter }}</span>
                  <span class="muted mono">
                    掌握度 {{ item.mastery === null ? '未观测' : item.mastery }}
                  </span>
                </div>
                <div class="muted plan-reason">{{ item.reason }}</div>
              </div>
            </li>
          </ol>
          <div v-if="plan.blocked.length" class="blocked">
            <div class="muted">被前置阻塞、暂不推进的知识点：</div>
            <ul>
              <li v-for="b in plan.blocked" :key="b.kc_id">
                {{ b.name }} —— {{ b.reason }}
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 右：可视化 -->
      <div class="col">
        <div class="panel">
          <h3 class="panel-title">知识点依赖图</h3>
          <p class="panel-sub">
            横轴为学习层级，箭头为前置依赖；颜色越绿掌握越好，灰色表示还没练过（未被观测 ≠ 薄弱）
          </p>
          <KnowledgeDAG
            :nodes="graph.nodes"
            :layers="graph.layers"
            :mastery="masteryMap"
            :threshold="threshold"
            :selected="selectedKc"
            @select="onSelectKc"
          />
        </div>

        <div class="panel">
          <h3 class="panel-title">
            掌握度变化曲线
            <span class="tag gray" v-if="selectedKc">
              {{ graph.nodes.find((n) => n.kc_id === selectedKc)?.name }}
            </span>
            <button v-if="selectedKc" class="btn-ghost mini" @click="onSelectKc(selectedKc)">
              看总体
            </button>
          </h3>
          <p class="panel-sub">
            {{ selectedKc ? '该知识点掌握概率随作答次数的变化' : '全部已观测知识点的平均掌握概率变化' }}
          </p>
          <LearningCurve :points="curve.points" :threshold="threshold" />
        </div>

        <div class="panel">
          <h3 class="panel-title">知识点掌握明细</h3>
          <div class="chips">
            <div
              v-for="it in mastery.items"
              :key="it.kc_id"
              class="chip"
              :class="[it.state, selectedKc === it.kc_id ? 'sel' : '']"
              @click="onSelectKc(it.kc_id)"
            >
              <div class="chip-name">{{ it.name }}</div>
              <div class="chip-val">
                {{ it.mastery === null ? '—' : it.mastery.toFixed(2) }}
                <span class="muted">({{ it.correct }}/{{ it.attempts }})</span>
              </div>
              <span :class="stateTag(it.state)">{{ stateText(it.state) }}</span>
            </div>
          </div>
        </div>

        <div class="panel">
          <h3 class="panel-title">最近作答</h3>
          <table class="mini-table">
            <thead>
              <tr>
                <th>#</th>
                <th>知识点</th>
                <th>结果</th>
                <th>掌握度</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="h in history" :key="h.seq_no">
                <td class="mono">{{ h.seq_no }}</td>
                <td>{{ h.kc_name }}</td>
                <td>
                  <span class="tag" :class="h.is_correct ? 'ok' : 'bad'">
                    {{ h.is_correct ? '对' : '错' }}
                  </span>
                </td>
                <td class="mono">{{ h.mastery_after === null ? '—' : h.mastery_after.toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
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

.stat-num .of {
  font-size: 14px;
  color: var(--text-dim);
  font-weight: 400;
}

.stat-num.ok {
  color: var(--ok);
}

.stat-num.bad {
  color: var(--bad);
}

.two-col {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

@media (max-width: 1100px) {
  .two-col {
    grid-template-columns: 1fr;
  }
}

.col {
  display: grid;
  gap: 16px;
}

.stem {
  font-size: 15px;
  line-height: 1.8;
  padding: 12px 14px;
  background: #f8fafc;
  border-left: 3px solid var(--primary);
  border-radius: 6px;
  margin-bottom: 14px;
}

.options {
  display: grid;
  gap: 8px;
}

.option {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  text-align: left;
  background: #fff;
  border: 1px solid var(--border);
  padding: 10px 14px;
  color: var(--text);
  line-height: 1.6;
}

.option:hover:not(:disabled) {
  border-color: var(--primary);
}

.option.picked {
  border-color: var(--primary);
  background: var(--primary-dim);
}

.option.right {
  border-color: var(--ok);
  background: var(--ok-dim);
}

.option.wrong {
  border-color: var(--bad);
  background: var(--bad-dim);
}

.opt-label {
  font-weight: 700;
  color: var(--text-dim);
  flex: 0 0 18px;
}

.actions {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.feedback {
  font-size: 13px;
}

.ok-text {
  color: var(--ok);
}

.bad-text {
  color: var(--bad);
}

.reason {
  margin: 0;
  padding: 12px 14px;
  background: #f8fafc;
  border-left: 3px solid var(--warn);
  border-radius: 6px;
  font-size: 13.5px;
  line-height: 1.8;
  color: #374151;
}

.evidence {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-top: 14px;
}

.ev-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 12px;
  background: #f8fafc;
  border-radius: 6px;
  font-size: 12.5px;
}

.plan {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
}

.plan li {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

.idx {
  flex: 0 0 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--primary-dim);
  color: var(--primary);
  font-size: 12px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

.plan-body {
  flex: 1;
  min-width: 0;
}

.plan-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.plan-reason {
  margin-top: 2px;
  line-height: 1.7;
}

.blocked {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px dashed var(--border);
  font-size: 12.5px;
}

.blocked ul {
  margin: 6px 0 0;
  padding-left: 18px;
  color: var(--text-dim);
}

.chips {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 8px;
}

.chip {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  cursor: pointer;
  background: #fff;
  transition: all 0.15s;
}

.chip:hover {
  border-color: var(--primary);
}

.chip.sel {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--primary-dim);
}

.chip.mastered {
  background: var(--ok-dim);
  border-color: #a7f3d0;
}

.chip.weak {
  background: var(--bad-dim);
  border-color: #fecaca;
}

.chip.unobserved {
  background: #f8fafc;
}

.chip-name {
  font-size: 12.5px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chip-val {
  font-size: 12px;
  margin: 2px 0 4px;
  font-family: ui-monospace, monospace;
}

.mini-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}

.mini-table th {
  text-align: left;
  font-weight: 500;
  color: var(--text-dim);
  padding: 6px 8px;
  border-bottom: 1px solid var(--border);
}

.mini-table td {
  padding: 6px 8px;
  border-bottom: 1px solid #f1f5f9;
}

.mini {
  padding: 3px 10px;
  font-size: 12px;
}
</style>
