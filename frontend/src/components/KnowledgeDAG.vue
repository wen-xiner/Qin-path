<script setup>
/**
 * 知识点依赖图（分层有向图，SVG 手绘）。
 * 横轴 = 学习层级（layers），纵轴 = 同层知识点。
 * 结点颜色 = 掌握度：绿=已掌握、橙=薄弱、灰=未观测。
 * 边 = 前置依赖，画成贝塞尔曲线。
 */
import { computed } from 'vue'

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  layers: { type: Array, default: () => [] },
  mastery: { type: Object, default: () => ({}) }, // kc_id -> 掌握度 or null
  selected: { type: String, default: '' },
  threshold: { type: Number, default: 0.7 },
})

const emit = defineEmits(['select'])

const NODE_W = 128
const NODE_H = 34
const COL_GAP = 74
const ROW_GAP = 14
const PAD_X = 16
const PAD_Y = 14

const nodeMap = computed(() => {
  const m = {}
  for (const n of props.nodes) m[n.kc_id] = n
  return m
})

const layout = computed(() => {
  const layers = props.layers.length
    ? props.layers
    : [props.nodes.map((n) => n.kc_id)]
  const pos = {}
  const maxRows = Math.max(...layers.map((l) => l.length), 1)
  const height = PAD_Y * 2 + maxRows * (NODE_H + ROW_GAP) - ROW_GAP
  layers.forEach((layer, col) => {
    const totalH = layer.length * (NODE_H + ROW_GAP) - ROW_GAP
    const startY = PAD_Y + (height - PAD_Y * 2 - totalH) / 2
    layer.forEach((kcId, row) => {
      pos[kcId] = {
        x: PAD_X + col * (NODE_W + COL_GAP),
        y: startY + row * (NODE_H + ROW_GAP),
        col,
      }
    })
  })
  const width = PAD_X * 2 + layers.length * NODE_W + (layers.length - 1) * COL_GAP
  return { pos, width, height, layers }
})

const edges = computed(() => {
  const out = []
  for (const n of props.nodes) {
    const to = layout.value.pos[n.kc_id]
    if (!to) continue
    for (const p of n.prerequisites || []) {
      const from = layout.value.pos[p]
      if (!from) continue
      const x1 = from.x + NODE_W
      const y1 = from.y + NODE_H / 2
      const x2 = to.x
      const y2 = to.y + NODE_H / 2
      const mx = (x1 + x2) / 2
      out.push({
        key: `${p}->${n.kc_id}`,
        d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
      })
    }
  }
  return out
})

function colorOf(kcId) {
  const v = props.mastery[kcId]
  if (v === undefined || v === null) return { fill: '#f1f5f9', stroke: '#cbd5e1', text: '#64748b' }
  if (v >= props.threshold) return { fill: '#d1fae5', stroke: '#059669', text: '#065f46' }
  if (v >= props.threshold * 0.6) return { fill: '#fef3c7', stroke: '#d97706', text: '#92400e' }
  return { fill: '#fee2e2', stroke: '#dc2626', text: '#991b1b' }
}

function label(kcId) {
  const n = nodeMap.value[kcId]
  if (!n) return kcId
  const v = props.mastery[kcId]
  const suffix = v === undefined || v === null ? '' : ` ${(v * 100).toFixed(0)}%`
  return n.name.length > 7 ? n.name.slice(0, 7) + '…' + suffix : n.name + suffix
}
</script>

<template>
  <div class="dag-scroll">
    <svg
      :width="layout.width"
      :height="layout.height"
      :viewBox="`0 0 ${layout.width} ${layout.height}`"
      class="dag"
    >
      <path v-for="e in edges" :key="e.key" :d="e.d" class="edge" />
      <g
        v-for="n in nodes"
        :key="n.kc_id"
        @click="emit('select', n.kc_id)"
        class="node-group"
      >
        <title>{{ n.name }}（{{ n.chapter }}）
掌握概率：{{ mastery[n.kc_id] === null || mastery[n.kc_id] === undefined ? '未观测' : mastery[n.kc_id].toFixed(2) }}</title>
        <rect
          :x="layout.pos[n.kc_id]?.x"
          :y="layout.pos[n.kc_id]?.y"
          :width="NODE_W"
          :height="NODE_H"
          rx="7"
          :fill="colorOf(n.kc_id).fill"
          :stroke="selected === n.kc_id ? '#2563eb' : colorOf(n.kc_id).stroke"
          :stroke-width="selected === n.kc_id ? 2.5 : 1.2"
        />
        <text
          :x="(layout.pos[n.kc_id]?.x || 0) + NODE_W / 2"
          :y="(layout.pos[n.kc_id]?.y || 0) + NODE_H / 2 + 4"
          text-anchor="middle"
          :fill="colorOf(n.kc_id).text"
          class="node-label"
        >
          {{ label(n.kc_id) }}
        </text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.dag-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 6px;
}

.edge {
  fill: none;
  stroke: #cbd5e1;
  stroke-width: 1.1;
}

.node-group {
  cursor: pointer;
}

.node-group:hover rect {
  filter: brightness(0.96);
}

.node-label {
  font-size: 11px;
  pointer-events: none;
  user-select: none;
}
</style>
