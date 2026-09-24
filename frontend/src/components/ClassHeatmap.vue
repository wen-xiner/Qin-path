<script setup>
/** 全班掌握度热力图：纵轴学生、横轴知识点、颜色为掌握概率。 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  perStudent: { type: Array, default: () => [] }, // [{student_id, mastery: [...] }]
  kcNames: { type: Array, default: () => [] },
  threshold: { type: Number, default: 0.7 },
})

const el = ref(null)
let chart = null

function render() {
  if (!chart) return
  const data = []
  let min = 1
  let max = 0
  props.perStudent.forEach((s, y) => {
    s.mastery.forEach((v, x) => {
      if (v === null || v === undefined) return
      data.push([x, y, v])
      if (v < min) min = v
      if (v > max) max = v
    })
  })
  if (max <= min) {
    max = 1
    min = 0
  }

  chart.setOption(
    {
      grid: { left: 74, right: 20, top: 20, bottom: 90 },
      tooltip: {
        formatter: (p) =>
          `${props.perStudent[p.value[1]]?.student_id} 号学生<br/>${
            props.kcNames[p.value[0]]
          }<br/>掌握概率 ${Number(p.value[2]).toFixed(3)}`,
      },
      xAxis: {
        type: 'category',
        data: props.kcNames,
        axisLabel: {
          rotate: 60,
          fontSize: 10,
          color: '#6b7280',
          interval: 0,
        },
        axisLine: { lineStyle: { color: '#d1d5db' } },
        splitArea: { show: false },
      },
      yAxis: {
        type: 'category',
        data: props.perStudent.map((s) => `${s.student_id} 号`),
        axisLabel: { fontSize: 10, color: '#6b7280' },
        axisLine: { lineStyle: { color: '#d1d5db' } },
      },
      visualMap: {
        min,
        max,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 6,
        itemWidth: 12,
        itemHeight: 90,
        text: ['掌握好', '薄弱'],
        textStyle: { fontSize: 11, color: '#6b7280' },
        inRange: {
          // 低=红（薄弱），高=绿（掌握），中间黄
          color: ['#fee2e2', '#fecaca', '#fef3c7', '#d1fae5', '#a7f3d0'],
        },
      },
      series: [
        {
          type: 'heatmap',
          data,
          itemStyle: { borderColor: '#fff', borderWidth: 1 },
          emphasis: { itemStyle: { borderColor: '#2563eb', borderWidth: 2 } },
        },
      ],
    },
    true
  )
}

onMounted(() => {
  chart = echarts.init(el.value)
  render()
  window.addEventListener('resize', resize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  if (chart) chart.dispose()
})

function resize() {
  if (chart) chart.resize()
}

watch(() => [props.perStudent, props.kcNames], render, { deep: true })
</script>

<template>
  <div ref="el" class="heatmap"></div>
</template>

<style scoped>
.heatmap {
  width: 100%;
  height: 420px;
}
</style>
