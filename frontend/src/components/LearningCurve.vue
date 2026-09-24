<script setup>
/** 掌握度变化曲线（ECharts 折线图）。 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  points: { type: Array, default: () => [] },
  threshold: { type: Number, default: 0.7 },
  title: { type: String, default: '' },
})

const el = ref(null)
let chart = null

function render() {
  if (!chart) return
  const steps = props.points.map((p) => p.step)
  const values = props.points.map((p) => (p.mastery === null ? null : p.mastery))
  chart.setOption(
    {
      grid: { left: 42, right: 16, top: 24, bottom: 28 },
      tooltip: {
        trigger: 'axis',
        formatter: (params) => {
          const i = params[0].dataIndex
          const p = props.points[i]
          return `第 ${p.step} 题<br/>${p.correct ? '答对' : '答错'}<br/>掌握概率 ${
            p.mastery === null ? '未观测' : p.mastery.toFixed(3)
          }`
        },
      },
      xAxis: {
        type: 'category',
        data: steps,
        name: '作答次数',
        nameLocation: 'middle',
        nameGap: 22,
        nameTextStyle: { fontSize: 11, color: '#6b7280' },
        axisLine: { lineStyle: { color: '#d1d5db' } },
        axisLabel: { fontSize: 10, color: '#6b7280' },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        axisLabel: { fontSize: 10, color: '#6b7280' },
        splitLine: { lineStyle: { color: '#f1f5f9' } },
      },
      series: [
        {
          type: 'line',
          data: values,
          smooth: true,
          connectNulls: true,
          symbolSize: 5,
          lineStyle: { width: 2.2, color: '#2563eb' },
          itemStyle: { color: '#2563eb' },
          areaStyle: { color: 'rgba(37,99,235,0.08)' },
          markLine: {
            silent: true,
            symbol: 'none',
            label: { formatter: `阈值 ${props.threshold}`, fontSize: 10, color: '#d97706' },
            lineStyle: { color: '#d97706', type: 'dashed', width: 1 },
            data: [{ yAxis: props.threshold }],
          },
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

watch(() => props.points, render, { deep: true })
</script>

<template>
  <div ref="el" class="curve"></div>
</template>

<style scoped>
.curve {
  width: 100%;
  height: 240px;
}
</style>
