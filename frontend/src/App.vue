<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, clearAuth, getUser, getToken } from './api/client'
import LoginView from './views/LoginView.vue'
import StudentView from './views/StudentView.vue'
import TeacherView from './views/TeacherView.vue'

const user = ref(getUser())
const meta = ref(null)
const error = ref('')

const loggedIn = computed(() => !!user.value && !!getToken())
const isTeacher = computed(() => user.value?.role === 'teacher')

async function loadMeta() {
  try {
    meta.value = await api.meta()
  } catch (e) {
    error.value = e.message
  }
}

function onLogin(payload) {
  user.value = payload.user
  error.value = ''
  loadMeta()
}

function logout() {
  clearAuth()
  user.value = null
  meta.value = null
}

onMounted(() => {
  if (loggedIn.value) loadMeta()
})
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <span class="logo">知途</span>
        <span class="brand-sub">知识追踪与自适应学习路径系统</span>
      </div>
      <div v-if="loggedIn" class="topbar-right">
        <span class="muted" v-if="meta">
          {{ meta.model }} · {{ meta.device }} · {{ meta.n_kc }} 个知识点
        </span>
        <span class="tag" :class="isTeacher ? 'warn' : ''">
          {{ isTeacher ? '教师端' : '学生端' }}
        </span>
        <span class="uname">{{ user.display_name || user.username }}</span>
        <button class="btn-ghost" @click="logout">退出</button>
      </div>
    </header>

    <main class="content">
      <div v-if="error" class="panel error-panel">
        <strong>出错了：</strong>{{ error }}
      </div>

      <LoginView v-if="!loggedIn" @logged-in="onLogin" />
      <template v-else>
        <StudentView v-if="!isTeacher" :meta="meta" />
        <TeacherView v-else :meta="meta" />
      </template>
    </main>

    <footer class="footer muted">
      知途 · 校级赛演示版本 ·
      <span v-if="meta && meta.data_source === 'synthetic'">
        当前使用合成数据集跑通链路，答辩指标以公开数据集实验报告为准
      </span>
      <span v-else-if="meta"> 数据集：{{ meta.data_source }} </span>
    </footer>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  height: 58px;
  background: #fff;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
  position: sticky;
  top: 0;
  z-index: 10;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.logo {
  font-size: 19px;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--primary);
}

.brand-sub {
  font-size: 12px;
  color: var(--text-dim);
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.uname {
  font-weight: 500;
}

.content {
  flex: 1;
  padding: 20px 22px;
  max-width: 1500px;
  width: 100%;
  margin: 0 auto;
}

.error-panel {
  border-color: var(--bad);
  background: var(--bad-dim);
  color: var(--bad);
  margin-bottom: 16px;
}

.footer {
  padding: 14px 22px;
  text-align: center;
  border-top: 1px solid var(--border);
  background: #fff;
}
</style>
