<script setup>
import { ref } from 'vue'
import { api, setAuth } from '../api/client'

const emit = defineEmits(['logged-in'])

const mode = ref('login')
const username = ref('')
const password = ref('')
const displayName = ref('')
const role = ref('student')
const className = ref('数据结构 2026 级 1 班')
const busy = ref(false)
const error = ref('')

async function submit() {
  error.value = ''
  busy.value = true
  try {
    if (mode.value === 'login') {
      const data = await api.login({ username: username.value.trim(), password: password.value })
      setAuth(data.access_token, data.user)
      emit('logged-in', data)
    } else {
      const data = await api.register({
        username: username.value.trim(),
        password: password.value,
        display_name: displayName.value.trim() || username.value.trim(),
        role: role.value,
        class_name: className.value.trim() || null,
      })
      setAuth(data.access_token, data.user)
      emit('logged-in', data)
    }
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

function useDemo(name) {
  mode.value = 'login'
  username.value = name
  password.value = '123456'
}
</script>

<template>
  <div class="login-wrap">
    <div class="panel login-card">
      <h1 class="title">知途</h1>
      <p class="panel-sub">
        现在的刷题平台只告诉你做了多少题，不告诉你到底哪里不会。<br />
        知途用知识追踪模型把每个知识点的掌握状态算出来，让推荐跟着漏洞走。
      </p>

      <div class="tabs">
        <button :class="['tab', mode === 'login' ? 'active' : '']" @click="mode = 'login'">
          登录
        </button>
        <button :class="['tab', mode === 'register' ? 'active' : '']" @click="mode = 'register'">
          注册
        </button>
      </div>

      <form @submit.prevent="submit">
        <label class="field">
          <span>用户名</span>
          <input v-model="username" placeholder="请输入用户名" autocomplete="username" />
        </label>

        <label class="field" v-if="mode === 'register'">
          <span>显示名称</span>
          <input v-model="displayName" placeholder="选填" />
        </label>

        <label class="field">
          <span>密码</span>
          <input v-model="password" type="password" placeholder="请输入密码" autocomplete="current-password" />
        </label>

        <template v-if="mode === 'register'">
          <label class="field">
            <span>身份</span>
            <select v-model="role">
              <option value="student">学生</option>
              <option value="teacher">教师</option>
            </select>
          </label>
          <label class="field">
            <span>班级</span>
            <input v-model="className" placeholder="班级名称" />
          </label>
        </template>

        <p v-if="error" class="err">{{ error }}</p>

        <button class="btn-primary submit" type="submit" :disabled="busy || !username || !password">
          {{ busy ? '处理中…' : mode === 'login' ? '登录' : '注册并登录' }}
        </button>
      </form>

      <div class="demo">
        <div class="muted">演示账号（密码均为 123456）</div>
        <div class="demo-btns">
          <button class="btn-ghost" @click="useDemo('stu01')">学生 stu01</button>
          <button class="btn-ghost" @click="useDemo('teacher')">教师 teacher</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.login-wrap {
  display: flex;
  justify-content: center;
  padding-top: 48px;
}

.login-card {
  width: 420px;
  padding: 28px 30px;
}

.title {
  margin: 0 0 6px 0;
  font-size: 26px;
  letter-spacing: 4px;
  color: var(--primary);
}

.tabs {
  display: flex;
  gap: 6px;
  margin: 18px 0 16px;
  border-bottom: 1px solid var(--border);
}

.tab {
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  padding: 8px 14px;
  color: var(--text-dim);
}

.tab.active {
  color: var(--primary);
  border-bottom-color: var(--primary);
  font-weight: 600;
}

.field {
  display: block;
  margin-bottom: 12px;
}

.field > span {
  display: block;
  font-size: 12px;
  color: var(--text-dim);
  margin-bottom: 4px;
}

select {
  font-family: inherit;
  font-size: 14px;
  padding: 9px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  width: 100%;
  background: #fff;
  color: var(--text);
}

.submit {
  width: 100%;
  margin-top: 8px;
  padding: 10px;
}

.err {
  color: var(--bad);
  font-size: 13px;
  margin: 4px 0;
}

.demo {
  margin-top: 20px;
  padding-top: 14px;
  border-top: 1px dashed var(--border);
}

.demo-btns {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}
</style>
