/**
 * 极简 API 客户端。用原生 fetch，不引入 axios。
 * token 存在 localStorage，所有请求自动带上。
 */

const TOKEN_KEY = 'zhitu_token'
const USER_KEY = 'zhitu_user'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function getUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || 'null')
  } catch {
    return null
  }
}

export function setAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

async function request(path, { method = 'GET', body } = {}) {
  const headers = { 'Content-Type': 'application/json' }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`

  const resp = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  })

  if (resp.status === 401) {
    clearAuth()
    const err = new Error('登录已过期，请重新登录')
    err.status = 401
    throw err
  }

  const text = await resp.text()
  const data = text ? JSON.parse(text) : null
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `请求失败（${resp.status}）`
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    err.status = resp.status
    throw err
  }
  return data
}

export const api = {
  register: (payload) => request('/api/auth/register', { method: 'POST', body: payload }),
  login: (payload) => request('/api/auth/login', { method: 'POST', body: payload }),
  me: () => request('/api/auth/me'),

  meta: () => request('/api/meta'),
  knowledgeGraph: () => request('/api/meta/knowledge-graph'),
  questions: (kcId) => request(`/api/meta/questions${kcId ? `?kc_id=${kcId}` : ''}`),
  models: () => request('/api/meta/models'),

  mastery: () => request('/api/student/mastery'),
  nextQuestion: () => request('/api/student/next-question'),
  answer: (questionId, chosenIndex) =>
    request('/api/student/answer', {
      method: 'POST',
      body: { question_id: questionId, chosen_index: chosenIndex },
    }),
  curve: (kcId) => request(`/api/student/curve${kcId ? `?kc_id=${kcId}` : ''}`),
  history: (limit = 50) => request(`/api/student/history?limit=${limit}`),

  classMastery: () => request('/api/teacher/class-mastery'),
  students: () => request('/api/teacher/students'),
}
