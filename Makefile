# 知途 · 常用命令（Windows 下可用 Git Bash 执行）
PY ?= python

.PHONY: help setup data exp train backend frontend test build clean

help:
	@echo "make setup     安装后端依赖"
	@echo "make data      构建数据集（自动选档位）"
	@echo "make exp       跑三模型 + 两基线的对比实验"
	@echo "make train     只训练 DKT"
	@echo "make backend   起后端 (8000)"
	@echo "make frontend  起前端 (5173)"
	@echo "make test      跑后端接口测试"
	@echo "make build     构建前端产物"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m ml.data.build_dataset

exp:
	$(PY) -m ml.run_experiments

train:
	$(PY) -m ml.train --model DKT

backend:
	$(PY) -m backend.app.seed
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(PY) -m pytest backend/tests -q

build:
	cd frontend && npm run build

clean:
	rm -rf frontend/dist ml/artifacts/models backend/zhitu.db
