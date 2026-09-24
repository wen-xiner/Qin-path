"""《数据结构》知识点空间建模。

设计文档第五节的要求：把一门课拆成若干知识点，并标注前置依赖。
这里是第一版的人工整理结果，共 26 个知识点。
后续要做「半自动化拆解」（把教材章节交给大模型抽取候选项再人工审核），
只需要替换 build_default_graph() 的数据来源，下游代码不用动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import json

from ml.config import PROCESSED_DIR


@dataclass(frozen=True)
class KnowledgeComponent:
    """一个知识点（知识成分）。"""

    kc_id: str
    name: str
    chapter: str
    prerequisites: tuple[str, ...] = ()
    description: str = ""

    @property
    def depth(self) -> int:
        return 0  # 占位，真实层级由 KnowledgeGraph 计算


# 26 个知识点，按教材章节组织。prerequisites 用的是 kc_id。
_KC_TABLE: list[tuple[str, str, str, tuple[str, ...], str]] = [
    # --- 第 1 章 绪论 ---
    ("complexity", "算法与复杂度分析", "第1章 绪论", (), "大 O 记号、时间与空间复杂度分析"),
    ("array", "数组与内存布局", "第1章 绪论", (), "一维/二维数组、随机访问、寻址公式"),
    # --- 第 2 章 线性表 ---
    ("list_adt", "线性表的逻辑结构", "第2章 线性表", ("array",), "线性表的定义、顺序存储与链式存储"),
    ("singly_linked_list", "单链表", "第2章 线性表", ("list_adt",), "结点、头指针、插入删除、逆置"),
    ("doubly_linked_list", "双向链表与循环链表", "第2章 线性表", ("singly_linked_list",), "双链表与循环链表的操作差异"),
    # --- 第 3 章 栈和队列 ---
    ("stack", "栈", "第3章 栈和队列", ("list_adt",), "后进先出、顺序栈与链栈、应用"),
    ("queue", "队列", "第3章 栈和队列", ("list_adt",), "先进先出、顺序队列与链队列"),
    ("deque_cyclic", "循环队列与双端队列", "第3章 栈和队列", ("queue",), "队满队空判定、双端队列"),
    # --- 第 4 章 串 ---
    ("string_kmp", "串与模式匹配", "第4章 串", ("array",), "朴素匹配、KMP 与 next 数组"),
    # --- 第 5 章 递归 ---
    ("recursion", "递归与分治", "第5章 递归", ("stack",), "递归三要素、递归转非递归、分治思想"),
    # --- 第 6 章 树 ---
    ("tree_basic", "树与二叉树的基本概念", "第6章 树和二叉树", ("recursion",), "树的术语、二叉树性质、存储结构"),
    ("binary_tree_traversal", "二叉树的遍历", "第6章 树和二叉树", ("tree_basic", "stack", "queue"), "先中后序、层次遍历、由遍历序列还原"),
    ("bst", "二叉搜索树", "第6章 树和二叉树", ("binary_tree_traversal",), "查找、插入、删除、有序性"),
    ("avl", "平衡二叉树 AVL", "第6章 树和二叉树", ("bst",), "平衡因子、四种旋转"),
    ("heap", "堆与优先队列", "第6章 树和二叉树", ("tree_basic",), "完全二叉树、上浮下沉、建堆复杂度"),
    ("huffman", "哈夫曼树与哈夫曼编码", "第6章 树和二叉树", ("heap",), "带权路径长度、编码构造"),
    ("union_find", "并查集", "第6章 树和二叉树", ("tree_basic",), "双亲表示、路径压缩、按秩合并"),
    # --- 第 7 章 图 ---
    ("graph_basic", "图的基本概念与存储", "第7章 图", ("array",), "邻接矩阵与邻接表、度数、连通性"),
    ("graph_traversal", "图的遍历", "第7章 图", ("graph_basic", "binary_tree_traversal"), "深度优先与广度优先"),
    ("mst", "最小生成树", "第7章 图", ("graph_traversal", "union_find"), "Prim 与 Kruskal 算法"),
    ("shortest_path", "最短路径", "第7章 图", ("graph_traversal", "heap"), "Dijkstra、Floyd 算法"),
    ("topo_sort", "拓扑排序与关键路径", "第7章 图", ("graph_traversal",), "AOV 网、AOE 网、入度法"),
    # --- 第 8 章 查找 ---
    ("search_basic", "顺序查找与折半查找", "第8章 查找", ("array", "complexity"), "查找长度、折半查找判定树"),
    ("hash", "散列表", "第8章 查找", ("array",), "散列函数、冲突处理、装填因子"),
    # --- 第 9 章 排序 ---
    ("sort_basic", "排序基本概念与简单排序", "第9章 排序", ("array", "complexity"), "稳定性、插入冒泡选择排序"),
    ("sort_advanced", "快速排序与归并排序", "第9章 排序", ("sort_basic", "recursion"), "划分、递归深度、非递归归并"),
    ("sort_heap", "堆排序", "第9章 排序", ("sort_basic", "heap"), "堆的建立与反复调整"),
    # --- 第 10 章 动态规划 ---
    ("dp_basic", "动态规划入门", "第10章 动态规划", ("recursion", "array"), "最优子结构、状态转移、记忆化"),
]


class KnowledgeGraph:
    """知识点依赖图（有向无环图）。"""

    def __init__(self, components: Iterable[KnowledgeComponent]):
        self._map: dict[str, KnowledgeComponent] = {kc.kc_id: kc for kc in components}
        self._validate()

    # ---------------------------------------------------------------- 构建
    @classmethod
    def build_default(cls) -> "KnowledgeGraph":
        comps = [
            KnowledgeComponent(kc_id=k, name=n, chapter=c, prerequisites=tuple(p), description=d)
            for k, n, c, p, d in _KC_TABLE
        ]
        return cls(comps)

    @classmethod
    def from_json(cls, path) -> "KnowledgeGraph":
        raw = json.loads(open(path, encoding="utf-8").read())
        comps = [
            KnowledgeComponent(
                kc_id=item["kc_id"],
                name=item["name"],
                chapter=item.get("chapter", ""),
                prerequisites=tuple(item.get("prerequisites", [])),
                description=item.get("description", ""),
            )
            for item in raw
        ]
        return cls(comps)

    def _validate(self) -> None:
        for kc in self._map.values():
            for p in kc.prerequisites:
                if p not in self._map:
                    raise ValueError(f"知识点 {kc.kc_id} 的前置 {p} 不存在")
        self.topological_order()  # 顺带检查无环

    # ---------------------------------------------------------------- 查询
    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self):
        return iter(self._map.values())

    def __contains__(self, kc_id: str) -> bool:
        return kc_id in self._map

    def __getitem__(self, kc_id: str) -> KnowledgeComponent:
        return self._map[kc_id]

    @property
    def kc_ids(self) -> list[str]:
        return list(self._map.keys())

    def prerequisites_of(self, kc_id: str, transitive: bool = False) -> list[str]:
        """前置知识点。transitive=True 时递归展开全部祖先。"""
        direct = list(self._map[kc_id].prerequisites)
        if not transitive:
            return direct
        seen: list[str] = []
        stack = list(direct)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.append(cur)
            stack.extend(self._map[cur].prerequisites)
        return seen

    def children_of(self, kc_id: str) -> list[str]:
        """以该知识点为前置的知识点。"""
        return [k.kc_id for k in self._map.values() if kc_id in k.prerequisites]

    def topological_order(self) -> list[str]:
        """Kahn 拓扑排序，返回学习顺序（前置在前）。"""
        indeg = {k: len(self._map[k].prerequisites) for k in self._map}
        ready = [k for k, d in indeg.items() if d == 0]
        out: list[str] = []
        while ready:
            cur = ready.pop(0)
            out.append(cur)
            for child in self.children_of(cur):
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
        if len(out) != len(self._map):
            raise ValueError("知识点依赖图存在环，请检查 prerequisities 配置")
        return out

    def learning_layers(self) -> list[list[str]]:
        """把知识点按「第几层能学」分层，第 0 层没有前置。"""
        depth: dict[str, int] = {}
        for kc_id in self.topological_order():
            pre = self._map[kc_id].prerequisites
            depth[kc_id] = 0 if not pre else max(depth[p] for p in pre) + 1
        layers: list[list[str]] = [[] for _ in range(max(depth.values()) + 1)]
        for kc_id, d in depth.items():
            layers[d].append(kc_id)
        return layers

    def depth_of(self, kc_id: str) -> int:
        for i, layer in enumerate(self.learning_layers()):
            if kc_id in layer:
                return i
        raise KeyError(kc_id)

    # ---------------------------------------------------------------- 导出
    def to_json(self, path=None) -> list[dict]:
        payload = [
            {
                "kc_id": kc.kc_id,
                "name": kc.name,
                "chapter": kc.chapter,
                "prerequisites": list(kc.prerequisites),
                "description": kc.description,
            }
            for kc in self
        ]
        if path is not None:
            path = str(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def summary(self) -> dict:
        layers = self.learning_layers()
        return {
            "course": "数据结构",
            "n_kc": len(self),
            "n_edges": sum(len(kc.prerequisites) for kc in self),
            "n_layers": len(layers),
            "layers": [
                {"layer": i, "size": len(layer), "kc_ids": layer} for i, layer in enumerate(layers)
            ],
        }


_default_graph: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    """进程内单例，避免重复构建。"""
    global _default_graph
    if _default_graph is None:
        _default_graph = KnowledgeGraph.build_default()
    return _default_graph


if __name__ == "__main__":
    g = get_knowledge_graph()
    g.to_json(PROCESSED_DIR / "knowledge_graph.json")
    print(json.dumps(g.summary(), ensure_ascii=False, indent=2))
