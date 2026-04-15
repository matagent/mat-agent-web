# MatAgent — AI 驱动的材料科学智能助手平台

## 项目详细说明书

> **版本**: 1.0  
> **分支**: `file-update-logic`  
> **生成日期**: 2026-04-15  
> **技术栈**: Python 3.13.4 / LangChain / MCP / FastAPI / Streamlit / VASP

---

## 目录

- [1. 项目概述](#1-项目概述)
  - [1.1 定位与愿景](#11-定位与愿景)
  - [1.2 核心功能矩阵](#12-核心功能矩阵)
  - [1.3 功能特性详解](#13-功能特性详解)
- [2. 系统架构](#2-系统架构)
  - [2.1 四层分离架构](#21-四层分离架构)
  - [2.2 数据流图](#22-数据流图)
  - [2.3 关键架构决策](#23-关键架构决策)
- [3. 技术栈详解](#3-技术栈详解)
  - [3.1 编程语言与运行时](#31-编程语言与运行时)
  - [3.2 LLM 与 Agent 框架](#32-llm-与-agent-框架)
  - [3.3 Web 服务框架](#33-web-服务框架)
  - [3.4 材料科学与 ML 库](#34-材料科学与-ml-库)
  - [3.5 基础设施与工具](#35-基础设施与工具)
- [4. MCP 工具 API 参考（21 个）](#4-mcp-工具-api-参考21-个)
  - [4.1 材料数据库查询（7 个）](#41-材料数据库查询7-个)
  - [4.2 结构建模（1 个）](#42-结构建模1-个)
  - [4.3 VASP 任务管理（10 个）](#43-vasp-任务管理10-个)
  - [4.4 ML 性质预测（2 个）](#44-ml-性质预测2-个)
  - [4.5 基础工具（1 个）](#45-基础工具1-个)
- [5. 功能模块详细说明](#5-功能模块详细说明)
  - [5.1 AI 智能对话模块](#51-ai-智能对话模块)
  - [5.2 材料查询模块](#52-材料查询模块)
  - [5.3 结构建模与可视化模块](#53-结构建模与可视化模块)
  - [5.4 ML 性质预测模块](#54-ml-性质预测模块)
  - [5.5 VASP 远程计算模块](#55-vasp-远程计算模块)
- [6. 数据库 Schema 设计](#6-数据库-schema-设计)
  - [6.1 客户端数据库：matagent.db](#61-客户端数据库matagentdb)
  - [6.2 客户端历史数据库：matagent_history.db](#62-客户端历史数据库matagent_historydb)
  - [6.3 服务端全局数据库：matagent_server_history.db](#63-服务端全局数据库matagent_server_historydb)
- [7. 环境部署指南](#7-环境部署指南)
  - [7.1 环境变量配置](#71-环境变量配置)
  - [7.2 依赖安装](#72-依赖安装)
  - [7.3 启动步骤](#73-启动步骤)
  - [7.4 端口清单](#74-端口清单)
- [8. 目录结构解析](#8-目录结构解析)
- [9. 核心算法说明](#9-核心算法说明)
  - [9.1 XGBoost 带隙预测流程](#91-xgboost-带隙预测流程)
  - [9.2 ALIGNN 多性质预测流程](#92-alignn-多性质预测流程)
  - [9.3 DOS 分析算法](#93-dos-分析算法)
- [10. 常见问题排查](#10-常见问题排查)

---

## 1. 项目概述

### 1.1 定位与愿景

**MatAgent** 是一个 **AI 驱动的材料科学智能助手平台**，基于 LangChain + MCP (Model Context Protocol) + LLM 架构构建。它集成了：

- 材料数据库查询（Materials Project + OQMD）
- 晶体结构建模与可视化
- 机器学习性质预测
- VASP 第一性原理远程计算全流程

用户通过**自然语言对话**即可完成从材料搜索、结构分析到第一性原理计算的全流程操作，大幅降低材料科学研究的技术门槛。

### 1.2 核心功能矩阵

| 功能域 | 能力 | 涉及组件 |
|--------|------|----------|
| **AI 对话** | 多模型支持、SSE 流式响应、工具自动调用 | LangChain Agent, DeepSeek/GLM-5 |
| **材料检索** | MP 数据库稳定查询、OQMD 辅助搜索 | mp-api, OQMD API |
| **结构可视化** | 2D 结构图 + 3D 交互式查看器 | pymatgen, ASE, matplotlib |
| **ML 预测** | XGBoost 快速带隙预测、ALIGNN 16 种性质 | XGBoost, ALIGNN Server |
| **VASP 计算** | SSH 远程任务全生命周期管理 | paramiko, HPC Cluster |
| **结果展示** | 能带/DOS 高质量图表 | matplotlib, plotly |

### 1.3 功能特性详解

#### AI 智能对话
- 支持 **DeepSeek Chat**（默认）、**DeepSeek Reasoner**（推理增强）、**GLM-5**（智谱）三种大模型
- **SSE 流式响应**：实时推送 `token` / `tool_start` / `tool_end` / `complete` 四种事件类型
- **Content Blocks 机制**：精确记录每个工具调用在回复文本中的位置索引，前端实现内联渲染
- **会话持久化**：支持多会话创建/切换/重命名/删除，历史记录可加载恢复

#### 材料数据库查询
- **Materials Project API**（主数据源）：稳定可靠的官方接口，提供材料基础信息、带隙、晶体结构等
- **OQMD 数据库**（辅助数据源）：Open Quantum Materials Database，补充 MP 未覆盖的材料数据
- 支持按化学式、材料 ID、元素组成等多条件组合搜索

#### 晶体结构与可视化
- **2D 结构图**：使用 ASE + Matplotlib 渲染高质量 PNG 图片
- **3D 交互式查看器**：使用 ASE 生成 HTML 格式的交互式 3D 结构，支持鼠标旋转/缩放
- 结构元数据展示：晶格参数（a/b/c/α/β/γ）、空间群、原子位点坐标

#### ML 性质预测
- **XGBoost 本地快速预测**：预训练模型直接推理，毫秒级返回带隙值
- **ALIGNN 远程多性质预测**：通过 HTTP 调用远程 ALIGNN 服务，支持 16 种物理化学性质预测

#### VASP 远程计算管理
- **SSH 安全连接**：使用 paramiko 库建立加密 SSH 隧道
- **四类计算任务**：
  - `relax` — 结构弛豫优化
  - `scf` — 自洽场计算
  - `band` — 能带结构计算
  - `dos` — 态密度计算
- **全生命周期管理**：创建任务 → 生成输入文件 → 修改 INCAR 参数 → 提交队列 → 监控状态 → 提取结果
- **安全机制**：危险命令正则过滤白名单，防止误操作

#### 结果可视化
- **能带结构图**：高分辨率能带色阶图，高对称点路径标注
- **DOS 综合分析图**（2×3 子图布局）：
  - TDOS 总态密度
  - PDOS 分波态密度
  - 积分 DOS 曲线
  - 元素贡献占比
  - 费米能级峰位分析
  - 轨道贡献分解

---

## 2. 系统架构

### 2.1 四层分离架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                           │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Streamlit Web App (端口: 8501)                      │ │
│  │   web_mcp_app.py (~2322 行)                                      │ │
│  │   • 对话界面 / 会话管理 / 结果渲染                                 │ │
│  │   • Content Blocks 内联工具调用展示                                │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
└───────────────────────────────┼──────────────────────────────────────┘
                                │ HTTP REST (POST /chat)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       服务层 (Service Layer)                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │           FastAPI Agent Server (端口: 8766)                      │ │
│  │   agent_mcp_server.py                                            │ │
│  │   • 会话管理 / 历史持久化 / SSE 流式转发                          │ │
│  │   • 工具路由 / LLM 调用 / Content Blocks 记录                    │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
└───────────────────────────────┼──────────────────────────────────────┘
        │                        │                        │
        │ stdio (MCP Client)     │ HTTP (Flask File)      │ Direct Import
        ▼                        ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      工具层 (Tool Layer)                                  │
│  ┌──────────────────────────┐  ┌────────────────────┐  ┌──────────────┐  │
│  │  MCP Tool Server (8000)  │  │  Flask File (6750)  │  │   OQMD 模块  │  │
│  │  mcp_server.py           │  │  flask_server.py    │  │   oqmd.py    │  │
│  │  • 21 个标准化工具函数   │  │  • 图片/HTML 缓存服务│  │  • OQMD 查询 │  │
│  └──────────┬───────────────┘  └────────────────────┘  └──────────────┘  │
└─────────────┼────────────────────────────────────────────────────────────┘
              │ MCP Protocol (tool calls)
              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     外部服务层 (External Services)                        │
│  ┌──────────┐ ┌──────┐ ┌──────────┐ ┌───────┐ ┌──────────┐ ┌─────────┐ │
│  │Materials │ │ OQMD │ │ HPC 集群 │ │ALIGNN │ │  VASP    │ │ SQLite  │ │
│  │Project   │ │ API  │ │(SSH/SFTP)│ │Server │ │ Software │ │ Database│ │
│  └──────────┘ └──────┘ └──────────┘ └───────┘ └──────────┘ └─────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
用户输入自然语言消息
        │
        ▼
┌─────────────────┐
│  Streamlit 前端  │ ◄── 加载历史会话 / 切换会话
│  web_mcp_app.py  │
└────────┬────────┘
         │ POST /chat { message, session_id, model }
         ▼
┌─────────────────────────┐
│  FastAPI Agent Server   │
│  agent_mcp_server.py    │
│  ┌───────────────────┐  │
│  │ 1. 创建/获取会话   │  │
│  │ 2. 构建 Agent Chain│  │
│  │ 3. 调用 LLM 推理   │  │
│  │ 4. 工具调度决策    │  │
│  │ 5. 记录 Content    │  │
│  │    Blocks          │  │
│  │ 6. SSE 流式推送    │  │
│  │ 7. 持久化存储      │  │
│  └────────┬──────────┘  │
└───────────┼─────────────┘
            │ MCP tool_call (stdio transport)
            ▼
┌─────────────────────────┐
│   MCP Tool Server       │
│   mcp_server.py         │
│  ┌───────────────────┐  │
│  │ 动态路由 21 个工具 │  │
│  │ 统一返回格式:     │  │
│  │ {"args":{},       │  │
│  │  "returns":{}}    │  │
│  └────────┬──────────┘  │
└───────────┼─────────────┘
            │
            ├──────────────► Materials Project API (mp-api)
            ├──────────────► OQMD Database REST API
            ├──────────────► HPC Cluster (paramiko SSH/SFTP)
            ├──────────────► ALIGNN Prediction Server (HTTP)
            ├──────────────► XGBoost Local Model
            ├──────────────► pymatgen / ASE (本地计算)
            └──────────────► SQLite Database
```

### 2.3 关键架构决策

| 决策项 | 方案 | 理由 |
|--------|------|------|
| **工具解耦** | MCP 协议 | 工具注册在 MCP Server 中动态发现，Agent 通过 `langchain-mcp-adapters` 自动加载；新增工具只需添加 `@mcp.tool()` 装饰器 |
| **双数据库策略** | 客户端 DB + 服务端 DB 分离 | 客户端 (`matagent.db` / `matagent_history.db`) 存储本地数据；服务端 (`matagent_server_history.db`) 聚合所有客户端请求用于全局审计 |
| **同步/异步桥接** | `MatAgentMCPSync` 包装类 | 使用 `ThreadPoolExecutor` 在新事件循环中运行异步 MCP 操作，使 FastAPI 同步端点可调用异步 Agent |
| **文件服务** | Flask 后台线程 | `MatFileServer` 使用 daemon Thread 启动 Flask，避免阻塞主进程；LRU 策略清理过期缓存（图片 50 个 / HTML 30 个 / 结构 30 个） |
| **SSH 连接管理** | 上下文管理器模式 | 自动管理连接生命周期，确保资源释放 |

---

## 3. 技术栈详解

### 3.1 编程语言与运行时

| 技术 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.13.4 | 主要开发语言，使用最新稳定版特性 |
| **uv** | 推荐 | 现代包管理器，替代 pip/conda，速度更快 |
| **pip** | 兼容 | 传统包管理器备选方案 |

### 3.2 LLM 与 Agent 框架

| 库 | 版本要求 | 用途 |
|----|---------|------|
| **LangChain** | >=0.2.2 | Agent 编排核心框架，负责 LLM 调度与工具链组装 |
| **langchain-mcp-adapters** | 最新 | 官方 MCP 协议适配器，实现 LangChain ↔ MCP 双向通信 |
| **langchain-openai** | 最新 | OpenAI 兼容接口适配，用于 DeepSeek/GLM 模型调用 |
| **FastMCP** | >=2.12.5 | MCP Server 快速构建框架，提供 `@mcp.tool()` 装饰器 |

#### LLM 模型配置

| 模型名称 | 类型 | 用途 | 接口 |
|----------|------|------|------|
| `deepseek-chat` | 聊天模型 | 默认对话模型 | OpenAI Compatible |
| `deepseek-reasoner` | 推理模型 | 复杂推理任务 | OpenAI Compatible |
| `glm-5` | 聊天模型 | 智谱大模型备选 | OpenAI Compatible |

### 3.3 Web 服务框架

| 框架 | 版本 | 端口 | 用途 |
|------|------|------|------|
| **Streamlit** | >=1.28.0 | 8501 | Web 前端 UI 框架 |
| **FastAPI** | >=0.128.1 | 8766 | Agent API 服务（Uvicorn ASGI） |
| **Flask** | 3.1.1 | 6750 | 静态文件/缓存服务 |
| **Uvicorn** | 最新 | — | FastAPI 的 ASGI 服务器 |

### 3.4 材料科学与 ML 库

| 库 | 版本要求 | 用途 |
|----|---------|------|
| **pymatgen** | >=2025.6.14 | 材料科学核心库：晶体结构解析/处理/可视化 |
| **mp-api** | 最新 | Materials Project 官方 Python SDK |
| **ASE** (Atomic Simulation Environment) | 最新 | 原子模拟环境：2D/3D 结构建模与可视化 |
| **XGBoost** | >=1.7.0 | 梯度提升树模型，用于快速带隙预测 |
| **scikit-learn** | 最新 | ML 工具集：特征预处理/模型评估 |
| **matminer** | 最新 | 材料特征工程库 |

### 3.5 基础设施与工具

| 库/工具 | 版本 | 用途 |
|---------|------|------|
| **paramiko** | 3.5.1 | SSH/SFTP 远程连接库 |
| **matplotlib** | ==3.10.3 | 科学绑图（能带/DOS/结构图） |
| **Pillow** | 最新 | 图像处理 |
| **plotly** | 最新 | 交互式图表 |
| **SQLite3** (内置) | — | 轻量级关系型数据库 |
| **python-dotenv** | 最新 | `.env` 环境变量加载 |

---

## 4. MCP 工具 API 参考（21 个）

所有 MCP 工具统一返回格式：

```json
{
  "args": { /* 输入参数 */ },
  "returns": { /* 返回结果 */ }
}
```

### 4.1 材料数据库查询（7 个）

#### 4.1.1 `search_materials`

| 属性 | 说明 |
|------|------|
| **功能** | 在 Materials Project 数据库中搜索材料 |
| **参数** | `query` (str): 搜索关键词或化学式 |
| **返回** | 匹配材料的列表，含 material_id、化学式、带隙等信息 |
| **示例调用** | `"帮我搜索 SiC 的相关材料"` |

#### 4.1.2 `get_material_details`

| 属性 | 说明 |
|------|------|
| **功能** | 获取指定材料的详细信息 |
| **参数** | `material_id` (str): MP 材料标识符（如 `mp-149`） |
| **返回** | 材料的完整属性：晶系、空间群、能量、带隙、形成能等 |
| **依赖** | 需要 MP API Key |

#### 4.1.3 `get_material_band_gap`

| 属性 | 说明 |
|------|------|
| **功能** | 查询指定材料的带隙值 |
| **参数** | `material_id` (str): MP 材料标识符 |
| **返回** | 带隙值 (eV)，区分直接/间接带隙 |

#### 4.1.4 `get_material_structure`

| 属性 | 说明 |
|------|------|
| **功能** | 获取材料的晶体结构数据并生成可视化 |
| **参数** | `material_id` (str): MP 材料标识符 |
| **返回** | CIF 内容、晶格参数、空间群、原子坐标；同时生成 2D PNG 和 3D HTML |
| **副作用** | 写入 `cache/temp_images/` 和 `cache/temp_3d/` 目录 |

#### 4.1.5 `get_material_webpage`

| 属性 | 说明 |
|------|------|
| **功能** | 获取 Materials Project 官方网页链接 |
| **参数** | `material_id` (str): MP 材料标识符 |
| **返回** | MP 官网对应材料的 URL |

#### 4.1.6 `oqmd_search`

| 属性 | 说明 |
|------|------|
| **功能** | 在 OQMD 数据库中搜索材料（辅助数据源） |
| **参数** | `query` (str): 搜索关键词或化学式 |
| **返回** | OQMD 匹配条目列表 |
| **实现位置** | `oqmd.py` 模块，通过 OQMD REST API 查询 |

#### 4.1.7 `oqmd_get_structure`

| 属性 | 说明 |
|------|------|
| **功能** | 从 OQMD 获取材料结构信息 |
| **参数** | `entry_id` (int): OQMD 条目 ID |
| **返回** | OQMD 材料的结构数据 |

---

### 4.2 结构建模（1 个）

#### 4.2.1 `build_structure`

| 属性 | 说明 |
|------|------|
| **功能** | 根据用户描述构建/修改晶体结构 |
| **参数** | `description` (str): 结构描述（如晶格参数、原子位置、空间群等） |
| **返回** | 构建后的 Structure 对象及可视化结果 |
| **能力** | 支持从零构建、修改已有结构、导入自定义 CIF |

---

### 4.3 VASP 任务管理（10 个）

> 所有 VASP 工具均通过 **SSH 连接远程 HPC 集群**执行，使用 `server/tryssh.py` 中的 `SSHConnection` 上下文管理器。

#### 4.3.1 `create_task`

| 属性 | 说明 |
|------|------|
| **功能** | 创建新的 VASP 计算任务目录 |
| **参数** | `task_name` (str), `calculation_type` (str: relax/scf/band/dos), `structure_info` (str/dict) |
| **返回** | 任务目录路径和初始配置 |
| **工作流位置** | 任务生命周期起点 |

#### 4.3.2 `list_dirs`

| 属性 | 说明 |
|------|------|
| **功能** | 列出远程 HPC 上的目录内容 |
| **参数** | `path` (str): 远程路径（默认为工作根目录） |
| **返回** | 目录和文件列表 |
| **用途** | 浏览 HPC 文件系统、检查任务目录 |

#### 4.3.3 `squeue`

| 属性 | 说明 |
|------|------|
| **功能** | 查看集群作业队列状态 |
| **参数** | 无（或可选的过滤条件） |
| **返回** | 当前用户的 SLURM 作业队列信息（JOBID/状态/时间等） |
| **底层命令** | `squeue -u $USER` |

#### 4.3.4 `create_mission`

| 属性 | 说明 |
|------|------|
| **功能** | 为已创建的任务生成 VASP 输入文件（POSCAR/POTCAR/KPOINTS/INCAR） |
| **参数** | `task_path` (str), `params` (dict: INCAR 参数等) |
| **返回** | 生成的输入文件清单 |
| **前置依赖** | 必须先执行 `create_task` |

#### 4.3.5 `submit_mission`

| 属性 | 说明 |
|------|------|
| **功能** | 将准备好的 VASP 任务提交到 SLURM 队列 |
| **参数** | `task_path` (str), `script_content` (str: 可选自定义脚本) |
| **返回** | SLURM JOBID |
| **底层命令** | `sbatch submit.sh` |
| **前置依赖** | 必须先执行 `create_mission` |

#### 4.3.6 `modify_incar`

| 属性 | 说明 |
|------|------|
| **功能** | 修改已有任务的 INCAR 参数 |
| **参数** | `task_path` (str), `modifications` (dict: {参数名: 新值}) |
| **返回** | 更新后的 INCAR 内容 |
| **典型场景** | 调整 ENCUT、KPOINTS、ISMEAR 等关键参数 |
| **安全限制** | 内置参数白名单校验 |

#### 4.3.7 `extract_result`

| 属性 | 说明 |
|------|------|
| **功能** | 从完成的任务中提取计算结果（OUTCAR/OSZICAR/CONTCAR 等） |
| **参数** | `task_path` (str), `result_type` (str: energy/forces/converged/structure) |
| **返回** | 解析后的结果数据 |
| **典型输出** | 总能量、原子受力、收敛标志、优化后结构 |

#### 4.3.8 `execute_command`

| 属性 | 说明 |
|------|------|
| **功能** | 在远程 HPC 上执行任意 Shell 命令 |
| **参数** | `command` (str): 要执行的命令 |
| **返回** | stdout/stderr/exit_code |
| **⚠️ 安全警告** | 有命令白名单过滤，危险操作会被拦截 |

#### 4.3.9 `extract_file`

| 属性 | 说明 |
|------|------|
| **功能** | 从远程 HPC 下载指定文件到本地 |
| **参数** | `remote_path` (str): 远程文件路径, `local_path` (str): 本地保存路径 |
| **返回** | 文件下载确认 |
| **传输协议** | SFTP |

#### 4.3.10 `read_file`

| 属性 | 说明 |
|------|------|
| **功能** | 读取远程 HPC 上文件的内容 |
| **参数** | `file_path` (str): 远程文件路径, `lines` (int): 读取行数（可选） |
| **返回** | 文件内容文本 |
| **用途** | 快速查看 OUTCAR/INCAR/SLURM 输出等 |

---

### 4.4 ML 性质预测（2 个）

#### 4.4.1 `predict_band_gap`

| 属性 | 说明 |
|------|------|
| **功能** | 使用 XGBoost 预训练模型快速预测材料带隙 |
| **参数** | `formula` (str): 材料化学式（如 `SiC`, `GaAs`, `Perovskite`） |
| **返回** | 预测的带隙值 (eV) + 特征向量 + 置信度评估 |
| **性能** | 本地推理，毫秒级响应 |
| **模型来源** | `myml/xgb_model.json` — 预训练 XGBoost 模型 |
| **特征工程** | `myml/featurizer.py` — 基于 `element_features.csv` 的 145 维化学式特征 |

**预测流程：**

```
输入化学式 "SiC"
    │
    ▼
featurizer.py 解析化学式
    │
    ├── 提取元素: Si, C
    ├── 查表 element_features.csv (每元素 ~72 维特征)
    └── 组合为 145 维特征向量 (统计量+比值+差值)
    │
    ▼
xgb_model.json (XGBoost 预训练模型) 推理
    │
    ▼
输出: band_gap = 2.36 eV (预估)
```

#### 4.4.2 `predict_with_alignn`

| 属性 | 说明 |
|------|------|
| **功能** | 使用 ALIGNN 模型预测多种材料性质 |
| **参数** | `structure_input` (str): CIF 文件路径或 material_id |
| **返回** | 16 种性质的预测值（见下表） |
| **性能** | 远程服务调用，秒级~十秒级响应 |
| **通信方式** | HTTP POST 到 ALIGNN Server |

**ALIGNN 可预测的 16 种性质：**

| 类别 | 性质 | 单位 |
|------|------|------|
| 电子性质 | 带隙 (band_gap) | eV |
| 电子性质 | 形成能 (formation_energy_per_atom) | eV/atom |
| 力学性质 | 体积模量 (bulk_modulus_kv) | GPa |
| 力学性质 | 剪切模量 (shear_modulus_gv) | GPa |
| 热学性质 | 德拜温度 (T_debye) | K |
| 热学性质 | 热容 (C_v) | J/(mol·K) |
| 超导性质 | 临界温度 (Tc_supercon) | K |
| 光学性质 | 折射率 (refractive_index) | — |
| ... | （共 16 种） | — |

---

### 4.5 基础工具（1 个）

#### 4.5.1 `get_time`

| 属性 | 说明 |
|------|------|
| **功能** | 获取当前服务器时间 |
| **参数** | 无 |
| **返回** | ISO 8601 格式的当前时间戳 |
| **用途** | 连通性测试、调试辅助 |

---

## 5. 功能模块详细说明

### 5.1 AI 智能对话模块

#### 5.1.1 模块组成

```
agent/
└── langchain_mcp_agent.py    # ~623 行，Agent 核心逻辑

agent_mcp_server.py           # FastAPI Agent Server 入口
web_mcp_app.py                # Streamlit 前端对话界面
```

#### 5.1.2 核心类与函数

| 类/函数 | 位置 | 职责 |
|---------|------|------|
| `MatAgentMCPSync` | `agent/langchain_mcp_agent.py` | 同步包装 MCP 异步操作的桥接类 |
| `create_mat_agent_with_tools()` | `agent/langchain_mcp_agent.py` | 工厂函数，创建完整的 LangChain Agent Chain |
| `chat()` | `agent_mcp_server.py` | FastAPI 端点，处理聊天请求 |
| `stream_chat()` | `agent_mcp_server.py` | SSE 流式端点，实时推送响应 |

#### 5.1.3 SSE 事件流协议

```
Client ──POST /chat/stream──▶ Server
                              │
                              ▼
                    ┌─────────────────┐
                    │  Event Stream   │
                    ├─────────────────┤
                    │ event: token    │ ← LLM 文本片段
                    │ data: {"..."}   │
                    │                 │
                    │ event: tool_start│ ← 工具调用开始
                    │ data: {...}     │   含 tool_name, args
                    │                 │
                    │ event: tool_end  │ ← 工具调用结束
                    │ data: {...}     │   含 result
                    │                 │
                    │ event: complete  │ ← 整体完成
                    │ data: {...}     │   含 content_blocks, model, duration
                    └─────────────────┘
                              │
                              ▼
Client ◀──text/event-stream── Server
```

#### 5.1.4 Content Blocks 机制

Content Blocks 是 MatAgent 的核心创新之一，它精确记录每个工具调用在最终回复中的**位置索引**：

```json
{
  "content_blocks": [
    {
      "type": "tool_result",
      "tool_name": "get_material_structure",
      "start_index": 45,
      "end_index": 180,
      "render_mode": "card",       // card / image / table / code / chart
      "data": { /* 工具返回数据 */ },
      "metadata": {
        "material_id": "mp-149",
        "image_url": "/files/images/xxx.png",
        "html_url": "/files/3d/xxx.html"
      }
    }
  ]
}
```

**前端渲染策略：**
- `text` — 直接显示工具返回文本
- `image` — 渲染图片（2D 结构图、能带图、DOS 图）
- `html` — 嵌入 iframe（3D 结构查看器）
- `table` — 表格形式展示（搜索结果列表）
- `code` — 代码块高亮显示（INCAR、POSCAR 内容）
- `chart` — Plotly 交互式图表

#### 5.1.5 Agent Chain 构建流程

```python
# 伪代码（来自 agent/langchain_mcp_agent.py）

async def create_mat_agent_with_tools():
    # 1. 创建 MCP Client (stdio transport)
    mcp_client = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    # 2. 将 MCP 工具转换为 LangChain Tools
    tools = await load_tools(mcp_client)
    
    # 3. 配置 LLM（支持 deepseek-chat/reasoner/glm-5）
    llm = ChatOpenAI(
        model=model_name,
        base_url=api_base,
        streaming=True
    )
    
    # 4. 构建带有记忆的 Agent Chain
    chain = (
        {"chat_history": ... , "input": ...}
        | ChatPromptTemplate.from_messages([...])
        | llm.bind_tools(tools)
        | output_parser
    )
    
    return chain
```

---

### 5.2 材料查询模块

#### 5.2.1 数据源对比

| 特性 | Materials Project (MP) | OQMD |
|------|------------------------|------|
| **数据规模** | >150,000 条材料 | >600,000 条 DFT 计算条目 |
| **数据质量** | 高（经过严格筛选） | 较高（自动化高通量） |
| **API 稳定性** | ⭐⭐⭐⭐⭐ 官方 SDK | ⭐⭐⭐ 第三方 REST |
| **主要用途** | 主数据源 | 补充/交叉验证 |
| **认证方式** | API Key | 无需认证（公开） |
| **查询方式** | `mp-api` Python SDK | 自定义 HTTP 请求 (`oqmd.py`) |

#### 5.2.2 MP 查询示例

```python
from mp_api import MPRester

with MPRester(MP_API_KEY) as mpr:
    # 搜索 SiC 相关材料
    results = mpr.materials.search(formula="SiC")
    
    # 获取详情
    material = mpr.materials.get(material_id="mp-149")
    
    # 获取带隙
    bandgap = material.band_gap  # eV
    
    # 获取结构
    structure = material.structure  # pymatgen Structure 对象
```

#### 5.2.3 OQMD 查询实现 (`oqmd.py`)

OQMD 模块通过其公开 REST API 进行查询：

- **基础 URL**: `https://oqmd.org/oqmd/api/`
- **搜索接口**: `/search/?formula={query}` 或 `/search/?element={element}`
- **结构接口**: `/entry/{entry_id}/`
- **返回格式**: JSON

---

### 5.3 结构建模与可视化模块

#### 5.3.1 可视化管线

```
pymatgen Structure 对象
        │
        ├──▶ 2D 可视化管线
        │     │
        │     ▼
        │   ase.plot.plot()  ──▶ matplotlib Figure
        │                             │
        │                             ▼
        │                   cache/temp_images/{uuid}.png
        │                             │
        │                             ▼
        │                  Flask File Server (6750)
        │                   /files/images/{uuid}.png
        │
        └──▶ 3D 可视化管线
              │
              ▼
            ase.visualize.view()  ──▶ HTML (NGL Viewer)
                                        │
                                        ▼
                              cache/temp_3d/{uuid}.html
                                        │
                                        ▼
                                   Flask File Server (6750)
                                    /files/3d/{uuid}.html
```

#### 5.3.2 结构元数据

每次生成结构可视化时，同时缓存结构元数据到 `cache/structure_info.json`：

```json
{
  "mp-149": {
    "formula": "SiC",
    "lattice": {
      "a": 4.348, "b": 4.348, "c": 4.348,
      "alpha": 90.0, "beta": 90.0, "gamma": 90.0
    },
    "space_group": "F-43m (216)",
    "num_sites": 2,
    "sites": [
      {"species": "Si", "coordinates": [0, 0, 0]},
      {"species": "C",  "coordinates": [0.25, 0.25, 0.25]}
    ],
    "image_2d": "/files/images/abc123.png",
    "image_3d": "/files/3d/def456.html"
  }
}
```

#### 5.3.3 Flask 文件服务 (`flask_server.py`)

```python
class MatFileServer:
    """后台线程运行的 Flask 文件服务器"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.cache_images = {}    # LRU: max 50 images
        self.cache_html = {}      # LRU: max 30 HTML files
        self.cache_structure = {} # LRU: max 30 structures
    
    def start(self):
        """以 daemon Thread 启动，不阻塞主进程"""
        t = Thread(target=self.app.run, kwargs={
            'host': '0.0.0.0',
            'port': 6750,
            'debug': False
        }, daemon=True)
        t.start()
    
    @self.app.route('/files/images/<filename>')
    def serve_image(filename):
        """返回 2D 结构图 PNG"""
        
    @self.app.route('/files/3d/<filename>')
    def serve_3d(filename):
        """返回 3D 结构 HTML"""
```

**路由清单：**

| HTTP 方法 | 路径 | 说明 |
|-----------|------|------|
| GET | `/files/images/<name>` | 2D 结构图 PNG |
| GET | `/files/3d/<name>` | 3D 交互式结构 HTML |
| GET | `/files/structures/<name>` | 结构元数据 JSON |
| POST | `/upload/cif` | 上传自定义 CIF 文件 |
| GET | `/custom/<name>` | 用户自定义结构的图片 |

---

### 5.4 ML 性质预测模块

#### 5.4.1 XGBoost 带隙预测 (`myml/bandgap_predict.py`)

**模型规格：**
- 算法: XGBoost (eXtreme Gradient Boosting)
- 训练数据: 来自 Materials Project / OQMD 的高通量 DFT 计算结果
- 模型文件: `myml/xgb_model.json`
- 特征维度: 145 维
- 预测目标: 带隙 (Band Gap, 单位: eV)

**完整预测代码流程：**

```python
# myml/bandgap_predict.py

import xgboost as xgb
import json
from .featurizer import FormulaFeaturizer

class BandGapPredictor:
    def __init__(self):
        # 加载预训练模型
        self.model = xgb.Booster()
        self.model.load_model('myml/xgb_model.json')
        
        # 初始化特征工程器（加载 element_features.csv）
        self.featurizer = FormulaFeaturizer('myml/element_features.csv')
    
    def predict(self, formula: str) -> dict:
        """
        输入: "SiC" 或 "GaAs" 或 "ABO3"
        输出: {"band_gap": 2.36, "features": [...], "confidence": 0.87}
        """
        # Step 1: 化学式 → 特征向量
        features = self.featurizer.featurize(formula)
        
        # Step 2: XGBoost 推理
        dmatrix = xgb.DMatrix([features])
        prediction = self.model.predict(dmatrix)[0]
        
        return {
            'band_gap': float(prediction),
            'features': features.tolist(),
            'confidence': self._estimate_confidence(features)
        }
```

#### 5.4.2 特征工程 (`myml/featurizer.py`)

**特征提取流程：**

```
输入化学式: "SrTiO3" (钙钛矿)
    │
    ▼
Step 1: 化学式解析
    ├── 元素列表: [Sr, Ti, O]
    ├── 元素计量比: {Sr: 1, Ti: 1, O: 3}
    └── 总原子数: 5
    │
    ▼
Step 2: 查表元素特征 (element_features.csv)
    ┌──────────┬───────────┬──────────┬────────┐
    │  Element  │ atomic_num│ electroneg│ radius │ ...
    ├──────────┼───────────┼──────────┼────────┤
    │ Sr (38)  │ 38.00     │ 0.95     │ 1.18   │ ...(~72维)
    │ Ti (22)  │ 22.00     │ 1.54     │ 0.605  │ ...(~72维)
    │ O  (8)   │ 8.00      │ 3.44     │ 0.73   │ ...(~72维)
    └──────────┴───────────┴──────────┴────────┘
    │
    ▼
Step 3: 组合特征工程 (145 维输出)
    ├── 全局统计特征 (~30 维):
    │   ├── 平均原子序数、平均电负性、平均原子半径
    │   ├── 最大/最小/方差/标准差
    │   ├── 价电子总数/平均价电子数
    │   └── 分子量
    ├── 计量加权特征 (~40 维):
    │   ├── 各属性的加权平均值 (按化学计量比加权)
    │   └── 加权方差
    ├── 元素对特征 (~50 维):
    │   ├── 电负性差值 (Δχ) — Pauling 规则
    │   ├── 原子半径比 (rA/rB)
    │   ├── 电荷转移趋势
    │   └── Goldschmidt 容忍因子
    └── 结构指示特征 (~25 维):
        ├── 氧化态组合熵
        ├── 配位数估计
        ├── s/p/d/f 电子占比
        └── 过渡金属标记
    │
    ▼
输出: numpy.array([2.35, 0.95, 1.18, ..., 0.42])  # shape: (145,)
```

**`element_features.csv` 数据结构：**

| 列名 | 示例值 | 说明 |
|------|--------|------|
| `element` | Si, C, O, Fe... | 元素符号 |
| `atomic_number` | 14, 6, 8, 26... | 原子序数 |
| `atomic_mass` | 28.085, 12.011, 15.999... | 原子质量 |
| `electronegativity_pauling` | 1.90, 2.55, 3.44... | Pauling 电负性 |
| `atomic_radius_pm` | 111, 77, 66... | 原子半径 (pm) |
| `valence_electrons` | 4, 4, 6... | 价电子数 |
| `common_oxidation_states` | [+4], [-4,+2,+4]... | 常见氧化态 |
| `group` | 14, 14, 16... | 族数 |
| `period` | 3, 2, 2... | 周期数 |
| `block` | p, p, p... | 区块 (s/p/d/f) |
| `ionization_energy_kJ` | 786.5, 1086.5, 1313.9... | 第一电离能 (kJ/mol) |
| `electron_affinity_kJ` | 133.6, 121.7, -141.0... | 电子亲和能 (kJ/mol) |
| `melting_point_K` | 1687, 3823, 54.36... | 熔点 (K) |
| `boiling_point_K` | 3538, 4098, 90.20... | 沸点 (K) |
| `density_g_cm3` | 2.33, 3.51, 0.00143... | 密度 (g/cm³) |
| `...` | ... | 共约 72 列特征 |

#### 5.4.3 ALIGNN 远程预测

ALIGNN (Atomistic Line Graph Neural Network) 是一种基于图神经网络的材料性质预测模型。

**调用流程：**

```python
# mcp_server.py 中的 predict_with_alignn 工具

@mcp.tool()
async def predict_with_alignn(structure_input: str) -> dict:
    """
    调用远程 ALIGNN Server 进行多性质预测
    """
    # 1. 准备输入（CIF 内容或 material_id）
    input_data = _prepare_alignn_input(structure_input)
    
    # 2. 发送 HTTP POST 到 ALIGNN Server
    response = requests.post(
        f"{ALIGNN_SERVER_URL}/predict",
        json=input_data,
        timeout=60  # ALIGNN 推理可能较慢
    )
    
    # 3. 解析 16 种性质预测结果
    predictions = response.json()
    
    return {
        "properties": predictions,  # 16 种性质
        "model_version": predictions.get("version"),
        "inference_time_ms": predictions.get("time")
    }
```

---

### 5.5 VASP 远程计算模块

#### 5.5.1 模块组成

```
server/
└── tryssh.py    # ~1473 行，SSH/VASP 核心
```

#### 5.5.2 SSH 连接管理

```python
# server/tryssh.py

import paramiko
from contextlib import contextmanager

@contextmanager
def SSHConnection(hostname, username, password=None, key_filename=None):
    """
    SSH 连接上下文管理器
    - 进入时建立连接
    - 退出时自动关闭
    - 支持 with 语法保证资源释放
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    client.connect(
        hostname=hostname,
        username=username,
        password=password,
        key_filename=key_filename
    )
    
    try:
        yield client
    finally:
        client.close()


# 使用示例
with SSHConnection(HOST, USER, KEY) as ssh:
    stdin, stdout, stderr = ssh.exec_command("ls -la")
    print(stdout.read().decode())
```

#### 5.5.3 VASP 任务完整生命周期

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ create_  │───▶│ create_      │───▶│ modify_      │───▶│ submit_     │
│ task     │    │ mission      │    │ incar        │    │ mission     │
│          │    │              │    │              │    │             │
│ 创建任务 │    │ 生成输入文件  │    │ 调整INCAR    │    │ 提交SLURM   │
│ 目录     │    │ POSCAR       │    │ 参数         │    │ sbatch      │
│          │    │ POTCAR       │    │              │    │             │
│          │    │ KPOINTS      │    │              │    │             │
│          │    │ INCAR        │    │              │    │             │
└──────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘
                                                            │
                                                     ┌──────▼────────┐
                                                     │   squeue      │◄──── 循环监控
                                                     │   查询状态     │
                                                     └──────┬────────┘
                                                            │
                                               ┌────────────▼────────┐
                                               │  extract_result     │
                                               │  提取计算结果       │
                                               │  • 能量/受力/收敛   │
                                               │  • CONTCAR/OUTCAR  │
                                               │  • 能带/DOS 数据    │
                                               └─────────────────────┘
```

#### 5.5.4 四种计算类型的 INCAR 参数模板

| 参数 | relax | scf | band | dos | 说明 |
|------|-------|-----|------|-----|------|
| `ISTART` | 0 | 0 | 1 | 1 | 波函数起始 |
| `ENCUT` | 520 | 520 | 520 | 520 | 截断能 (eV) |
| `ISMEAR` | 0 | 0 | -5 | -5 | smearing 方法 |
| `SIGMA` | 0.05 | 0.05 | 0.01 | 0.01 | smearing 宽度 |
| `EDIFF` | 1E-5 | 1E-6 | 1E-6 | 1E-6 | 电子收敛标准 |
| `EDIFFG` | -0.02 | — | — | — | 离子步收敛 (eV/Å) |
| `NSW` | 100 | 0 | 0 | 0 | 离子步数 |
| `ISIF` | 3 | 0 | 0 | 0 | 应力计算 |
| `LCHARG` | F | T | F | F | 写电荷密度 |
| `LWAVE` | F | T | T | F | 写波函数 |
| `ICHARG` | 2 | 2 | 11 | 11 | 电荷读入方式 |
| `NBANDS` | auto | auto | auto* | auto* | 能带数 |
| `NEDOS` | — | — | — | 3001 | DOS 点数 |
| `LORBIT` | — | — | — | 11 | PDOS 详细程度 |

> *band/dos 的 NBANDS 通常由 scf 计算确定

#### 5.5.5 安全机制

```python
# server/tryssh.py 中的命令白名单

ALLOWED_COMMANDS = [
    r'^ls\s',           # 列出目录
    r'^cat\s',          # 查看文件
    r'^head\s',         # 查看前几行
    r'^tail\s',         # 查看后几行
    r'^grep\s',         # 搜索内容
    r'^pwd$',           # 显示当前目录
    r'^squeue',         # 查看队列
    r'^sacct',          # 查看账目
    r'^sbatch\s.*sh$',  # 提交脚本
    r'^scancel\s+\d+$', # 取消作业
]

def execute_command(command: str) -> dict:
    for pattern in ALLOWED_COMMANDS:
        if re.match(pattern, command.strip()):
            # 白名单内的命令，允许执行
            return safe_exec(command)
    
    raise SecurityError(f"Command not allowed: {command}")
```

---

## 6. 数据库 Schema 设计

系统使用 **3 个独立的 SQLite 数据库**，职责分离明确：

### 6.1 客户端数据库：`matagent.db`

**用途**: 存储客户端本地的材料和用户偏好数据

| 表名 | 说明 | 关键字段 |
|------|------|----------|
| `materials` | 用户收藏/查询过的材料 | id, material_id, formula, band_gap, structure_cache, created_at |
| `user_preferences` | 用户设置 | key, value (KV 存储) |

**特点**: 仅 Streamlit 前端直接读写，不与服务端共享

---

### 6.2 客户端历史数据库：`matagent_history.db`

**用途**: 客户端的会话管理和聊天记录

#### sessions 表

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK | 会话自增 ID |
| `session_uuid` | TEXT UNIQUE | 全局唯一会话标识 (UUID) |
| `title` | TEXT | 会话标题（自动提取或用户命名） |
| `model` | TEXT | 使用的 LLM 模型 (deepseek-chat/glm-5...) |
| `created_at` | DATETIME | 创建时间 |
| `updated_at` | DATETIME | 最后更新时间 |
| `is_deleted` | INTEGER DEFAULT 0 | 软删除标记 |

#### chats 表

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK | 消息自增 ID |
| `session_id` | INTEGER FK | 关联 sessions.id |
| `role` | TEXT | 角色 (user/assistant/system/tool) |
| `content` | TEXT | 消息内容 (JSON 格式，含文本+工具调用) |
| `timestamp` | DATETIME | 消息时间戳 |
| `token_count` | INTEGER | Token 数量估算 |

#### tool_calls 表

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK | 调用记录 ID |
| `chat_id` | INTEGER FK | 关联 chats.id |
| `tool_name` | TEXT | 工具名称 (如 get_material_structure) |
| `args_json` | TEXT | 输入参数 JSON |
| `result_json` | TEXT | 返回结果 JSON |
| `duration_ms` | INTEGER | 执行耗时 |
| `status` | TEXT | success/error/timeout |
| `timestamp` | DATETIME | 调用时间 |

---

### 6.3 服务端全局数据库：`matagent_server_history.db`

**用途**: 服务端聚合所有客户端请求的全局审计日志

| 表名 | 说明 | 额外字段（相比客户端） |
|------|------|----------------------|
| `client_sessions` | 所有客户端的会话 | client_ip, user_agent, client_id |
| `global_chats` | 全局聊天记录 | session_uuid, model, duration_ms |
| `tool_call_logs` | 全局工具调用日志 | **content_blocks** (JSON), **error_traceback**, **server_timestamp` |

**`content_blocks` 字段结构：**

```json
[
  {
    "type": "tool_result",
    "tool_name": "get_material_structure",
    "start_index": 45,
    "end_index": 180,
    "render_mode": "card",
    "metadata": { "material_id": "mp-149" }
  }
]
```

**为什么需要三套数据库？**

1. **`matagent.db`** — 纯客户端数据，离线可用，不含敏感对话
2. **`matagent_history.db`** — 客户端对话历史，支持多设备同步
3. **`matagent_server_history.db`** — 服务端审计日志，用于：
   - 全局用量统计和分析
   - 工具调用成功率监控
   - Content Blocks 精确追踪
   - 问题诊断和调试

---

## 7. 环境部署指南

### 7.1 环境变量配置

创建 `.env` 文件（参考 `config/.env.example`）：

```bash
# ================================
# LLM 配置
# ================================

# DeepSeek API (推荐)
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 智谱 GLM-5 (可选)
ZHIPU_API_KEY=xxxxxxxx.xxxxxxxx.xxxxxxxx
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 默认模型选择: deepseek-chat | deepseek-reasoner | glm-5
DEFAULT_MODEL=deepseek-chat

# ================================
# Materials Project API
# ================================
MP_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

# ================================
# HPC / SSH 配置 (VASP 计算)
# ================================
HPC_HOST=hpc.cluster.edu.cn
HPC_USERNAME=your_username
HPC_SSH_KEY_PATH=/home/user/.ssh/id_rsa
HPC_WORK_DIR=/home/your_username/matagent_work/

# ================================
# ALIGNN 预测服务
# ================================
ALIGNN_SERVER_URL=http://your-alignn-server:5000

# ================================
# 服务器端口配置
# ================================
STREAMLIT_PORT=8501
AGENT_API_PORT=8766
MCP_SERVER_PORT=8000
FILE_SERVER_PORT=6750
```

### 7.2 依赖安装

#### 方式 A：使用 uv（推荐）

```bash
# 克隆项目
git clone <repo-url> mat-agent-web
cd mat-agent-web

# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

#### 方式 B：使用 pip

```bash
pip install -r config/requirements.txt
```

#### Node.js 依赖（用于 docx/pptx 导出功能）

```bash
npm install
```

### 7.3 启动步骤

```bash
# Step 0: 加载环境变量
export $(cat .env | xargs)

# Step 1: 启动 MCP 工具服务器 (端口 8000)
python mcp_server.py &
echo "MCP Server started on port 8000"

# Step 2: 启动 FastAPI Agent Server (端口 8766)
# 该服务内部会同时启动 Flask 文件服务 (端口 6750)
python agent_mcp_server.py &
echo "Agent Server started on port 8766"
echo "File Server started on port 6750"

# Step 3: 启动 Streamlit 前端 (端口 8501)
streamlit run web_mcp_app.py --server.port 8501 &
echo "Web Frontend started on port 8501"

# Step 4: 打开浏览器访问
# http://localhost:8501
```

#### 一键启动脚本（建议创建 `start.sh`）

```bash
#!/bin/bash
set -e

export $(cat .env | xargs)

echo "=== Starting MatAgent Platform ==="

# 启动 MCP Server
python mcp_server.py &
MCP_PID=$!
echo "[OK] MCP Server (PID: $MCP_PID) on :8000"
sleep 2

# 启动 Agent Server (包含 Flask File Server)
python agent_mcp_server.py &
AGENT_PID=$!
echo "[OK] Agent Server (PID: $AGENT_PID) on :8766"
echo "[OK] File Server on :6750"
sleep 2

# 启动 Streamlit Frontend
streamlit run web_mcp_app.py --server.port 8501 &
FRONTEND_PID=$!
echo "[OK] Frontend (PID: $FRONTEND_PID) on :8501"

echo ""
echo "=== All Services Started ==="
echo "Frontend: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services..."

# 等待中断信号
trap "kill $MCP_PID $AGENT_PID $FRONTEND_PID; exit" SIGINT SIGTERM
wait
```

### 7.4 端口清单

| 端口 | 服务 | 协议 | 说明 |
|------|------|------|------|
| **8501** | Streamlit Web Frontend | HTTP | 用户访问入口 |
| **8766** | FastAPI Agent Server | HTTP/REST | Agent API + SSE 流 |
| **8000** | MCP Tool Server | stdio (本地) | 工具函数服务 |
| **6750** | Flask File Server | HTTP | 图片/HTML 缓存服务 |

**防火墙配置（生产环境）：**

```bash
# 仅暴露前端端口给外部
iptables -A INPUT -p tcp --dport 8501 -j ACCEPT
iptables -A INPUT -p tcp --dport 8766 -s 127.0.0.1 -j ACCEPT  # 仅本地
iptables -A INPUT -p tcp --dport 6750 -s 127.0.0.1 -j ACCEPT  # 仅本地
# 8000 端口是 stdio 不监听网络
```

---

## 8. 目录结构解析

```
mat-agent-web/                          # 项目根目录
│
├── agent_mcp_server.py                 # [核心] FastAPI Agent Server (8766)
│                                       #   - /chat 端点: 同步聊天
│                                       #   - /chat/stream 端点: SSE 流式
│                                       #   - 会话 CRUD 管理
│                                       #   - 历史记录查询
│                                       #   - 内嵌启动 MatFileServer (6750)
│
├── mcp_server.py                       # [核心] MCP Tool Server (8000)
│                                       #   - 21 个 @mcp.tool() 装饰的工具函数
│                                       #   - Material Project API 封装
│                                       #   - OQDB 查询封装
│                                       #   - XGBoost 预测集成
│                                       #   - ALIGNN 远程预测调用
│                                       #   - VASP/SSH 操作代理
│                                       #   - 约 2046 行代码
│
├── flask_server.py                     # [核心] Flask 文件服务 (6750)
│                                       #   - MatFileServer 类
│                                       #   - 图片/HTML/JSON 缓存服务
│                                       #   - LRU 清理策略
│                                       #   - 后台 Daemon Thread 运行
│
├── web_mcp_app.py                      # [核心] Streamlit Web 前端 (8501)
│                                       #   - 对话界面 (st.chat_message)
│                                       #   - 侧边栏 (会话管理/设置)
│                                       #   - Content Blocks 渲染引擎
│                                       #   - 图片/3D/表格/代码展示
│                                       #   - 能带/DOS 图表绑定
│                                       #   - 约 2322 行代码
│
├── oqmd.py                             # [模块] OQMD 数据库查询
│                                       #   - OQDM REST API 封装
│                                       #   - search / get_structure 函数
│
├── agent/                              # [包] Agent 核心逻辑
│   └── langchain_mcp_agent.py          #   - MatAgentMCPSync 类
│                                       #   - create_mat_agent_with_tools()
│                                       #   - LangChain Chain 构建
│                                       #   - Prompt Template 定义
│                                       #   - 约 623 行代码
│
├── server/                             # [包] 远程服务管理
│   └── tryssh.py                       #   - SSHConnection 上下文管理器
│                                       #   - VASP 任务全生命周期
│                                       #   - SLURM 队列操作
│                                       #   - 命令安全白名单
│                                       #   - SFTP 文件传输
│                                       #   - 约 1473 行代码
│
├── myml/                               # [包] 机器学习预测模块
│   ├── bandgap_predict.py              #   - BandGapPredictor 类
│                                       #   - XGBoost 模型加载与推理
│                                       #   - 约 251 行
│   ├── featurizer.py                   #   - FormulaFeaturizer 类
│                                       #   - 145 维化学式特征工程
│                                       #   - element_features.csv 解析
│                                       #   - 约 364 行
│   ├── element_features.csv            #   - 元素特征数据表 (~118 元素 × 72 特征)
│   ├── element_features_bandgap.csv    #   - 带隙专用特征数据
│   └── xgb_model.json                  #   - 预训练 XGBoost 模型权重
│
├── db/                                 # [包] 数据库管理
│   └── databasemanage.py               #   - DatabaseManager 类
│                                       #   - 3 个 SQLite DB 的 CRUD
│                                       #   - 会话/聊天/工具调用持久化
│                                       #   - 约 427 行
│
├── config/                             # [目录] 配置文件
│   ├── loadenv.py                      #   - EnvConfig 类 (.env 加载与校验)
│   ├── .env.example                    #   - 环境变量模板 (复制为 .env 使用)
│   ├── pyproject.toml                  #   - uv 项目配置 (133 个依赖声明)
│   ├── requirements.txt                #   - pip 依赖锁定文件
│   ├── uv.toml                         #   - uv 镜像源配置 (国内加速)
│   └── config.toml                     #   - Streamlit 主题配置 (颜色/字体)
│
├── cache/                              # [运行时] 缓存目录
│   ├── temp_images/                    #   - 2D 结构图 PNG (LRU max 50)
│   ├── temp_3d/                        #   - 3D 结构 HTML (LRU max 30)
│   └── structure_info.json             #   - 结构元数据缓存
│
├── calculation_output/                 # [运行时] VASP 计算结果输出
│                                       #   - 从 HPC 下载的结果文件
│                                       #   - OUTCAR, CONTCAR, vasprun.xml...
│
├── custom_structures/                  # [运行时] 用户自定义结构
│                                       #   - 上传的 CIF 文件
│                                       #   - 对应的结构图片
│
├── cifs/                               # [运行时] CIF 文件存储
│                                       #   - 从 MP/OQMD 下载的 CIF
│                                       #   - 用于 VASP 输入转换
│
├── web/                                # [静态] Web 资源
│   └── assets/
│       └── logo.png                    #   - MAT Agent Logo
│
├── package.json                        # Node.js 依赖 (docx/pptx/pdf 导出)
│
├── matagent_server_history.db          # [生成] 服务端全局数据库
│
├── README.md                           # 项目说明文档
│
└── PROJECT_MANUAL.md                   # ← 本文档
```

---

## 9. 核心算法说明

### 9.1 XGBoost 带隙预测流程

#### 9.1.1 算法概述

XGBoost (eXtreme Gradient Boosting) 是一种基于梯度提升决策树的集成学习算法，在本项目中用于**仅需化学式即可快速预测材料带隙**，无需任何 DFT 计算。

#### 9.1.2 完整推理管线

```
┌──────────────────────────────────────────────────────────────┐
│                    输入: 化学式字符串                          │
│                     例: "CsSnI3" (钙钛矿)                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: 化学式解析 (FormulaParser)                          │
│                                                              │
│  输入: "CsSnI3"                                              │
│  输出:                                                       │
│    elements     = ["Cs", "Sn", "I"]                          │
│    counts       = [1, 1, 3]                                  │
│    total_atoms  = 5                                          │
│    fractions    = [0.2, 0.2, 0.6]  (计量比)                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: 元素特征查表 (Element Feature Lookup)               │
│                                                              │
│  对每个元素，从 element_features.csv 查询 72 维特征向量:      │
│                                                              │
│  Cs (铯, Z=55):                                             │
│    [55.00, 132.91, 0.79, 3.03, 267, 1, 6, s, 375.7, 45.5, │
│     301.6, 28.4, 1.93, 9.73, ... ]  (72 维)                 │
│                                                              │
│  Sn (锡, Z=50):                                             │
│    [50.00, 118.71, 1.96, 1.39, 145, 4, 14, p, 708.6, 107.3│
│     141.1, 141.1, 7.31, 5.77, ... ]  (72 维)                 │
│                                                              │
│  I  (碘, Z=53):                                             │
│    [53.00, 126.90, 2.66, 1.40, 140, 7, 17, p, 1008.4, 295.2│
│     -295.1, 4.94, 4.63, ... ]  (72 维)                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: 组合特征工程 (Compositional Feature Engineering)   │
│                                                              │
│  3.1 全局统计特征 (~30 维):                                  │
│  ├── mean(Z), mean(χ), mean(r), ...                         │
│  ├── std(Z), range(χ), var(r), ...                          │
│  ├── n_elements, total_atoms, valence_electron_total         │
│  └── molecular_weight                                       │
│                                                              │
│  3.2 计量加权特征 (~40 维):                                  │
│  ├── Σ(fi * Zi), Σ(fi * χi), Σ(fi * ri), ...                │
│  ├── weighted_variance of each property                      │
│  └── entropy_of_composition = -Σ(fi * log(fi))               │
│                                                              │
│  3.3 元素对差异特征 (~50 维):                                │
│  ├── Δχ_max (最大电负性差值)                                 │
│  ├── Δχ_mean (平均电负性差值)                                 │
│  ├── r_min/r_max (半径比)                                    │
│  ├── ΔZ (原子序数差)                                         │
│  └── δ (电负性差异化度量)                                    │
│                                                              │
│  3.4 物理化学指示特征 (~25 维):                              │
│  ├── Goldschmidt tolerance factor t = (rA + rO)/[√2(rB+rO)]│
│  ├── octahedral factor μ = rB/rO                             │
│  ├── valence_orbital_ratio (s/p/d/f electron ratio)          │
│  ├── transition_metal_fraction                               │
│  └── oxidation_state_combination_entropy                     │
│                                                              │
│  最终输出: 145 维浮点数向量                                   │
│  [2.35, 0.95, 1.18, 0.67, 3.04, 0.20, ... ]               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 4: XGBoost 模型推理                                   │
│                                                              │
│  模型: myml/xgb_model.json                                   │
│  结构: 145 维输入 → 200 棵决策树集成 → 1 维输出              │
│                                                              │
│  inference:                                                  │
│    DMatrix(input_vector)                                     │
│    → booster.predict(dmatrix)                                │
│    → band_gap = 1.31 eV                                      │
│                                                              │
│  模型超参数 (训练时设定):                                     │
│    n_estimators  = 200                                       │
│    max_depth     = 6                                         │
│    learning_rate = 0.05                                      │
│    objective     = reg:squarederror                          │
│    eval_metric   = rmse                                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 5: 后处理与置信度估计                                  │
│                                                              │
│  输出字典:                                                    │
│  {                                                           │
│    "band_gap": 1.31,          # 预测带隙 (eV)                │
│    "formula": "CsSnI3",       # 输入化学式                    │
│    "model": "xgboost_v1.0",  # 模型版本                      │
│    "features": [...],        # 145维特征 (可复用)             │
│    "confidence": 0.82,       # 置信度分数                     │
│    "prediction_type": "semiconductor",  # 分类               │
│    "warning": null            # 异常警告                      │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

#### 9.1.3 置信度估计算法

置信度基于**叶子节点样本覆盖度和预测方差**：

```python
def estimate_confidence(xgb_model, feature_vector):
    """基于 XGBoost 叶子覆盖度的置信度估计"""
    
    # 1. 获取每棵树的预测叶节点
    leaf_indices = xgb_model.predict(
        xgb.DMatrix([feature_vector]), 
        pred_leaf=True
    )  # shape: (1, 200)
    
    # 2. 计算每个叶节点的训练样本数
    coverages = []
    for tree_idx, leaf_idx in enumerate(leaf_indices[0]):
        leaf_count = xgb_model.get_booster().trees()[tree_idx][leaf_idx].cover
        coverages.append(leaf_count)
    
    # 3. 归一化为置信度 (0~1)
    avg_coverage = np.mean(coverages)
    confidence = min(avg_coverage / MAX_LEAF_COVERAGE, 1.0)
    
    # 4. 结合预测值的梯度信息微调
    prediction = xgb_model.predict(xgb.DMatrix([feature_vector]))[0]
    
    return confidence
```

---

### 9.2 ALIGNN 多性质预测流程

#### 9.2.1 算法概述

ALIGNN (Atomistic Line Graph Neural Network) 是一种专门针对晶体材料设计的图神经网络架构。与传统 GNN 不同，ALIGNN 同时建模**原子间键**（原子图）和**键间角度关系**（线图），能够更精确地捕捉晶体局部几何结构。

#### 9.2.2 ALIGNN 架构示意

```
输入: Crystal Structure (CIF / POSCAR)
         │
         ▼
┌─────────────────────┐
│  图构建 (Graph       │
│  Construction)      │
│                     │
│  Atom Graph (G):     │
│  Nodes = Atoms       │
│  Edges = Bonds       │
│  (cutoff = 8Å)      │
│                     │
│  Line Graph (L):     │
│  Nodes = Edges(G)    │
│  Edges = Bond angles │
│  (共享原子的键对)    │
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│  ALIGNN Message Passing Layers (×4 层)            │
│                                                   │
│   ┌─────────────────────────────────────────┐     │
│   │  ALIGNN Layer t:                          │     │
│   │                                           │     │
│   │  1. Atom → Bond 消息传递                  │     │
│   │     h_{ij}^{(t)} = φ_v(h_i^{(t)}, h_j^{(t)},  │
│   │                        e_{ij}^{(t)})     │     │
│   │                                           │     │
│   │  2. Bond → Angle 消息传递 (线图)          │     │
│   │     u_{ijk}^{(t)} = φ_u(h_{ij}^{(t)},     │     │
│   │                         h_{jk}^{(t)},     │     │
│   │                         cos θ_{ijk})      │     │
│   │                                           │     │
│   │  3. Angle → Bond 反向更新                 │     │
│   │     e_{ij}^{(t+1)} = ψ_e(e_{ij}^{(t)},    │     │
│   │                      Σ u_{ijk})           │     │
│   │                                           │     │
│   │  4. Bond → Atom 反向更新                 │     │
│   │     h_i^{(t+1)} = ψ_h(h_i^{(t)},         │     │
│   │                     Σ h_{ij})             │     │
│   └─────────────────────────────────────────┘     │
│                                                   │
└─────────┬──────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Readout & Predict   │
│                     │
│  Global Pooling:    │
│  h_crystal = Σ h_i  │
│                     │
│  Task-specific MLPs:│
│  y_bandgap = MLP_bg(h_crystal)     → 2.36 eV   │
│  y_formation = MLP_fe(h_crystal)   → -2.1 eV   │
│  y_bulk_mod = MLP_bm(h_crystal)    → 180 GPa   │
│  ... (共 16 个 MLP heads)          │
└─────────────────────┘
```

#### 9.2.3 调用流程

```python
# MatAgent → ALIGNN Server 的调用链

1. 用户输入: "预测 mp-149 的所有性质"
        │
        ▼
2. Agent 决策: 调用 get_material_structure("mp-149") 获取 CIF
        │
        ▼
3. Agent 决策: 调用 predict_with_alignn(cif_content)
        │
        ▼
4. MCP Server 执行:
   ├── POST {ALIGNN_URL}/predict
   ├── Body: {"cif": "...", "props": ["all"]}
   └── Timeout: 60s (ALIGNN 推理较慢)
        │
        ▼
5. ALIGNN Server 处理:
   ├── 解析 CIF → pymatgen Structure
   ├── 构建原子图 + 线图 (cutoff=8Å)
   ├── 4 层 ALIGNN Message Passing
   ├── 16 个 MLP Head 并行预测
   └── 返回 JSON 结果
        │
        ▼
6. 返回 Agent → 前端展示:
   {
     "properties": {
       "band_gap": 1.12,
       "formation_energy_per_atom": -0.45,
       "bulk_modulus_kv": 85.3,
       // ... 共 16 种
     },
     "inference_time_ms": 3240,
     "model_version": "alignn-jdft2d-v2.0"
   }
```

---

### 9.3 DOS 分析算法

#### 9.3.1 数据来源

DOS 数据来自 VASP `doscar_to_dict()` 解析（pymatgen 内置）：

```python
from pymatgen.io.vasp import Vasprun

vasprun = Vasprun("vasprun.xml")
dos_data = vasprun.complete_dos
```

#### 9.3.2 2×3 综合分析图布局

```
┌──────────────────────────────────────────────────────────┐
│                    DOS 综合分析面板                        │
├──────────────────────┬───────────────────────────────────┤
│                      │                                   │
│   [1] TDOS 总态密度   │   [4] 元素贡献占比 (饼图/堆叠)    │
│                      │                                   │
│   Energy (eV)        │   Si: 45%                         │
│   ↑  DOS             │   C:  55%                         │
│   │  ╱╲              │                                   │
│   │ ╱  ╲_______      │                                   │
│   │╱ Ef      ╲___    │                                   │
│   └───────────→      │                                   │
│   -8   -4   0   4   8│                                   │
│                      │                                   │
├──────────────────────┼───────────────────────────────────┤
│                      │                                   │
│   [2] PDOS 分波态密度 │   [5] 费米能级峰位分析             │
│                      │                                   │
│   DOS ↑              │   Peak Analysis:                  │
│   │  s (---)         │   • Peak 1: -4.2 eV (σ bond)      │
│   │  p (╱╲)          │   • Peak 2: -1.8 eV (π bond)      │
│   │  d (~~)          │   • Peak 3: +2.5 eV (anti-bond)   │
│   │                  │                                   │
│   └────→ E           │   Ef = 0.0 eV                     │
│                      │   N(Ef) = 3.2 states/eV           │
├──────────────────────┼───────────────────────────────────┤
│                      │                                   │
│   [3] 积分 DOS       │   [6] 轨道贡献分解 (热力图)         │
│                      │                                   │
│   electrons ↑        │   Orbital × Energy heatmap:       │
│   6 ┤    ╭──╮        │                                   │
│   4 ┤  ╭─╯  ╰─╮       │        s    p    d               │
│   2 ┤╭─╯      ╰      │   Si   ██   ████  ░░             │
│   0 ┼──────────       │   C    ███  █████ █              │
│     -8  0   8         │                                   │
└──────────────────────┴───────────────────────────────────┘
```

#### 9.3.3 关键分析指标

| 指标 | 公式/定义 | 物理意义 |
|------|-----------|----------|
| **N(E<sub>f</sub>)** | 费米面处 DOS 值 | 金属性强度指标；>0 表示金属性 |
| **积分电子数** | ∫<sub>-∞</sub><sup>E</sup> N(E)dE | 占据态累计电子数 |
| **带隙 (间接)** | VBM-CBM 最小间距 | 间接跃迁所需最小能量 |
| **带隙 (直接)** | 相同 k 点 VBM-CBM 间距 | 直接跃迁能量 |
| **有效质量** | m\* = ħ² (∂²E/∂k²)<sup>-1</sup> | 载流子迁移率指标 |
| **轨道杂化程度** | 各轨道在 E<sub>f</sub> 附近贡献比例 | 成键特征判断 |

---

## 10. 常见问题排查

### 10.1 启动问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ModuleNotFoundError: No module named 'mp_api'` | 依赖未安装 | `uv sync` 或 `pip install mp-api` |
| `Connection refused on port 8766` | Agent Server 未启动 | 先启动 `agent_mcp_server.py` |
| `Streamlit 无法连接后端` | CORS 或端口错误 | 确保 8766/8000 端口可达 |
| `.env 文件未找到` | 环境变量未配置 | `cp config/.env.example .env` 并填写 |
| `paramiko.AuthenticationException` | SSH 密钥/密码错误 | 检查 `HPC_SSH_KEY_PATH` 或 `HPC_PASSWORD` |
| `KeyError: MP_API_KEY` | MP API Key 未设置 | 在 [materialsproject.org](https://materialsproject.org/heatmap) 申请 API Key |

### 10.2 MCP 连接问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `MCP client connection timeout` | MCP Server (8000) 未启动或崩溃 | 检查 `mcp_server.py` 日志 |
| `Tool not found: xxx` | 工具名拼写错误或未注册 | 查看 `mcp_server.py` 中 `@mcp.tool()` 装饰器列表 |
| `stdio transport error` | MCP 进程意外退出 | 重启 MCP Server；检查依赖是否齐全 |

### 10.3 VASP/HPC 问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `SSH connection timed out` | HPC 集群不可达或网络问题 | ping 测试；VPN 是否连接 |
| `Permission denied (publickey)` | SSH 密钥未授权到 HPC | `ssh-copy-id user@hpc_host` |
| `sbatch: command not found` | SLURM 未安装在 HPC 默认 PATH | 检查 HPC 的 module 系统: `module load slurm` |
| `OUTCAR not found` | 任务尚未完成或目录错误 | 先用 `squeue` 确认任务已完成 |
| `POTCAR not found` | VASP_PSPATH 未设置 | 在 HPC 上设置: `export VASP_PSPATH=/path/to/potcars` |

### 10.4 ML 预测问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `xgb_model.json not found` | 模型文件缺失 | 确认 `myml/xgb_model.json` 存在 |
| `element_features.csv parse error` | CSV 文件损坏 | 重新下载或检查文件编码 (UTF-8) |
| `Unknown element: Xxx` | 输入了无效元素符号 | 检查化学式拼写 (区分大小写) |
| `ALIGNN connection timeout` | ALIGNN Server 不可达 | 检查 `ALIGNN_SERVER_URL` 和网络连通性 |

### 10.5 前端显示问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 3D 结构不显示 (空白) | NGL Viewer JS 加载失败 | 检查网络（需 CDN 访问）；考虑本地托管 |
| 图片显示 404 | Flask File Server 未启动或缓存过期 | 确认 6750 端口正常；重新触发结构查询 |
| SSE 流中断 | 浏览器/网络问题 | 刷新页面；检查代理/防火墙设置 |
| Content Block 不渲染 | `start_index`/`end_index` 错误 | 检查 Agent Server 返回的 content_blocks 格式 |

### 10.6 性能优化建议

| 场景 | 建议 |
|------|------|
| **首次 MP 查询慢** | MP API 有速率限制；后续查询有缓存 |
| **大量结构可视化** | LRU 自动清理旧缓存（图片 50 / HTML 30） |
| **XGBoost 批量预测** | 当前逐个调用；可扩展为批量推理接口 |
| **ALIGNN 响应慢** | 远程推理正常耗时 3-10s；可在前端加 loading 动画 |
| **HPC 文件传输慢** | 大 OUTCAR (>100MB) 压缩后再 SFTP 下载 |
| **数据库膨胀** | 定期清理 `matagent_server_history.db` 旧记录 |

### 10.7 日志与调试

**日志位置（建议）：**

```bash
# MCP Server 日志
tail -f logs/mcp_server.log

# Agent Server 日志
tail -f logs/agent_server.log

# Flask 文件服务日志
tail -f logs/file_server.log

# Streamlit 前端日志 (终端输出)
# 直接查看启动 Streamlit 的终端
```

**开启 Debug 模式：**

```bash
# Agent Server Debug
python agent_mcp_server.py --debug

# MCP Server Debug (FastMCP verbose)
MCP_DEBUG=1 python mcp_server.py

# Streamlit Debug
streamlit run web_mcp_app.py --server.port 8501 --logger.level=debug
```

**常用数据库查询（调试用）：**

```bash
# 查看最近的工具调用
sqlite3 matagent_server_history.db "
SELECT tool_name, status, datetime(timestamp,'localtime') 
FROM tool_call_logs 
ORDER BY timestamp DESC LIMIT 20;
"

# 查看各工具调用的平均耗时
sqlite3 matagent_server_history.db "
SELECT tool_name, 
       COUNT(*) as call_count,
       AVG(duration_ms) as avg_ms
FROM tool_call_logs 
GROUP BY tool_name 
ORDER BY call_count DESC;
"
```

---

## 附录 A: 文件大小与代码量统计

| 文件 | 行数 (约) | 职责复杂度 |
|------|-----------|-----------|
| `mcp_server.py` | ~2046 | ★★★★★ (21 个工具 + 外部 API) |
| `web_mcp_app.py` | ~2322 | ★★★★★ (完整前端 UI) |
| `server/tryssh.py` | ~1473 | ★★★★☆ (SSH/VASP 全流程) |
| `agent/langchain_mcp_agent.py` | ~623 | ★★★★☆ (Agent 核心逻辑) |
| `db/databasemanage.py` | ~427 | ★★★☆☆ (CRUD 操作) |
| `myml/featurizer.py` | ~364 | ★★★☆☆ (特征工程) |
| `myml/bandgap_predict.py` | ~251 | ★★★☆☆ (XGBoost 预测) |
| `flask_server.py` | ~200 | ★★☆☆☆ (文件服务) |
| `oqmd.py` | ~150 | ★★☆☆☆ (OQDM 封装) |
| **总计** | **~7856 行** | — |

---

## 附录 B: 依赖清单 (部分核心)

```
# config/pyproject.toml 中的主要依赖 (133 个):

# --- LLM & Agent ---
langchain>=0.2.2
langchain-mcp-adapters
langchain-openai
fastmcp>=2.12.5

# --- Web Framework ---
fastapi>=0.128.1
uvicorn[standard]
streamlit>=1.28.0
flask==3.1.1

# --- Materials Science ---
pymatgen>=2025.6.14
mp-api
ase

# --- Machine Learning ---
xgboost>=1.7.0
scikit-learn
matminer
numpy
pandas

# --- Remote Connection ---
paramiko==3.5.1

# --- Visualization ---
matplotlib==3.10.3
pillow
plotly

# --- Utilities ---
python-dotenv
httpx
requests
pydantic
```

---

## 附录 C: 快速参考卡

### 常用 API 端点

```
POST /chat           → 同步聊天 (非流式)
POST /chat/stream    → SSE 流式聊天
GET  /sessions       → 获取会话列表
POST /sessions       → 创建新会话
PUT  /sessions/{id}  → 更新会话 (重命名)
DELETE /sessions/{id} → 删除会话
GET  /history/{id}   → 获取会话历史
```

### MCP 工具速查

```
材料查询:  search_materials / get_material_details / get_material_band_gap /
           get_material_structure / get_material_webpage /
           oqmd_search / oqmd_get_structure

结构:      build_structure

VASP:      create_task / list_dirs / squeue / create_mission /
           submit_mission / modify_incar / extract_result /
           execute_command / extract_file / read_file

预测:      predict_band_gap / predict_with_alignn

其他:      get_time
```

### VASP 任务类型

```
relax → 结构优化 (NSW=100, ISIF=3)
scf   → 自洽场   (NSW=0,  ISIF=0, LCHARG=T, LWAVE=T)
band  → 能带结构 (ICHARG=11, 从 SCF 波函数出发)
dos   → 态密度   (ICHARG=11, NEDOS=3001, LORBIT=11)
```

---

> **文档结束**  
> 如有问题或建议，请提交 Issue 或联系维护者。  
> 最后更新: 2026-04-15
