# EVE Hacking Terminal Game / EVE 黑客终端游戏

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Introduction
This project is a terminal-based faithful recreation of the hacking minigame from the MMORPG **EVE Online**. It simulates the mechanics of exploring a hexagonal grid, managing virus coherence, and battling defensive subsystems to reach the System Core.

### Features
- **Procedural Map Generation**: Every hacking session generates a unique map using a snowflake ID seed.
- **Hexagonal Grid Navigation**: Navigate through nodes using a 6-direction movement system tailored for terminal rendering.
- **Combat System**: Turn-based combat with "Coherence" (Health) and "Strength" (Attack) mechanics.
- **Entities**:
  - **Virus**: Your player character, equipped with utilities and shields.
  - **Defensive Subsystems**: Firewalls, Anti-Virus, Restoration Nodes, and Suppressors.
  - **Utilities**: Pickups like Self-Repair, Kernel Rot, Polymorphic Shield, and Secondary Vector.
- **UI/UX**: Curses-based interface with color coding, status indicators, and distance hints.

### Installation & Requirements
This game is written in Python and runs in the terminal.

**Prerequisites:**
- Python 3.8+
- `windows-curses` (for Windows users)

**Installation:**
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Run the game from the `eve-hack` directory:
```bash
python main.py
```

**Controls:**
- **Mouse**: Click nodes to move/interact. Click utility belt items to use them.
- **Enter**: Restart the game after Game Over.

---

<a name="chinese"></a>
## 中文 (Chinese)

### 简介
本项目是 MMORPG **EVE Online** 中黑客小游戏的终端复刻版。它高度还原了探索六边形网格、管理病毒一致性（Coherence）以及与防御子系统战斗以攻破系统核心（System Core）的游戏机制。

### 特性
- **程序化地图生成**: 使用雪花算法（Snowflake ID）作为种子，每次游戏都会生成独一无二的地图。
- **六边形网格导航**: 专为终端渲染定制的六方向导航系统。
- **战斗系统**: 包含“一致性”（生命值）和“强度”（攻击力）的回合制战斗机制。
- **游戏实体**:
  - **病毒 (Virus)**: 玩家控制的角色，可装备工具和护盾。
  - **防御子系统**: 防火墙 (Firewall)、反病毒软件 (Anti-Virus)、修复节点 (Restoration Node) 和抑制器 (Suppressor)。
  - **工具 (Utilities)**: 可拾取的道具，如自我修复 (Self-Repair)、核心腐烂 (Kernel Rot)、多态护盾 (Polymorphic Shield) 和二级向量 (Secondary Vector)。
- **UI/UX**: 基于 Curses 的界面，包含颜色编码、状态指示器和距离提示。

### 安装与要求
本游戏使用 Python 编写，运行于终端环境。

**前置要求:**
- Python 3.8+
- `windows-curses` (仅 Windows 用户需要)

**安装步骤:**
1. 克隆代码仓库。
2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

### 使用说明
在 `eve-hack` 目录下运行游戏:
```bash
python main.py
```

**操作指南:**
- **鼠标**: 点击节点进行移动或交互。点击工具栏图标使用道具。
- **回车 (Enter)**: 游戏结束（Game Over）后重启游戏。

THANKS: https://github.com/cmbasnett/eve-hack
THANKS: https://wiki.eveuniversity.org/Hacking
THANKS: https://eve.huijiwiki.com/wiki/%E7%A0%B4%E8%AF%91
