# 交互式图片筛选系统

基于数据清洗后的`edge_cases_result.json`文件，提供交互式的前后端筛选系统，用于人工审核和标记需要review的图像数据。

## 功能特性

### 核心功能
- 📁 **文件加载**: 支持`edge_cases_result.json`文件加载和副本创建
- 🏷️ **类别筛选**: 按类别筛选review样本，支持分页展示
- 🖱️ **交互选择**: 支持点击选择、Ctrl+滑动批量选择
- 🔄 **模式切换**: 正选模式（选中设为accept）和反选模式（选中设为reject）
- 💾 **状态管理**: 先更新类别级别decision，再更新外层decision
- 📥 **结果下载**: 支持修改后JSON文件的下载

### 用户体验
- 🏞️ **瀑布式布局**: 响应式图片网格展示
- 🔍 **图片预览**: 左键点击放大查看，右键快速选择
- ⌨️ **快捷键支持**: 完整的键盘操作支持
- 📊 **性能优化**: 懒加载、内存管理、预加载机制
- 📱 **响应式设计**: 适配不同屏幕尺寸

### 技术特性
- 🚀 **高性能**: 支持大数据集的高效处理
- 🛡️ **安全访问**: 安全的本地文件访问控制
- 📝 **详细日志**: 完整的操作日志和错误处理
- 🔄 **实时同步**: 前后端状态的实时同步

## 快速开始

### 环境要求

- Python 3.8+
- 现代浏览器（Chrome 80+, Firefox 75+, Safari 13+）

### 一键启动

```bash
# 进入项目目录
cd src/interact_visual

# 启动服务
./start_server.sh
```

启动脚本会自动：
1. 检查Python版本
2. 创建虚拟环境
3. 安装依赖包
4. 启动Flask服务器

### 手动启动

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python app.py
```

### 访问系统

启动成功后，访问：http://localhost:5000

## 使用指南

### 基本操作流程

1. **加载文件**
   - 输入`edge_cases_result.json`文件路径
   - 点击"📂 加载文件"按钮

2. **选择类别**
   - 从下拉菜单选择要筛选的类别
   - 系统自动加载该类别的review样本

3. **选择模式**
   - ✅ **正选模式**: 选中的图片设为accept，未选中设为reject
   - ❌ **反选模式**: 选中的图片设为reject，未选中设为accept

4. **选择图片**
   - 🖱️ **右键点击**: 单个图片选择/取消选择
   - 🖱️ **Ctrl+鼠标滑过**: 批量选择图片
   - 🖱️ **左键点击**: 放大查看图片

5. **保存更改**
   - 点击"💾 保存更改"按钮
   - 系统先更新类别级别，再更新外层decision

6. **下载结果**
   - 点击"📥 下载结果"按钮
   - 下载修改后的JSON文件副本

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl + 鼠标滑过 | 批量选择图片 |
| 左键点击 | 放大图片 |
| 右键点击 | 选择/取消选择 |
| 空格键 | 模态框中选择图片 |
| ← → | 切换图片 |
| ESC | 关闭模态框 |
| F | 查看快捷键帮助 |

### 高级功能

#### 性能优化
- **懒加载**: 图片仅在进入视口时加载
- **内存管理**: 自动清理不可见图片的内存
- **预加载**: 自动预加载下一页图片
- **性能监控**: 实时显示内存使用情况

#### 数据处理
- **分页加载**: 每页20张图片，避免内存溢出
- **状态同步**: 前后端状态实时同步
- **错误恢复**: 完善的错误处理和恢复机制

## 项目结构

```
src/interact_visual/
├── app.py                 # Flask后端主应用
├── start_server.sh        # 一键启动脚本
├── requirements.txt        # Python依赖包
├── README.md             # 项目文档
├── templates/
│   └── index.html        # 前端主页面
└── static/
    ├── css/
    │   └── style.css     # 样式文件
    └── js/
        ├── main.js        # 核心JavaScript逻辑
        └── performance.js # 性能优化模块
```

## API接口

### 核心接口

#### 加载数据
```http
POST /api/load_review_data
Content-Type: application/json

{
    "file_path": "/path/to/edge_cases_result.json"
}
```

#### 筛选图片
```http
POST /api/filter_by_category
Content-Type: application/json

{
    "category": "SV_POLICE_CAR",
    "page": 1,
    "per_page": 20
}
```

#### 更新决策
```http
POST /api/update_decisions
Content-Type: application/json

{
    "updates": [
        {
            "image_id": "xxx",
            "category": "SV_POLICE_CAR",
            "decision": "accept"
        }
    ],
    "selection_mode": "positive"
}
```

#### 保存更改
```http
POST /api/save_changes
```

#### 下载文件
```http
GET /api/download_result/<filename>
```

## 配置说明

### 环境变量

```bash
# Flask配置
export FLASK_ENV=development
export FLASK_DEBUG=1

# 服务器配置
export HOST=0.0.0.0
export PORT=5000
```

### 性能配置

在`app.py`中可以调整以下配置：

```python
# 分页大小
PER_PAGE = 20

# 最大文件大小
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# 日志级别
LOG_LEVEL = logging.INFO
```

## 故障排除

### 常见问题

#### 1. 端口被占用
```bash
# 查看占用端口的进程
lsof -i :5000

# 终止进程
kill -9 <PID>

# 或使用脚本
./start_server.sh restart
```

#### 2. 图片无法加载
- 检查图片路径是否正确
- 确认图片文件存在
- 检查文件权限

#### 3. 虚拟环境问题
```bash
# 删除虚拟环境
rm -rf venv

# 重新创建
./start_server.sh start
```

#### 4. 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 日志查看

```bash
# 查看应用日志
./start_server.sh logs

# 或直接查看
tail -f logs/app.log
```

### 服务管理

```bash
# 启动服务
./start_server.sh start

# 停止服务
./start_server.sh stop

# 重启服务
./start_server.sh restart

# 查看状态
./start_server.sh status
```

## 开发指南

### 本地开发

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动开发服务器
python app.py

# 前端文件修改后自动刷新
# 开发模式已启用热重载
```

### 代码结构

- **后端**: Flask应用，提供RESTful API
- **前端**: 纯HTML/CSS/JavaScript，无框架依赖
- **样式**: CSS Grid + Flexbox响应式布局
- **交互**: 原生JavaScript，支持现代浏览器特性

### 扩展开发

1. **添加新的API接口**: 在`app.py`中添加新的路由
2. **扩展前端功能**: 在`main.js`中添加新的方法
3. **样式调整**: 修改`style.css`文件
4. **性能优化**: 在`performance.js`中添加优化逻辑

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 更新日志

### v1.0.0 (2024-01-14)
- ✨ 初始版本发布
- 🎯 完整的图片筛选功能
- 🚀 高性能的瀑布式展示
- 📱 响应式设计
- ⌨️ 完整的快捷键支持
- 🔧 一键启动脚本
- 📊 性能优化和监控
