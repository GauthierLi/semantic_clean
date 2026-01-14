#!/bin/bash

# 交互式图片筛选系统启动脚本
# 作者: Spec
# 版本: 1.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements_interactive.txt"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/app.pid"

# 默认端口配置
DEFAULT_PORT=8023
PORT=${SERVER_PORT:-$DEFAULT_PORT}  # 支持环境变量SERVER_PORT

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python版本
check_python() {
    log_info "检查Python版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "未找到Python命令，请先安装Python 3.8或更高版本"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "找到Python版本: $PYTHON_VERSION"
    
    # 检查版本是否满足要求
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_success "Python版本满足要求"
    else
        log_error "需要Python 3.8或更高版本，当前版本: $PYTHON_VERSION"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log_info "创建Python虚拟环境..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        log_success "虚拟环境创建完成"
    else
        log_info "虚拟环境已存在"
    fi
}

# 激活虚拟环境
activate_venv() {
    log_info "激活虚拟环境..."
    source "$VENV_DIR/bin/activate"
    
    if [ $? -eq 0 ]; then
        log_success "虚拟环境激活成功"
    else
        log_error "虚拟环境激活失败"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    log_info "检查并安装依赖包..."
    
    # 创建requirements文件（如果不存在）
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_info "创建requirements文件..."
        cat > "$REQUIREMENTS_FILE" << EOF
flask==2.3.3
flask-cors==4.0.0
werkzeug==2.3.7
pathlib2==2.3.7
EOF
        log_success "requirements文件创建完成"
    fi
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装依赖
    pip install -r "$REQUIREMENTS_FILE"
    
    if [ $? -eq 0 ]; then
        log_success "依赖包安装完成"
    else
        log_error "依赖包安装失败"
        exit 1
    fi
}

# 创建日志目录
create_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        log_info "日志目录创建完成: $LOG_DIR"
    fi
}

# 检查端口是否被占用
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "端口 $PORT 已被占用"
        
        # 尝试杀掉占用端口的进程
        local pid=$(lsof -ti:$PORT)
        if [ ! -z "$pid" ]; then
            log_info "发现占用端口 $PORT 的进程 PID: $pid"
            read -p "是否要终止该进程? (y/n): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                kill -9 $pid 2>/dev/null || true
                sleep 2
                log_info "进程已终止"
            else
                log_error "请手动处理端口占用问题后重试"
                exit 1
            fi
        fi
    fi
}

# 启动Flask应用
start_app() {
    log_info "启动Flask应用..."
    
    # 设置环境变量
    export FLASK_APP=app.py
    export FLASK_ENV=development
    export FLASK_DEBUG=1
    
    # 切换到应用目录
    cd "$SCRIPT_DIR"
    
    # 启动应用并记录PID
    nohup $PYTHON_CMD app.py > "$LOG_DIR/app.log" 2>&1 &
    local app_pid=$!
    
    # 保存PID
    echo $app_pid > "$PID_FILE"
    
    # 等待应用启动
    sleep 3
    
    # 检查应用是否启动成功
    if kill -0 $app_pid 2>/dev/null; then
        log_success "Flask应用启动成功 (PID: $app_pid)"
        log_info "访问地址: http://localhost:$PORT"
        log_info "日志文件: $LOG_DIR/app.log"
        
        # 测试健康检查接口
        if curl -s http://localhost:$PORT/api/health > /dev/null; then
            log_success "健康检查通过"
        else
            log_warning "健康检查失败，请检查日志"
        fi
    else
        log_error "Flask应用启动失败，请检查日志: $LOG_DIR/app.log"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "交互式图片筛选系统启动脚本"
    echo ""
    echo "用法: $0 [选项] [端口]"
    echo ""
    echo "选项:"
    echo "  start     启动服务 (默认端口: $DEFAULT_PORT)"
    echo "  stop      停止服务"
    echo "  restart   重启服务"
    echo "  status    查看服务状态"
    echo "  logs      查看日志"
    echo "  help      显示帮助信息"
    echo ""
    echo "端口设置:"
    echo "  方法1: $0 start <端口号>     # 命令行指定端口"
    echo "  方法2: SERVER_PORT=<端口号> $0 start  # 环境变量指定端口"
    echo "  方法3: export SERVER_PORT=<端口号>    # 设置环境变量"
    echo ""
    echo "示例:"
    echo "  $0 start          # 启动服务 (默认端口: $DEFAULT_PORT)"
    echo "  $0 start 8080     # 启动服务 (端口: 8080)"
    echo "  $0 stop           # 停止服务"
    echo "  $0 restart        # 重启服务"
    echo "  SERVER_PORT=9000 $0 start  # 使用环境变量指定端口"
}

# 停止服务
stop_service() {
    log_info "停止服务..."
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            sleep 2
            
            # 如果进程仍在运行，强制杀死
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid
            fi
            
            log_success "服务已停止"
        else
            log_warning "服务未运行"
        fi
        
        rm -f "$PID_FILE"
    else
        # 尝试查找并杀死占用端口的进程
        local pid=$(lsof -ti:$PORT 2>/dev/null || true)
        if [ ! -z "$pid" ]; then
            kill -9 $pid 2>/dev/null || true
            log_success "已终止占用端口的进程"
        else
            log_warning "未找到运行中的服务"
        fi
    fi
}

# 重启服务
restart_service() {
    log_info "重启服务..."
    stop_service
    sleep 2
    start_app
}

# 查看服务状态
check_status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 $pid 2>/dev/null; then
            log_success "服务正在运行 (PID: $pid)"
            
            # 显示内存使用情况
            if command -v ps &> /dev/null; then
                local memory=$(ps -p $pid -o rss= 2>/dev/null | tr -d ' ' || echo "N/A")
                if [ "$memory" != "N/A" ]; then
                    local memory_mb=$((memory / 1024))
                    log_info "内存使用: ${memory_mb}MB"
                fi
            fi
            # 测试健康检查
            if curl -s http://localhost:$PORT/api/health > /dev/null; then
                log_success "健康检查通过"
            else
                log_warning "健康检查失败"
            fi
        fi
    else
        local pid=$(lsof -ti:$PORT 2>/dev/null || true)
        if [ ! -z "$pid" ]; then
            log_warning "发现未记录的进程 (PID: $pid)"
        else
            log_info "服务未运行"
        fi
    fi
}

# 查看日志
show_logs() {
    if [ -f "$LOG_DIR/app.log" ]; then
        log_info "显示应用日志 (最后50行):"
        tail -50 "$LOG_DIR/app.log"
    else
        log_warning "日志文件不存在: $LOG_DIR/app.log"
    fi
}

# 主函数
main() {
    local command=${1:-start}
    local port_arg=""
    
    # 检查端口参数
    if [ "$command" = "start" ] && [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
        port_arg="$2"
        export SERVER_PORT="$port_arg"
        log_info "使用指定端口: $port_arg"
    elif [ "$command" = "start" ] && [ -n "$SERVER_PORT" ]; then
        log_info "使用环境变量端口: $SERVER_PORT"
    else
        log_info "使用默认端口: $DEFAULT_PORT"
    fi
    
    # 更新PORT变量
    PORT=${SERVER_PORT:-$DEFAULT_PORT}
    
    case $command in
        start)
            check_python
            create_venv
            activate_venv
            install_dependencies
            create_log_dir
            check_port
            start_app
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 信号处理
trap 'log_warning "接收到中断信号，正在清理..."; stop_service; exit 130' INT TERM

# 运行主函数
main "$@"