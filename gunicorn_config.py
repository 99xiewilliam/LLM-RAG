# Gunicorn配置文件
import multiprocessing
import os

# 服务器绑定设置
bind = "0.0.0.0:50052"  # 监听地址和端口

# Worker进程数量（由于使用GPU资源和异步操作，建议只用一个worker）
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"  # 使用uvicorn作为worker

# 进程名称
proc_name = "rag_api"

# 最大请求数和超时设置
max_requests = 1000
max_requests_jitter = 50
timeout = 300  # 5分钟超时
graceful_timeout = 120
keepalive = 5

# 日志设置
errorlog = "logs/gunicorn-error.log"
accesslog = "logs/gunicorn-access.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 守护进程设置
daemon = False  # 生产环境可以设为True

# 预加载
preload_app = True

# Worker管理
# 如果worker挂掉，会自动重启
max_worker_restart = 5

# 进程文件
pidfile = "rag_api.pid"
