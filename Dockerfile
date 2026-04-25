# 使用官方Python运行时作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（包括LightGBM所需的库）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口（使用Railway的动态端口）
EXPOSE $PORT

# 设置环境变量
ENV FLASK_ENV=production

# 启动命令（使用环境变量PORT）
CMD gunicorn --bind 0.0.0.0:$PORT app:app