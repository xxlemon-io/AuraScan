# 使用 Python 3.12 作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖和 Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 下载 tessdata_best（高精度中文/英文模型）
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata_best && \
    curl -L -o /usr/share/tesseract-ocr/4.00/tessdata_best/chi_sim.traineddata \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata && \
    curl -L -o /usr/share/tesseract-ocr/4.00/tessdata_best/eng.traineddata \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# 暴露 tessdata 路径，允许运行时切换
ENV OCR_TESSDATA_DEFAULT_DIR=/usr/share/tesseract-ocr/4.00/tessdata
ENV OCR_TESSDATA_BEST_DIR=/usr/share/tesseract-ocr/4.00/tessdata_best
ENV OCR_TESSDATA_PROFILE=best

# 验证 Tesseract 安装
RUN tesseract --version && \
    tesseract --list-langs

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
