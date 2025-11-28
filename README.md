# AuraScan

基于 PaddleOCR 的现代化 OCR 文字识别应用，提供简洁美观的 Web 界面。

## 特性

- 🚀 基于 PaddleOCR Hub Serving 模式
- 🎨 现代化 UI 设计（玻璃拟态风格）
- 🔒 安全的内部网络架构（仅前端端口暴露）
- 📱 响应式设计，支持桌面和移动端
- 🖼️ 支持拖拽、粘贴、点击上传图片

## 快速开始

### 前置要求

- Docker
- Docker Compose

### 启动服务

```bash
docker-compose up -d
```

### 访问应用

打开浏览器访问：`http://localhost:8080`

### 停止服务

```bash
docker-compose down
```

## 架构说明

```
宿主机:8080 (仅此端口暴露)
    ↓
Nginx (webapp)
    ├─ 静态文件: / → index.html
    └─ API 代理: /api/ → paddleocr-api:8000 (内部网络)
```

- **前端服务** (`webapp`): 暴露 `8080` 端口，提供 Web 界面
- **API 服务** (`paddleocr-api`): 仅内部访问，通过 Nginx 反向代理

## API 使用

### 端点

- **URL**: `/api/predict/ocr_system`
- **方法**: `POST`
- **Content-Type**: `multipart/form-data`

### 请求示例

```bash
curl -X POST http://localhost:8080/api/predict/ocr_system \
  -F "images=@test_image.jpg"
```

### 响应格式

```json
{
  "msg": "Success",
  "results": [
    [
      {
        "confidence": 0.99,
        "text": "识别到的文本",
        "text_region": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
      }
    ]
  ]
}
```

## 配置说明

### 修改 OCR 模型

编辑 `compose.yml` 中的 `command` 参数：

```yaml
command: hub serving start -m ch_pp-ocrv3 -p 8000
```

可用模型：
- `ch_pp-ocrv3`: 中文 OCR 模型 v3（默认）
- `ch_pp-ocrv2`: 中文 OCR 模型 v2
- `en_pp-ocrv3`: 英文 OCR 模型 v3

### GPU 支持

如需使用 GPU，在 `paddleocr-api` 服务中添加：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

并在 `command` 中添加 `--use_gpu true`。

## 常见问题

### 服务启动失败

```bash
# 检查端口占用
netstat -ano | findstr :8080

# 查看日志
docker-compose logs paddleocr-api
docker-compose logs webapp
```

### API 请求失败

- 确认服务正常运行：`docker-compose ps`
- 检查 Nginx 配置：`docker-compose exec webapp cat /etc/nginx/conf.d/default.conf`
- 查看日志：`docker-compose logs webapp`

### 识别结果为空

- 确认图片格式支持（JPG、PNG、BMP）
- 检查图片是否包含清晰可识别的文本

## 文件结构

```
OCRapp/
├── compose.yml          # Docker Compose 配置
├── nginx.conf           # Nginx 反向代理配置
├── index.html           # 前端页面
└── README.md            # 本文件
```

## 参考资源

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Docker Compose](https://docs.docker.com/compose/)

