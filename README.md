# AuraScan

基于 Tesseract OCR 的现代化 OCR 文字识别应用，提供简洁美观的 Web 界面。

## 特性

- 🚀 基于 Tesseract OCR 引擎（内置 `tessdata_best` 高精度模型）
- 🎨 现代化 UI 设计（玻璃拟态风格）
- 🔒 安全的内部网络架构（仅前端端口暴露）
- 📱 响应式设计，支持桌面和移动端
- 🖼️ 支持拖拽、粘贴、点击上传图片
- 🌏 支持中文和英文识别
- 🧠 自适应图像预处理 + 文本区域切分，显著提升中文识别精度
- ♻️ 自动置信度监控与单字模式回退，弱场景也能兜底

## 快速开始

### 前置要求

- Docker
- Docker Compose

### 启动服务

```bash
docker-compose up -d
```

### 访问应用

打开浏览器访问：`http://localhost:54811`

### 停止服务

```bash
docker-compose down
```

## 架构说明

```
宿主机:54811 (仅此端口暴露)
    ↓
Nginx (webapp)
    ├─ 静态文件: / → index.html
    └─ API 代理: /api/ → ocr-api:8000 (内部网络)
```

- **前端服务** (`webapp`): 暴露 `54811` 端口，提供 Web 界面
- **API 服务** (`ocr-api`): 仅内部访问，通过 Nginx 反向代理

## API 使用

### 端点

- **URL**: `/api/predict/ocr_system`
- **方法**: `POST`
- **Content-Type**: `multipart/form-data`

### 请求示例

```bash
curl -X POST http://localhost:54811/api/predict/ocr_system \
  -F "images=@test_image.jpg" \
  -F "mode=single_line" \
  -F "whitelist=乐快乐"
```

#### 可选参数（查询字符串与 FormData 均支持）

| 参数名      | 类型 | 说明 |
|-------------|------|------|
| `mode`      | `single_char` / `single_line` / `single_block` / `auto` | 指定场景，自动切换 `psm` |
| `psm`       | 整数 | 直接指定 Tesseract `--psm` 模式，优先级最高 |
| `whitelist` | 字符串 | 允许出现的字符（自动过滤危险符号）|
| `blacklist` | 字符串 | 禁止出现的字符 |

> 示例：`/api/predict/ocr_system?mode=single_char&whitelist=乐呵`

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

### 关键环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OCR_TESSDATA_PROFILE` | `best` | `best` / `default`，控制使用 `tessdata_best` 还是系统默认模型 |
| `OCR_LANG` | `chi_sim+eng` | Tesseract 语言组合 |
| `TESSERACT_CONFIG` | `--oem 1 --psm 6` | 基础配置，可追加 `--dpi 300` 等 |
| `OCR_FALLBACK_THRESHOLD` | `0.5` | 平均置信度低于该值时自动触发单字模式重试 |
| `OCR_TESSDATA_BEST_DIR` | `/usr/share/tesseract-ocr/4.00/tessdata_best` | 高精度模型目录 |
| `OCR_TESSDATA_DEFAULT_DIR` | `/usr/share/tesseract-ocr/4.00/tessdata` | 系统默认模型目录 |

### 修改 OCR 语言

Tesseract 支持多种语言识别。默认配置为中文（简体）和英文。

如需修改语言，可设置 `OCR_LANG` 或在 `app.py` 中指定：

```python
ocr_data = pytesseract.image_to_data(
    image,
    lang='chi_sim+eng',  # 修改为所需语言代码
    ...
)
```

常用语言代码：
- `chi_sim`: 中文（简体）
- `chi_tra`: 中文（繁体）
- `eng`: 英文
- `jpn`: 日文
- `kor`: 韩文

多个语言用 `+` 连接，如 `chi_sim+eng+jpn`。

### 查看可用语言

在容器内运行：

```bash
docker-compose exec ocr-api tesseract --list-langs
```

## 常见问题

### 服务启动失败

```bash
# 检查端口占用
netstat -ano | findstr :54811

# 查看日志
docker-compose logs ocr-api
docker-compose logs webapp
```

### API 请求失败

- 确认服务正常运行：`docker-compose ps`
- 检查 Nginx 配置：`docker-compose exec webapp cat /etc/nginx/conf.d/default.conf`
- 查看日志：`docker-compose logs webapp`

### 识别结果为空

- 确认图片格式支持（JPG、PNG、BMP）
- 检查图片是否包含清晰可识别的文本
- 确认所需语言包已安装（默认包含中文和英文）

### 中文识别效果不佳

- 确保图片清晰，文字对比度高
- 尝试调整图片大小和分辨率
- 检查是否安装了中文语言包：`tesseract --list-langs | grep chi_sim`
- 合理使用 `mode`/`psm`：海报、横幅建议 `mode=single_line`，大字号单字使用 `mode=single_char`
- 配置 `whitelist` 仅保留目标字符，可显著降低噪声
- 通过 `OCR_TESSDATA_PROFILE=best` 切换至高精度模型
- 后端已包含灰度化、倾斜校正、自适应阈值、区域切分及低置信度自动重试；若仍乱码，请提供样例图便于进一步分析

## 文件结构

```
AuraScan/
├── compose.yml          # Docker Compose 配置
├── Dockerfile           # API 服务镜像构建文件
├── nginx.conf           # Nginx 反向代理配置
├── index.html           # 前端页面
├── app.py               # FastAPI 后端服务
├── requirements.txt     # Python 依赖
└── README.md            # 本文件
```

## 参考资源

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [pytesseract](https://github.com/madmaze/pytesseract)
- [Docker Compose](https://docs.docker.com/compose/)
