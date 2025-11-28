"""
Tesseract OCR FastAPI 后端服务
提供 OCR 文字识别 API 接口
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import uvicorn
from PIL import Image
import io
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="Tesseract OCR API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """应用启动时验证 Tesseract 可用性"""
    try:
        # 验证 Tesseract 是否可用
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract 版本: {version}")
        # 检查中文语言包是否可用
        langs = pytesseract.get_languages()
        logger.info(f"可用语言: {', '.join(langs)}")
        if 'chi_sim' not in langs:
            logger.warning("中文语言包 (chi_sim) 未安装，可能影响中文识别效果")
    except Exception as e:
        logger.error(f"Tesseract 初始化检查失败: {e}")
        # 不抛出异常，允许服务启动，在首次请求时再检查


@app.get("/")
async def root():
    """健康检查端点"""
    return {"status": "ok", "message": "Tesseract OCR API is running"}


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/predict/ocr_system")
async def predict_ocr_system(images: UploadFile = File(...)):
    """
    OCR 识别接口
    接收图片文件，返回识别结果
    """
    try:
        # 读取上传的图片
        image_bytes = await images.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 执行 OCR 识别
        logger.info(f"开始识别图片: {images.filename}")
        
        # 使用 pytesseract 获取详细数据（包括坐标和置信度）
        # 配置：使用中文和英文，输出结构化数据
        ocr_data = pytesseract.image_to_data(
            image,
            lang='chi_sim+eng',
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # 假设统一文本块
        )
        
        # 格式化结果以匹配 Hub Serving 的响应格式
        formatted_results = []
        
        # 按行号组织数据（level 5 是单词级别，需要按行分组）
        lines = {}
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = float(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0.0
            
            # 处理单词级别的数据（level 5）且置信度有效
            if ocr_data['level'][i] == 5 and conf > 0 and text:
                line_num = ocr_data['line_num'][i]
                left = ocr_data['left'][i]
                top = ocr_data['top'][i]
                width = ocr_data['width'][i]
                height = ocr_data['height'][i]
                
                # 计算边界框的四个角点坐标
                x1, y1 = left, top
                x2, y2 = left + width, top
                x3, y3 = left + width, top + height
                x4, y4 = left, top + height
                
                if line_num not in lines:
                    lines[line_num] = {
                        'text': text,
                        'confidences': [conf],
                        'bbox': {'min_x': x1, 'min_y': y1, 'max_x': x2, 'max_y': y3}
                    }
                else:
                    # 合并同一行的文本
                    lines[line_num]['text'] += " " + text
                    lines[line_num]['confidences'].append(conf)
                    # 扩展边界框
                    lines[line_num]['bbox']['min_x'] = min(lines[line_num]['bbox']['min_x'], x1)
                    lines[line_num]['bbox']['min_y'] = min(lines[line_num]['bbox']['min_y'], y1)
                    lines[line_num]['bbox']['max_x'] = max(lines[line_num]['bbox']['max_x'], x2)
                    lines[line_num]['bbox']['max_y'] = max(lines[line_num]['bbox']['max_y'], y3)
        
        # 转换为最终格式
        for line_num in sorted(lines.keys()):
            line_data = lines[line_num]
            avg_confidence = sum(line_data['confidences']) / len(line_data['confidences']) if line_data['confidences'] else 0.0
            bbox = line_data['bbox']
            
            formatted_results.append({
                "text": line_data['text'],
                "confidence": avg_confidence / 100.0,  # 转换为 0-1 范围
                "text_region": [
                    [bbox['min_x'], bbox['min_y']],  # 左上
                    [bbox['max_x'], bbox['min_y']],  # 右上
                    [bbox['max_x'], bbox['max_y']],  # 右下
                    [bbox['min_x'], bbox['max_y']]   # 左下
                ]
            })
        
        # 如果没有识别到文本，尝试使用简单的 image_to_string 方法
        if not formatted_results:
            logger.warning("使用 image_to_data 未识别到文本，尝试使用 image_to_string")
            simple_text = pytesseract.image_to_string(image, lang='chi_sim+eng', config='--psm 6').strip()
            if simple_text:
                # 获取整个图像的边界框
                img_width, img_height = image.size
                formatted_results.append({
                    "text": simple_text,
                    "confidence": 0.5,  # 默认置信度
                    "text_region": [[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]]
                })
        
        # 返回格式化的结果
        response_data = {
            "msg": "Success",
            "results": [formatted_results] if formatted_results else [[]]
        }
        
        logger.info(f"识别完成，共识别 {len(formatted_results)} 个文本区域")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"OCR 识别错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
