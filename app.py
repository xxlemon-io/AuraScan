"""
PaddleOCR FastAPI 后端服务
提供 OCR 文字识别 API 接口
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import uvicorn
import numpy as np
from PIL import Image
import io
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="PaddleOCR API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 OCR 实例（延迟初始化）
ocr = None


def init_ocr():
    """初始化 PaddleOCR 实例"""
    global ocr
    if ocr is None:
        logger.info("正在初始化 PaddleOCR...")
        try:
            # 使用中文 OCR 模型 v3
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
            logger.info("PaddleOCR 初始化成功")
        except Exception as e:
            logger.error(f"PaddleOCR 初始化失败: {e}")
            raise
    return ocr


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化 OCR"""
    init_ocr()


@app.get("/")
async def root():
    """健康检查端点"""
    return {"status": "ok", "message": "PaddleOCR API is running"}


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
        # 获取 OCR 实例
        ocr_instance = init_ocr()
        
        # 读取上传的图片
        image_bytes = await images.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 numpy 数组
        img_array = np.array(image)
        
        # 执行 OCR 识别
        logger.info(f"开始识别图片: {images.filename}")
        result = ocr_instance.ocr(img_array, cls=True)
        
        # 格式化结果以匹配 Hub Serving 的响应格式
        formatted_results = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    formatted_results.append({
                        "text": line[1][0],  # 识别的文本
                        "confidence": float(line[1][1]),  # 置信度
                        "text_region": line[0]  # 文本区域坐标
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

