"""
Tesseract OCR FastAPI 后端服务
提供 OCR 文字识别 API 接口
"""
import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import re
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract 语言与配置支持环境变量覆盖
OCR_LANG = os.getenv("OCR_LANG", "chi_sim+eng")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--oem 1 --psm 6")
OCR_TESSDATA_PROFILE = os.getenv("OCR_TESSDATA_PROFILE", "best").lower()
OCR_TESSDATA_DEFAULT_DIR = os.getenv(
    "OCR_TESSDATA_DEFAULT_DIR", "/usr/share/tesseract-ocr/4.00/tessdata"
)
OCR_TESSDATA_BEST_DIR = os.getenv(
    "OCR_TESSDATA_BEST_DIR", "/usr/share/tesseract-ocr/4.00/tessdata_best"
)


def configure_tessdata_prefix():
    """根据环境变量选择 tessdata 目录"""
    dir_map = {
        "best": OCR_TESSDATA_BEST_DIR,
        "default": OCR_TESSDATA_DEFAULT_DIR,
    }
    target_dir = dir_map.get(OCR_TESSDATA_PROFILE, OCR_TESSDATA_BEST_DIR)

    if target_dir and Path(target_dir).exists():
        os.environ["TESSDATA_PREFIX"] = target_dir
        logger.info("TESSDATA_PREFIX set to %s", target_dir)
    else:
        logger.warning(
            "指定的 tessdata 目录不存在 (%s)，继续使用系统默认目录",
            target_dir,
        )

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
    configure_tessdata_prefix()
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


CHARSET_ALLOWED_PATTERN = re.compile(
    r"[^0-9A-Za-z\u4e00-\u9fff，。！？：；、“”‘’（）《》〈〉·—_%+\-=]"
)


def deskew(gray_image: np.ndarray) -> np.ndarray:
    """简单倾斜矫正"""
    coords = np.column_stack(np.where(gray_image < 255))
    if coords.size == 0:
        return gray_image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return gray_image
    height, width = gray_image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    针对中文文本进行基础预处理：
    1. 转灰度
    2. 自适应放大（最短边 < 800 像素时）
    3. 去噪 + 自适应/OTSU 二值化 + 形态学处理
    """
    rgb_image = image.convert("RGB")
    img_array = np.array(rgb_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    min_dim = min(height, width)
    if min_dim < 800:
        scale = 800 / min_dim
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = deskew(gray)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    combined = cv2.bitwise_and(otsu_binary, adaptive)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

    return Image.fromarray(morphed), morphed


def detect_text_regions(binary_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """通过投影 + 轮廓检测亮色文本区域"""
    inverted = 255 - binary_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    regions: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 5000:
            continue
        regions.append((x, y, w, h))
    regions.sort(key=lambda box: (box[1], box[0]))
    return regions


def sanitize_charset(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    clean = CHARSET_ALLOWED_PATTERN.sub("", value)
    clean = clean.replace(" ", "")
    return clean or None


def determine_psm(image: Image.Image, user_psm: Optional[int], mode: Optional[str]) -> int:
    if user_psm:
        return user_psm

    normalized_mode = (mode or "").lower()
    if normalized_mode == "single_char":
        return 10
    if normalized_mode == "single_line":
        return 7
    if normalized_mode == "single_block":
        return 6

    width, height = image.size
    if height == 0 or width == 0:
        return 6

    aspect = width / height
    shortest = min(width, height)

    if shortest < 200:
        return 10
    if aspect >= 2.0:
        return 7
    if aspect <= 0.5:
        return 5
    return 6


def build_tesseract_config(psm: int, whitelist: Optional[str], blacklist: Optional[str]) -> str:
    config_parts = [TESSERACT_CONFIG.strip(), f"--psm {psm}"]
    if whitelist:
        config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    if blacklist:
        config_parts.append(f"-c tessedit_char_blacklist={blacklist}")
    return " ".join(part for part in config_parts if part)


def accumulate_ocr_lines(
    ocr_data: dict,
    offset_x: int,
    offset_y: int,
    region_tag: str,
    container: dict,
) -> None:
    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        try:
            conf = float(ocr_data["conf"][i])
        except (ValueError, TypeError):
            conf = 0.0

        if ocr_data["level"][i] != 5 or conf <= 0 or not text:
            continue

        left = ocr_data["left"][i] + offset_x
        top = ocr_data["top"][i] + offset_y
        width = ocr_data["width"][i]
        height = ocr_data["height"][i]

        x1, y1 = left, top
        x2, y2 = left + width, top
        x3, y3 = left + width, top + height
        x4, y4 = left, top + height

        key = (
            region_tag,
            ocr_data["block_num"][i],
            ocr_data["par_num"][i],
            ocr_data["line_num"][i],
        )

        if key not in container:
            container[key] = {
                "text": text,
                "confidences": [conf],
                "bbox": {"min_x": x1, "min_y": y1, "max_x": x3, "max_y": y3},
            }
        else:
            container[key]["text"] += f" {text}"
            container[key]["confidences"].append(conf)
            bbox = container[key]["bbox"]
            bbox["min_x"] = min(bbox["min_x"], x1)
            bbox["min_y"] = min(bbox["min_y"], y1)
            bbox["max_x"] = max(bbox["max_x"], x3)
            bbox["max_y"] = max(bbox["max_y"], y3)


def lines_to_results(lines: dict) -> tuple[list[dict], list[float]]:
    results = []
    confidences = []
    for key in sorted(lines.keys()):
        line_data = lines[key]
        avg_confidence = (
            sum(line_data["confidences"]) / len(line_data["confidences"])
            if line_data["confidences"]
            else 0.0
        )
        bbox = line_data["bbox"]
        confidence_value = avg_confidence / 100.0
        confidences.append(confidence_value)
        results.append(
            {
                "text": line_data["text"],
                "confidence": confidence_value,
                "text_region": [
                    [bbox["min_x"], bbox["min_y"]],
                    [bbox["max_x"], bbox["min_y"]],
                    [bbox["max_x"], bbox["max_y"]],
                    [bbox["min_x"], bbox["max_y"]],
                ],
            }
        )
    return results, confidences


def run_ocr_with_config(
    image: Image.Image,
    regions: list[tuple[int, int, int, int]],
    config: str,
) -> tuple[list[dict], list[float]]:
    lines: dict = {}
    for region_idx, (rx, ry, rw, rh) in enumerate(regions):
        region_image = image.crop((rx, ry, rx + rw, ry + rh))
        ocr_data = pytesseract.image_to_data(
            region_image,
            lang=OCR_LANG,
            output_type=pytesseract.Output.DICT,
            config=config,
        )
        accumulate_ocr_lines(ocr_data, rx, ry, f"region_{region_idx}", lines)
    return lines_to_results(lines)


@app.get("/")
async def root():
    """健康检查端点"""
    return {"status": "ok", "message": "Tesseract OCR API is running"}


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/predict/ocr_system")
async def predict_ocr_system(
    request: Request,
    images: UploadFile = File(...),
    mode: Optional[str] = Form(None),
    whitelist: Optional[str] = Form(None),
    blacklist: Optional[str] = Form(None),
    psm: Optional[int] = Form(None),
):
    """
    OCR 识别接口
    接收图片文件，返回识别结果
    """
    try:
        # Query 参数优先级更高
        query_mode = request.query_params.get("mode")
        if query_mode:
            mode = query_mode
        query_psm = request.query_params.get("psm")
        if query_psm:
            try:
                psm = int(query_psm)
            except ValueError:
                logger.warning("收到非法 psm 参数: %s", query_psm)

        query_whitelist = request.query_params.get("whitelist")
        if query_whitelist:
            whitelist = query_whitelist
        query_blacklist = request.query_params.get("blacklist")
        if query_blacklist:
            blacklist = query_blacklist

        sanitized_whitelist = sanitize_charset(whitelist)
        sanitized_blacklist = sanitize_charset(blacklist)

        # 读取上传的图片
        image_bytes = await images.read()
        original_image = Image.open(io.BytesIO(image_bytes))

        try:
            image, binary_map = preprocess_image(original_image)
            logger.info("已对上传图片进行预处理")
        except Exception as preprocess_error:
            logger.warning(f"图片预处理失败，直接使用原图: {preprocess_error}")
            image = original_image.convert("L")
            binary_map = np.array(image.convert("L"))

        regions = detect_text_regions(binary_map)
        if not regions:
            regions = [(0, 0, image.width, image.height)]
        logger.info("检测到 %s 个文本区域", len(regions))
        
        selected_psm = determine_psm(image, psm, mode)
        effective_config = build_tesseract_config(
            selected_psm, sanitized_whitelist, sanitized_blacklist
        )
        logger.info(
            "OCR 配置: lang=%s, psm=%s, mode=%s, whitelist=%s, blacklist=%s",
            OCR_LANG,
            selected_psm,
            mode or "auto",
            sanitized_whitelist if sanitized_whitelist else "None",
            sanitized_blacklist if sanitized_blacklist else "None",
        )

        # 执行 OCR 识别
        logger.info(f"开始识别图片: {images.filename}")
        formatted_results, confidence_values = run_ocr_with_config(
            image, regions, effective_config
        )
        last_config_used = effective_config
        mean_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        logger.info("初次识别平均置信度: %.2f", mean_confidence)

        fallback_threshold = float(os.getenv("OCR_FALLBACK_THRESHOLD", "0.5"))
        if (
            (not formatted_results or mean_confidence < fallback_threshold)
            and selected_psm != 10
        ):
            logger.info(
                "低置信度 (%.2f < %.2f)，触发单字模式重试",
                mean_confidence,
                fallback_threshold,
            )
            fallback_config = build_tesseract_config(
                10, sanitized_whitelist, sanitized_blacklist
            )
            fallback_results, fallback_confidences = run_ocr_with_config(
                image, regions, fallback_config
            )
            if fallback_results:
                formatted_results = fallback_results
                confidence_values = fallback_confidences
                last_config_used = fallback_config
                mean_confidence = (
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else 0.0
                )
                logger.info(
                    "单字模式成功，平均置信度提升到 %.2f", mean_confidence
                )
            else:
                logger.warning("单字模式重试仍未获得有效结果")
        
        # 如果没有识别到文本，尝试使用简单的 image_to_string 方法
        if not formatted_results:
            logger.warning("使用 image_to_data 未识别到文本，尝试使用 image_to_string")
            simple_text = pytesseract.image_to_string(
                image, lang=OCR_LANG, config=last_config_used
            ).strip()
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
