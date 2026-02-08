import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import transforms
import torch
import cv2
import os
import random

model = None
device = None

MOCK_TRANSLATIONS = {
    'ja': {("日本のマンガ", "Японская манга"), ("攻撃！", "Атака!"), ("完", "Конец"), 
           ("愛", "Любовь"), ("死ね！", "Умри!")},
    'ko': {("만화", "Манхва"), ("공격", "Атака")}
}

def init_models():
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Устройство: {device}")
    
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        num_classes = 3
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        model_path = "models/detection_fasterrcnn.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("✅ Модель загружена!")
        else:
            print("⚠️  Демо-режим (без модели)")
            
        model.to(device)
        model.eval()
        return "✅ Все модели готовы!"
    except Exception as e:
        print(f"⚠️  Ошибка: {e}")
        return f"⚠️  Ошибка инициализации: {e}"

def process_detection(image):
    """🔍 Детекция пузырей"""
    global model
    if model is None:
        return image, "🔧 Сначала нажмите 'Инициализировать'"
    
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for box, label in zip(predictions[0]['boxes'], predictions[0]['labels']):
        x1, y1, x2, y2 = box.cpu().numpy()
        if label.item() == 1: 
            color, text = 'red', "📝 Текст"
        elif label.item() == 2: 
            color, text = 'blue', "💥 SFX"
        else: 
            continue
        
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=3, 
                               edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-15, text, color=color, fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    return Image.open(buf), f"🔍 Найдено: {len(predictions[0]['boxes'])} объектов"

def process_translation(image):
    """✨ Полный перевод"""
    global model
    if model is None:
        return image, "🔧 Сначала 'Инициализировать'"
    
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    result = image.copy()
    count = 0
    
    for box, label in zip(predictions[0]['boxes'], predictions[0]['labels']):
        if label.item() not in [1, 2]: continue
            
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        crop = image.crop((x1, y1, x2, y2))
        
        # Mock перевод
        orig, trans = random.choice(list(MOCK_TRANSLATIONS['ja']))
        
        # Простая замена текста
        draw = ImageDraw.Draw(crop)
        font_size = max(16, min(crop.height // 4, 32))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0,0), trans, font=font)
        x = (crop.width - (bbox[2]-bbox[0])) // 2
        y = (crop.height - (bbox[3]-bbox[1])) // 2
        
        # Обводка
        for dx in [-2, -1, 1, 2]:
            for dy in [-2, -1, 1, 2]:
                draw.text((x+dx, y+dy), trans, font=font, fill=(255,255,255))
        draw.text((x, y), trans, font=font, fill=(0,0,0))
        
        result.paste(crop, (x1, y1))
        count += 1
    
    return result, f"✅ Переведено: {count} пузырей"

# ✅ ИСПРАВЛЕННЫЙ Gradio 6.0
def create_interface():
    with gr.Blocks(title="🎌 Автопереводчик манги") as demo:
        gr.Markdown("# 🚀 Манга → Русский (автоматически!)")
        
        with gr.Row():
            input_image = gr.Image(label="📁 Загрузите страницу", type="pil", height=500)
            output_image = gr.Image(label="🎯 Результат", type="pil", height=500)
        
        with gr.Row():
            init_btn = gr.Button("🔧 Инициализировать", variant="stop")
            detect_btn = gr.Button("🔍 Детекция", variant="secondary")
            translate_btn = gr.Button("✨ Перевод", variant="primary")
        
        status = gr.Textbox(label="📊 Статус", interactive=False)
        
        gr.Examples(
            examples=[
                ["Manga/MangaAisazuNihaIrarenai/000.jpg"],
                ["Manga/AkkeraKanjinchou/000.jpg"],
            ],
            inputs=[input_image],
            label="📚 Тестовые примеры"
        )
        
        # Кнопки
        init_btn.click(fn=init_models, outputs=status)
        detect_btn.click(fn=process_detection, inputs=[input_image], outputs=[output_image, status])
        translate_btn.click(fn=process_translation, inputs=[input_image], outputs=[output_image, status])
    
    return demo

if __name__ == "__main__":
    print("🎌 Запуск автопереводчика манги...")
    print("📁 Модель: models/detection_fasterrcnn.pth")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        debug=True
        # ✅ Убраны устаревшие параметры Gradio 6.0
    )
