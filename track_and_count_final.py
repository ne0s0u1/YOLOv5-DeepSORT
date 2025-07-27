# track_and_count_final.py (v3.0 - 视觉与稳定性增强版)

import sys
import torch
import cv2
import numpy as np

# --- 导入我们自己的 DeepSORT 模块 ---
try:
    from deepsort_for_yolo import DeepSort
except ImportError:
    print("\n[错误] DeepSORT 模块 (deepsort_for_yolo.py) 未找到。")
    sys.exit(1)

def main():
    # --- 1. 参数配置 ---
    video_path = 'test.mp4'  # <-- 修改为您自己的视频文件路径
    output_path = 'output_video_enhanced.mp4'
    yolo_model_source = './yolov5'
    
    CONF_THRESHOLD = 0.45  # 稍微提高置信度，过滤掉一些不确定的检测，有助于稳定
    # 类别COCO: 0:person, 2:car, 3:motorcycle, 5:bus, 7:truck
    TARGET_CLASSES = [0, 2, 3, 5, 7] 

    # --- 2. 初始化模型 ---
    print("正在加载模型...")
    # 加载 YOLOv5 (建议使用中等模型以获得更好的检测效果)
    try:
        model = torch.hub.load(yolo_model_source, 'yolov5m', source='local', pretrained=True)
    except Exception as e:
        print(f"\n[错误] YOLOv5 模型加载失败。")
        sys.exit(1)
        
    model.conf = CONF_THRESHOLD
    
    # <<< 关键修改：调整DeepSORT参数以获得更稳定的跟踪 >>>
    # max_age: 物体失踪多少帧后才会被删除。增加此值可应对短暂遮挡。
    # n_init: 物体连续被检测到多少帧后才被确认为新轨迹。增加此值可减少因噪声产生的错误轨迹。
    try:
        deepsort = DeepSort(
            model_path="ckpt.t7",
            max_dist=0.2,
            min_confidence=0.3,
            nms_max_overlap=0.5,
            max_iou_distance=0.7,
            max_age=90,  # 从 70 增加到 90
            n_init=4,    # 从 3 增加到 4
            nn_budget=100,
            use_cuda=torch.cuda.is_available()
        )
    except Exception as e:
        print(f"\n[错误] DeepSORT 初始化失败。")
        sys.exit(1)

    print("模型加载完成。")

    # --- 3. 视频处理设置 ---
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    LINE_X_POSITION = width // 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- 4. 计数器和状态变量 ---
    memory = {}
    already_counted = set()
    
    # <<< 关键修改：为不同类别定义更鲜艳的颜色 >>>
    # BGR 格式: (Blue, Green, Red)
    COLOR_PERSON = (0, 255, 255)  # 亮黄色
    COLOR_VEHICLE = (255, 0, 255) # 亮粉色 (洋红)
    
    counters = {
        0: {'L_to_R': 0, 'R_to_L': 0, 'name': 'Person', 'color': COLOR_PERSON},
        2: {'L_to_R': 0, 'R_to_L': 0, 'name': 'Car', 'color': COLOR_VEHICLE},
        3: {'L_to_R': 0, 'R_to_L': 0, 'name': 'Motorcycle', 'color': COLOR_VEHICLE},
        5: {'L_to_R': 0, 'R_to_L': 0, 'name': 'Bus', 'color': COLOR_VEHICLE},
        7: {'L_to_R': 0, 'R_to_L': 0, 'name': 'Truck', 'color': COLOR_VEHICLE}
    }
    
    print("开始处理视频...")
    frame_idx = 0
    
    # --- 5. 主循环 ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        
        # 5.1 YOLOv5 检测
        results = model(frame)
        
        bbox_xywh, confidences, class_ids = [], [], []
        
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) in TARGET_CLASSES:
                x_center, y_center = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                bbox_xywh.append([x_center, y_center, w, h])
                confidences.append(float(conf))
                class_ids.append(int(cls))

        if len(bbox_xywh) > 0:
            # 5.2 DeepSORT 更新
            outputs = deepsort.update(torch.tensor(bbox_xywh), confidences, class_ids, frame)
            
            # 5.3 处理跟踪结果并计数
            if len(outputs) > 0:
                for x1, y1, x2, y2, track_id, cls_id in outputs:
                    # <<< 关键修改：增强视觉效果 >>>
                    color = counters.get(cls_id, {}).get('color', (255, 255, 255)) # 默认为白色
                    box_thickness = 3 # 框线粗细
                    font_scale = 1.2 # 字体大小
                    font_thickness = 2 # 字体粗细
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                    
                    label = f"ID:{track_id} {counters.get(cls_id, {}).get('name', '')}"
                    # 计算标签文本的尺寸以便绘制背景
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1) # 绘制实心背景
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness) # 黑色字体

                    # 计数逻辑 (保持不变)
                    center_x = (x1 + x2) // 2
                    if track_id not in memory: memory[track_id] = []
                    memory[track_id].append(center_x)
                    if len(memory[track_id]) >= 2:
                        prev_x = memory[track_id][-2]
                        if prev_x < LINE_X_POSITION and center_x >= LINE_X_POSITION and (track_id, 'L_to_R') not in already_counted:
                            if cls_id in counters: counters[cls_id]['L_to_R'] += 1
                            already_counted.add((track_id, 'L_to_R'))
                        elif prev_x > LINE_X_POSITION and center_x <= LINE_X_POSITION and (track_id, 'R_to_L') not in already_counted:
                            if cls_id in counters: counters[cls_id]['R_to_L'] += 1
                            already_counted.add((track_id, 'R_to_L'))
        
        # 绘制垂直线
        cv2.line(frame, (LINE_X_POSITION, 0), (LINE_X_POSITION, height), (0, 255, 255), 4) # 加粗黄线
        
        y_offset = 50
        text_font_scale = 1.5
        text_thickness = 3
        # 合并车辆计数
        total_vehicle_L_to_R = sum(counters[cls_id]['L_to_R'] for cls_id in [2, 3, 5, 7] if cls_id in counters)
        total_vehicle_R_to_L = sum(counters[cls_id]['R_to_L'] for cls_id in [2, 3, 5, 7] if cls_id in counters)
        
        person_L_to_R = counters.get(0, {}).get('L_to_R', 0)
        person_R_to_L = counters.get(0, {}).get('R_to_L', 0)
        
        vehicle_text = f"Vehicle: L->R: {total_vehicle_L_to_R}, R->L: {total_vehicle_R_to_L}"
        person_text = f"Person: L->R: {person_L_to_R}, R->L: {person_R_to_L}"

        # 修正了参数顺序
        cv2.putText(frame, vehicle_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, (0, 0, 0), text_thickness + 2) # 黑色描边
        cv2.putText(frame, vehicle_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, COLOR_VEHICLE, text_thickness)
        y_offset += 60
        cv2.putText(frame, person_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, (0, 0, 0), text_thickness + 2) # 黑色描边
        cv2.putText(frame, person_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, COLOR_PERSON, text_thickness)

        # 写入输出视频
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("处理完成。")

if __name__ == '__main__':
    main()