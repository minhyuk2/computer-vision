import sys
import cv2
from ultralytics import YOLO

def draw_bounding_box(image_path, model_path, conf_threshold=0.5):
    model = YOLO(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 이미지 파일을 읽을 수 없습니다: {image_path}")
        sys.exit(1)

    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls_id = box
            cls_id = int(cls_id)

            if cls_id == 1:
                # 바운딩 박스 그리기
                start_point = (int(x1), int(y1))
                end_point = (int(x2), int(y2))
                color = (0, 0, 255)
                thickness = 3
                cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.imshow("Empire State Building Detection", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hw2_3.py <image_path>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]
    MODEL_PATH = "./best.pt"

    draw_bounding_box(IMAGE_PATH, MODEL_PATH, conf_threshold=0.5)
