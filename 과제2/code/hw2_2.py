import sys
from ultralytics import YOLO

def check_empire_state(model_path, image_path, conf_threshold=0.5):
    # 모델 로드
    model = YOLO(model_path)

    # 객체 감지 수행
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    #엠파이어스테이트 빌딩이 있는 경우에 True 출력
    for result in results:
        for box in result.boxes.data:
            cls_id = int(box[5])
            if cls_id == 1:
                print("True")
                return
    print("False")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hw2_2.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "./best.pt"

    check_empire_state(model_path, image_path)
