import cv2  # OpenCV 라이브러리 import
import numpy as np  # Numpy 라이브러리 import
import sys  # 시스템 관련 모듈 import

# 전역 변수 설정
points = []  # 사용자가 선택한 4개의 좌표를 저장할 리스트
image = None  # 현재 이미지
original_image = None  # 원본 이미지

# 마우스 클릭 이벤트 처리 함수
def on_mouse_click(event, x, y, flags, param):
    global points, image
    # 왼쪽 버튼 클릭 시, 최대 4개의 좌표를 저장
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])  # 클릭한 좌표 추가
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 클릭한 위치에 초록색 원 그리기
        cv2.imshow("Image", image)  # 이미지 갱신하여 표시

# 원근 변환 함수
def perspective_transform(image_path):
    global image, original_image, points
    image = cv2.imread(image_path)  # 이미지 파일 읽기

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    original_image = image.copy()  # 원본 이미지 복사본 생성

    cv2.namedWindow("Image")  # 이미지 창 생성
    cv2.imshow("Image", image)  # 이미지를 화면에 표시
    cv2.setMouseCallback("Image", on_mouse_click)  # 마우스 클릭 이벤트 등록

    # 사용자가 4개의 점을 선택할 때까지 기다림
    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:  # 4개의 점이 선택되면 루프 종료
            break
        elif key == 27:  # ESC 키를 누르면 프로그램 종료
            print("프로그램을 종료합니다.")
            cv2.destroyAllWindows()
            return  # sys.exit() 대신 함수 종료로 변경

    cv2.destroyWindow("Image")  # 이미지 창 닫기

    pts1 = np.float32(points)  # 선택한 좌표를 float32 타입으로 변환

    # 변환 후 이미지의 폭과 높이 계산
    width_top = np.linalg.norm(pts1[0] - pts1[1])  # 상단 폭
    width_bottom = np.linalg.norm(pts1[3] - pts1[2])  # 하단 폭
    width = max(int(width_top), int(width_bottom))  # 폭 중 큰 값 선택

    height_left = np.linalg.norm(pts1[0] - pts1[3])  # 왼쪽 높이
    height_right = np.linalg.norm(pts1[1] - pts1[2])  # 오른쪽 높이
    height = max(int(height_left), int(height_right))  # 높이 중 큰 값 선택

    # 원근 변환 후 목적지 좌표 설정
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 원근 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(original_image, matrix, (width, height))  # 변환 적용

    cv2.imshow("Transformed", result)  # 변환된 이미지 표시

    # 임의의 키 입력 시 종료
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()  # 모든 창 닫기

# 메인 함수
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_2.py <이미지 경로>")
    else:
        perspective_transform(sys.argv[1])
