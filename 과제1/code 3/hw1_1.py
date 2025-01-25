import cv2
import numpy as np
import sys

def count_checkerboard_squares(image_path):
    image = cv2.imread(image_path)  # 이미지 파일 읽기

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 대비 향상을 위한 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # 가우시안 블러로 노이즈 감소
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 에지 검출
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # 허프 변환을 이용한 선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=50, maxLineGap=10)

    if lines is None:
        print("선을 검출할 수 없습니다.")
        return

    # 수직선과 수평선 분류
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 기울기 계산
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # 수직선 (기울기 약 90도)
            if abs(angle) > 80:
                vertical_lines.append((x1, y1, x2, y2))
            # 수평선 (기울기 약 0도)
            elif abs(angle) < 10:
                horizontal_lines.append((x1, y1, x2, y2))

    if not vertical_lines or not horizontal_lines:
        print("충분한 수직선 또는 수평선을 검출하지 못했습니다.")
        return

    # 수직선과 수평선 정렬
    vertical_lines.sort(key=lambda x: x[0])
    horizontal_lines.sort(key=lambda x: x[1])

    # 수직선과 수평선의 x, y 좌표 추출
    vertical_positions = [ (x1 + x2) // 2 for x1, y1, x2, y2 in vertical_lines ]
    horizontal_positions = [ (y1 + y2) // 2 for x1, y1, x2, y2 in horizontal_lines ]

    # 중복되는 좌표 제거 (근사값 처리)
    def remove_duplicates(positions, tolerance=10):
        if not positions:
            return []
        positions = sorted(positions)
        unique_positions = [positions[0]]
        for pos in positions[1:]:
            if abs(pos - unique_positions[-1]) > tolerance:
                unique_positions.append(pos)
        return unique_positions

    unique_vertical_positions = remove_duplicates(vertical_positions)
    unique_horizontal_positions = remove_duplicates(horizontal_positions)

    num_cols_detected = len(unique_vertical_positions) - 1
    num_rows_detected = len(unique_horizontal_positions) - 1

    # 예상되는 체커보드의 크기 (8x8 또는 10x10)
    def decide_size(num):
        return 8 if abs(num - 8) < abs(num - 10) else 10

    num_cols = decide_size(num_cols_detected)
    num_rows = decide_size(num_rows_detected)

    # 최종적으로 결정된 행과 열의 수를 출력
    if num_rows == num_cols and num_rows in [8, 10]:
        print(f"{num_rows} x {num_cols}")
    else:
        # 행과 열이 다르게 검출되었거나 8, 10이 아닌 경우 가까운 값으로 조정
        avg_num = int(round((num_rows_detected + num_cols_detected) / 2))
        final_size = decide_size(avg_num)
        print(f"{final_size} x {final_size}")

    # 결과 시각화 (옵션)
    # 수직선과 수평선 그리기
    for x in unique_vertical_positions:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 1)
    for y in unique_horizontal_positions:
        cv2.line(image, (0, y), (image.shape[1], y), (255, 0, 0), 1)

    # 교차점 그리기
    for x in unique_vertical_positions:
        for y in unique_horizontal_positions:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_1.py <이미지 경로>")
    else:
        count_checkerboard_squares(sys.argv[1])

