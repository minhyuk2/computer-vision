import cv2
import numpy as np
import sys

def get_chess(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return 0, 0  # 두 개의 값 반환으로 수정하여 예외 발생 방지

    # 자동 투시 변환
    image = automatic_perspective_transform(image)
    if image is None:
        print("투시 변환에 실패했습니다.")
        return 0, 0  # 두 개의 값 반환으로 수정하여 예외 발생 방지

    # 흑백 이미지로 변환 및 블러 처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 원 검출
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=12, # 원 간 최소 거리
        param1=65, # 캐니 에지 검출기 상위 임계값
        param2=19, # 원 검출 임계값
        minRadius=9, # 검출할 원의 최소 반지름
        maxRadius=30 # 검출할 원의 최대 반지름
    )

    if circles is not None:
        circles = np.int32(np.around(circles[0, :]))

        # 반지름을 기준으로 원을 내림차순 정렬 후 중복 제거
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        filtered_circles = []

        for c in circles:
            center_c = np.array([c[0], c[1]])
            radius_c = c[2]
            overlap = False

            for fc in filtered_circles:
                center_fc = np.array([fc[0], fc[1]])
                radius_fc = fc[2]

                distance = np.linalg.norm(center_c - center_fc)
                tolerance = +2  # 중복 기준 조정
                # 중첩 여부 판단
                if distance < (radius_c + radius_fc + tolerance):
                    overlap = True
                    break

            if not overlap:
                filtered_circles.append(c)

        bright_count = 0
        dark_count = 0
        brightness_threshold = 128

        for i in filtered_circles:
            center = (i[0], i[1])
            small_radius = i[2] // 4

            # 중심의 작은 부분만 마스킹
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, center, small_radius, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]

            if mean_val > brightness_threshold:
                bright_count += 1
            else:
                dark_count += 1

            cv2.circle(image, center, i[2], (255, 0, 255), 2)

    else:
        bright_count = 0
        dark_count = 0


    return bright_count, dark_count


def automatic_perspective_transform(image):
    # 흑백 및 블러 처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 엣지 검출과 닫힘 연산
    edges = cv2.Canny(thresh, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("윤곽을 찾을 수 없습니다.")
        return None

    # 가장 큰 사각형 찾기
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            # Check if rectangle is axis-aligned
            if is_rectangle_axis_aligned(rect):
                # 이미지가 이미 평면이므로 변환하지 않음
                return image
            else:
                # 투시 변환 적용
                maxWidth, maxHeight = get_max_width_height(rect)
                dst = np.array(
                    [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
                    dtype="float32",
                )
                matrix = cv2.getPerspectiveTransform(rect, dst)
                result = cv2.warpPerspective(image, matrix, (int(maxWidth), int(maxHeight)))
                return result
    else:
        print("사각형을 찾을 수 없습니다.")
        return None

def is_rectangle_axis_aligned(rect, angle_threshold=5):
    # 각 변의 각도를 계산
    angles = []
    for i in range(4):
        pt1 = rect[i]
        pt2 = rect[(i + 1) % 4]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Adjust angle to be between -90 and 90 degrees
        angle = (angle + 90) % 180 - 90
        angles.append(angle)
    # 각도가 0 또는 90에 가까운지 확인
    for angle in angles:
        if not (abs(angle) < angle_threshold or abs(abs(angle) - 90) < angle_threshold):
            return False
    return True

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def get_max_width_height(rect):
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    return maxWidth, maxHeight

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_4.py <이미지 경로>")
    else:
        try:
            bright_count, dark_count = get_chess(sys.argv[1])
            print(f"w:{bright_count} b:{dark_count}")
        except TypeError:
            print("get_chess 함수에서 예기치 않은 반환값이 발생했습니다. 함수가 (bright_count, dark_count) 형식으로 반환하는지 확인하세요.")