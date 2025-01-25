import cv2
import numpy as np
import sys


def automatic_perspective_transform(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 적용
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 적응형 이진화
    thresh = cv2.adaptiveThreshold(gray_blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # 엣지 검출
    edges = cv2.Canny(thresh, 50, 150)
    # 닫힘 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 컨투어 찾기
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("윤곽을 찾을 수 없습니다.")
        return

    # 모든 컨투어를 면적 기준으로 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 사각형 찾기
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            break
    else:
        print("사각형을 찾을 수 없습니다.")
        return

    # 투시 변환 적용
    maxWidth, maxHeight = get_max_width_height(rect)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image, matrix, (int(maxWidth), int(maxHeight)))

    cv2.imshow("Transformed", result)

    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()
    sys.exit()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # 좌상단
    rect[2] = pts[np.argmax(s)]  # 우하단
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단

    return rect


def get_max_width_height(rect):
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    return (maxWidth, maxHeight)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_3.py <이미지 경로>")
    else:
        automatic_perspective_transform(sys.argv[1])
