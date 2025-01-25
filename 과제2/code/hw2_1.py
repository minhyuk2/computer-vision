import os
import struct
import cv2
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog

def load_mnist(path, kind):

    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def extract_orb_features(images, max_features=128):
    orb = cv2.ORB_create(nfeatures=max_features)
    feature_list = []
    for img in images:
        reshaped_img = img.reshape(28, 28).astype(np.uint8)
        keypoints, descriptors = orb.detectAndCompute(reshaped_img, None)
        if descriptors is not None:
            feature_list.append(descriptors.flatten()[:max_features])
        else:
            feature_list.append(np.zeros(max_features, dtype=np.float32))
    return np.array(feature_list, dtype=np.float32)

def extract_sift_features(images, max_features=128):
    sift = cv2.SIFT_create()
    feature_list = []
    for img in images:
        reshaped_img = img.reshape(28, 28).astype(np.uint8)
        keypoints, descriptors = sift.detectAndCompute(reshaped_img, None)
        if descriptors is not None:
            feature_list.append(descriptors.flatten()[:max_features])
        else:
            feature_list.append(np.zeros(max_features, dtype=np.float32))
    return np.array(feature_list, dtype=np.float32)

def extract_hog_features(images):
    feature_list = []
    for img in images:
        reshaped_img = img.reshape(28, 28)
        hog_feat = hog(
            reshaped_img,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        feature_list.append(hog_feat)
    return np.array(feature_list, dtype=np.float32)

def scale_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def measure_time_and_accuracy(model, X_train, X_test, y_train, y_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    train_time = end_train - start_train
    test_time = end_test - start_test

    cm = confusion_matrix(y_test, y_pred)

    return accuracy, train_time, test_time, cm

def main():
    data_path = './mnist_data'

    X_train_full, y_train = load_mnist(data_path, kind='train')
    X_test_full, y_test = load_mnist(data_path, kind='t10k')

    # 데이터 정규화
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test_full = X_test_full.astype(np.float32) / 255.0

    # =============================================================
    # 1) kNN + ORB
    # =============================================================
    print("\n[1) kNN + ORB]")
    X_train_orb = extract_orb_features(X_train_full)
    X_test_orb = extract_orb_features(X_test_full)
    # 스케일링
    X_train_orb_scaled = scale_data(X_train_orb)
    X_test_orb_scaled = scale_data(X_test_orb)

    knn_orb = KNeighborsClassifier(n_neighbors=3)
    acc_orb, train_t_orb, test_t_orb, cm_orb = measure_time_and_accuracy(
        knn_orb, X_train_orb_scaled, X_test_orb_scaled, y_train, y_test
    )
    print(f"정확도: {acc_orb:.4f}, 학습 시간: {train_t_orb:.2f}s, 추론 시간: {test_t_orb:.2f}s")
    print("혼동 행렬:")
    print(cm_orb)

    # =============================================================
    # 2) kNN + SIFT
    # =============================================================
    print("\n[2) kNN + SIFT]")
    X_train_sift = extract_sift_features(X_train_full)
    X_test_sift = extract_sift_features(X_test_full)
    X_train_sift_scaled = scale_data(X_train_sift)
    X_test_sift_scaled = scale_data(X_test_sift)

    knn_sift = KNeighborsClassifier(n_neighbors=3)
    acc_sift, train_t_sift, test_t_sift, cm_sift = measure_time_and_accuracy(
        knn_sift, X_train_sift_scaled, X_test_sift_scaled, y_train, y_test
    )
    print(f"정확도: {acc_sift:.4f}, 학습 시간: {train_t_sift:.2f}s, 추론 시간: {test_t_sift:.2f}s")
    print("혼동 행렬:")
    print(cm_sift)

    # =============================================================
    # 3) SVM + ORB
    # =============================================================
    print("\n[3) SVM + ORB]")
    X_train_orb2 = extract_orb_features(X_train_full)
    X_test_orb2 = extract_orb_features(X_test_full)
    X_train_orb2_scaled = scale_data(X_train_orb2)
    X_test_orb2_scaled = scale_data(X_test_orb2)

    svm_orb = SVC(kernel='linear')
    acc_svm_orb, train_t_svm_orb, test_t_svm_orb, cm_svm_orb = measure_time_and_accuracy(
        svm_orb, X_train_orb2_scaled, X_test_orb2_scaled, y_train, y_test
    )
    print(f"정확도: {acc_svm_orb:.4f}, 학습 시간: {train_t_svm_orb:.2f}s, 추론 시간: {test_t_svm_orb:.2f}s")
    print("혼동 행렬:")
    print(cm_svm_orb)

    # =============================================================
    # 4) SVM + HOG
    # =============================================================
    print("\n[4) SVM + HOG]")
    X_train_hog = extract_hog_features(X_train_full)
    X_test_hog = extract_hog_features(X_test_full)
    X_train_hog_scaled = scale_data(X_train_hog)
    X_test_hog_scaled = scale_data(X_test_hog)

    svm_hog = SVC(kernel='linear')
    acc_svm_hog, train_t_svm_hog, test_t_svm_hog, cm_svm_hog = measure_time_and_accuracy(
        svm_hog, X_train_hog_scaled, X_test_hog_scaled, y_train, y_test
    )
    print(f"정확도: {acc_svm_hog:.4f}, 학습 시간: {train_t_svm_hog:.2f}s, 추론 시간: {test_t_svm_hog:.2f}s")
    print("혼동 행렬:")
    print(cm_svm_hog)

    # =============================================================
    # 5) kNN + HOG
    # =============================================================
    print("\n[5) kNN + HOG]")
    X_train_hog_knn = extract_hog_features(X_train_full)
    X_test_hog_knn = extract_hog_features(X_test_full)

    X_train_hog_knn_scaled = scale_data(X_train_hog_knn)
    X_test_hog_knn_scaled = scale_data(X_test_hog_knn)

    knn_hog = KNeighborsClassifier(n_neighbors=3)
    acc_knn_hog, train_t_knn_hog, test_t_knn_hog, cm_knn_hog = measure_time_and_accuracy(
        knn_hog, X_train_hog_knn_scaled, X_test_hog_knn_scaled, y_train, y_test
    )

    print(f"정확도: {acc_knn_hog:.4f}, 학습 시간: {train_t_knn_hog:.2f}s, 추론 시간: {test_t_knn_hog:.2f}s")
    print("혼동 행렬:")
    print(cm_knn_hog)

    # =============================================================
    # 결과 요약
    # =============================================================
    print("\n[비교 결과 요약]")
    print("1) kNN + ORB   => 정확도: {:.4f}, 학습시간: {:.2f}s, 추론시간: {:.2f}s".format(acc_orb, train_t_orb, test_t_orb))
    print("2) kNN + SIFT  => 정확도: {:.4f}, 학습시간: {:.2f}s, 추론시간: {:.2f}s".format(acc_sift, train_t_sift, test_t_sift))
    print("3) SVM + ORB   => 정확도: {:.4f}, 학습시간: {:.2f}s, 추론시간: {:.2f}s".format(acc_svm_orb, train_t_svm_orb, test_t_svm_orb))
    print("4) SVM + HOG   => 정확도: {:.4f}, 학습시간: {:.2f}s, 추론시간: {:.2f}s".format(acc_svm_hog, train_t_svm_hog, test_t_svm_hog))
    print("5) kNN + HOG   => 정확도: {:.4f}, 학습시간: {:.2f}s, 추론시간: {:.2f}s".format(acc_knn_hog, train_t_knn_hog, test_t_knn_hog))

if __name__ == "__main__":
    main()
