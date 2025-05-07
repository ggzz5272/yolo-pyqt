from PyQt6 import uic
from PyQt6.QtWidgets import QApplication
import csv
import cv2
from PyQt6.QtGui import QImage, QPixmap
import os
from PyQt6.QtCore import QTimer
import torch  # PyTorch 라이브러리 추가
import sys
sys.path.append("yolov5")  # YOLOv5 디렉토리를 경로에 추가
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes  # scale_coords -> scale_boxes로 수정
from utils.torch_utils import select_device

hiddenimports=['cv2']

Form, Window = uic.loadUiType("res/gui.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# YOLOv5 모델 로드
device = select_device('cpu')  # CPU 사용 명시
model_path = "c:/robot/pygui2/yolov5/porthole.pt"  # YOLOv5 모델 경로 수정

# 모델 파일 경로 확인
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)

# 로드된 모델 경로 확인
print(f"Loaded model from: {model_path}")

def detect_objects(frame):
    """
    YOLOv5 모델을 사용하여 객체 감지 수행.
    """
    results = model(frame)  # YOLOv5 모델 추론
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]  # 라벨과 좌표 추출
    return labels, coords

def update_frame():
    ret, frame = cap.read()
    if ret:
        # YOLOv5 객체 감지 수행
        labels, coords = detect_objects(frame)

        # 감지된 객체를 프레임에 표시 (예: 경계 상자)
        for coord in coords:
            x1, y1, x2, y2, conf = coord
            if conf > 0.5:  # 신뢰도 임계값
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # OpenCV 이미지를 PyQt 이미지로 변환
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        form.labelwep.setPixmap(QPixmap.fromImage(qt_image))

def save_data():
    # 올바른 속성 이름으로 수정합니다.
    name = form.lineEditname.text()
    phone = form.lineEditphone.text()
    memo = form.textEditmemo.toPlainText()
    photo_path = form.lineEditphoto.text()  # 사진 경로 가져오기

    # TXT 파일로 저장합니다.
    with open("output.txt", mode="w", encoding="utf-8") as file:
        file.write("Name\tPhone\tMemo\tPhoto\n")
        file.write(f"{name}\t{phone}\t{memo}\t{photo_path}\n")

    print("Data saved to output.txt!")

def save_image():
    # lineEditname에서 이름을 가져옵니다.
    name = form.lineEditname.text()
    if not name:
        print("Name is empty. Cannot save image.")
        return

    # 이미지 저장 경로 설정
    photos_dir = os.path.join(os.getcwd(), "photos")
    os.makedirs(photos_dir, exist_ok=True)  # photos 디렉토리 생성 (이미 존재하면 무시)
    filename = f"{name}.png"
    filepath = os.path.join(photos_dir, filename)

    # 현재 프레임 저장
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filepath, frame)
        form.lineEditphoto.setText(filepath)  # 사진 경로를 lineEditphoto에 설정
        print(f"Image saved as {filepath}")
    else:
        print("Failed to capture image.")

# 타이머를 사용해 labelwep에 웹캠 프레임 업데이트
timer = QTimer()
timer.timeout.connect(update_frame)
timer.start(30)

form.btnSave.clicked.connect(save_data)
form.btnsaveimage.clicked.connect(save_image)

# 종료 시 웹캠 해제
app.aboutToQuit.connect(lambda: cap.release())

window.show()
app.exec()