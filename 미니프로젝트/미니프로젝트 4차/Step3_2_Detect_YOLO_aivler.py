import cv2
import numpy as np
from ultralytics import YOLO
import time  # 타이머 추가

# 모델 불러오기
model = YOLO('./my_face_aug_yolo11.pt', task='classify')

# OpenCV에서 사용할 카메라
cap = cv2.VideoCapture(0)

# 카메라 실행 확인
if not cap.isOpened():
    print('웹캠 실행 불가')
    exit()

# 시작 시간 기록
start_time = time.time()

count = 0
my_Face = 0
# 매 프레임마다 실행될 무한 루프
while True:
    # 경과 시간 계산
    elapsed_time = time.time() - start_time
    if elapsed_time > 100:  # 100초가 지나면 종료
        print('100초 경과 - 프로그램 종료')
        print(f'Step 3: YOLOv11n: {my_Face}: {count}, {(my_Face/(count))*100:.2f}%')
        break
    
    ret, frame = cap.read()
    if not ret:
        print('프레임 로드 불가')
        break
    
    frame = frame.astype(np.uint8)
    frame = cv2.flip(frame, 1)  # 카메라 좌우 반전
    
    # 예측값 생성
    results = model.predict(source=frame, verbose=True, save=False)
    
    # 예측된 얼굴 정보 처리
    for r in results:
        r_b = r.boxes
        if not r_b.cls is None:
            for idx in range(len(r_b)):
                x1, y1, x2, y2 = int(r_b.xyxy[idx][0]), int(r_b.xyxy[idx][1]), int(r_b.xyxy[idx][2]), int(r_b.xyxy[idx][3])
                
                if r_b.cls[idx] == 0:  # 다른 사람 얼굴로 예측
                    color = (0, 0, 255)
                    conf = r_b.conf[idx] * 100
                    label_text = f'Other Face conf-score: {conf:.2f}'
                    count+=1
                else:  # 본인 얼굴로 예측
                    color = (0, 255, 0)
                    conf = r_b.conf[idx] * 100
                    count+=1
                    my_Face+=1
                    label_text = f'My Face conf-score: {conf:.2f}'
                
                # 프레임에 바운딩 박스와 텍스트 추가
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 프레임 화면에 표시
    cv2.imshow('Face_Detection', frame)
    
    # 'q' 키로 반복문 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
