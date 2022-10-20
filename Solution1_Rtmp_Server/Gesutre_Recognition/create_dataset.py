import cv2
import mediapipe as mp
import numpy as np
import time, os

### 입력 영상으로부터 데이터 생성
### 제스처 프리셋에 규정된 제스처들을 순차적으로 각각 입력받음
### 입력 : 제스처 영상
### 출력 : 제스처 데이터셋(csv)

actions = ['rewind','advance','stop']   # 제스처 프리셋
secs_for_action = 10                    # 제스처 입력 시간

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)               # opencv의 입력 : 웹캠
created_time = int(time.time())         # 파일 생성 시간 초기화
os.makedirs('dataset', exist_ok=True)   # dataset 디렉토리 생성

while cap.isOpened():
    data = []                           # 데이터 리스트
    for idx, action in enumerate(actions):
        ret, img = cap.read()

        # 안내 메시지 출력
        img = cv2.flip(img, 1)
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(1500)
        start_time = time.time()
        # 제스처 입력 시간 동안만 입력 및 연산 수행
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            # opencv와 mediapipe간 color system 변환
            # mediapipie로 이미지 처리
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 결과값을 이용해 연산 수행
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))   # [x, y, z, visibility]를 가지는 21개의 점 배열 생성
                    # 값 할당
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    angle = np.degrees(angle) # Convert radian to degree
                    # 배열로 변환후, 마지막에 프리셋 숫자 추가 [15] - > [16]
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)
                    # data 리스트에 값 추가
                    data.append(angle_label)
                    # mediapipe 렌더링 출력
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            # 출력 프레임 크기 조절
            cv2.resize(img, (1000, 750))
            cv2.imshow('img', img)
            # escape 처리
            if cv2.waitKey(1) == ord('q'):
                break
    # 배열로 변환 및 csv파일로 저장
    data = np.array(data)
    with open(f'dataset/data_{created_time}.csv', 'w', encoding='UTF-8', newline='') as file:
        np.savetxt(file,data, delimiter=',')
    break