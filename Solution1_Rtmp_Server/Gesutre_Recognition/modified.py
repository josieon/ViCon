import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
import socket

### 서버로부터 영상 입력받아 모델로 예측 수행
### 예측 결과값을 클라이언트로 전송해주는 프로그램

actions = ['rew','adv', 'stop', 'OK']           # 제스처 프리셋
model = load_model('models/mode_1648573934.h5') # 모델 로드
# 소켓 configuration
port = 44444
host = '223.194.7.72'
server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))

# 소켓 통신
server_sock.listen(1)
print("기다리는 중")
client_sock, addr = server_sock.accept()
print('Connected by', addr)

## wowza 스트리밍 서버와 통신
# n = int(input())
# if n==1:
#  cap=cv2.VideoCapture("rtmp://223.194.7.87:1935/live/test")

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
  max_num_hands = 1,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5
)

cap = cv2.VideoCapture(0)
action_seq = [] # 예측 결과 배열
cmd_seq = []    # 더블 체크를 위한 배열
cmdmode = 0     # 더블 체크 flag
input_gesture = None

while cap.isOpened:
  ret, img = cap.read()
  if not ret:
    continue
  # flag에 따라 결과값을 넣는 배열 결정
  if cmdmode == 0:
    action_seq.append(def_gesture(img))
  else:
    cmd_seq.append(def_gesture(img))

  # 폰트 출력
  img = cv2.flip(img, 1)
  if cmdmode == 1:
    remain_time = 10 - int(time.time()) + int(entered_time)
    if remain_time < 0:
      cmdmode = 0
    cv2.putText(img,
                text = f'{input_gesture}? Waiting for {remain_time} secs',
                org = (10, 30),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color = (0, 0, 0))
  img = cv2.resize(img, (1000, 750))
  cv2.imshow('Game', img)
  # escape 처리
  if cv2.waitKey(1) == ord('q'):
    break

  # flag 0 일때(normal case)
  if cmdmode == 0:
    # 30프레임 이상일때만 처리
    if len(action_seq) < 30:
      continue
    # if문안의 비교문 개수로 프레임별 비교, 확정의 경우
    if len(set(action_seq[-30:-1])) == 1 and action_seq[-1] != None and action_seq[-1] != 'OK':
      # 결과값 결정
      input_gesture = action_seq[-1]
      # 배열 clear
      action_seq.clear()
      cmdmode = 1
      entered_time = time.time()
  
  # flag 1 일때(double check)
  else:
    # 30프레임 이상일때만 처리
    if len(cmd_seq) < 30:
      continue
    # double check를 묻고, 맞다면 출력
    if len(set(cmd_seq[-30:-1])) == 1 and cmd_seq[-1] == 'OK':
      print(actions.index(input_gesture))
      client_sock.send(actions.index(input_gesture).to_bytes(4, byteorder='little'))
      cmdmode = 0
      # 배열 clear
      cmd_seq.clear()

# 데이터셋을 생성했던 연산과 동일한 연산 수행하여
# 모델에 데이터 입력, predict 수행
def def_gesture(img):
  img = cv2.flip(img, 1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  result = hands.process(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  if result.multi_hand_landmarks is not None:
    for res in result.multi_hand_landmarks:
      joint = np.zeros((21, 3))   # [x, y, z]를 가지는 21개의 점 배열 생성
      for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z]
      # Compute angles between joints
      v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]    # Parent joint
      v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :] # Child joint
      v = v2 - v1 # [20, 3]
      v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
      # Get angle using arcos of dot product
      angle = np.arccos(np.einsum('nt,nt->n',
      v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
      v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :])) # [15,]
      angle = np.degrees(angle) # Convert radian to degree
      data = np.array([angle],dtype = np.float32)
      # 모델에 예측.
      y_pred = model.predict(data)
      i_pred = int(np.argmax(y_pred))
      # conf는 라벨일 확률
      conf = y_pred[:, i_pred]
      # 일정 확률 이상일 때, 확정 및 반환
      if conf > 0.999:
        return actions[i_pred]
      else:
        return None