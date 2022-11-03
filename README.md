# ViCon

## Touchless Youtube Control Application
&nbsp;Vicon은 머신러닝을 이용해 사용자의 제스처에 따라 유튜브를 조작하는 애플리케이션이다.

<img src="https://user-images.githubusercontent.com/76546008/199517656-3dfa9c72-b42a-4a35-ba7e-71af7f0b21c6.gif" width="640"/>

---

## 목차

1. [프로젝트 배경](#프로젝트-배경)
1. [블록 다이어그램](#블록-다이어그램)
1. [내용](#내용)
1. [시행착오](#시행착오)
1. [파일 설명](#파일-설명)
1. [참고 링크](#참고_링크)
1. [Dependency](#Dependency)
1. [Contirubutors](#Contirubutors)

시연 영상 1 : https://youtu.be/tMfvdW4jtSI  
시연 영상 2 : https://youtu.be/-JKxQX7OKOY

---

## 프로젝트 배경

- Why: 레시피 영상이 요리 초보의 진도에 맞춰 재생되게 하고 싶다
- What: 레시피 영상이 요리 초보의 진도에 맞춰 재생되게 하는 시스템
- How: 안드로이드 어플리케이션

---

## 블록 다이어그램
<!-- <img width="864" alt="스크린샷 2022-03-30 오전 3 32 08" src="https://user-images.githubusercontent.com/88064555/160681059-60287651-0453-441f-8509-bf327c3f328f.png"> -->

<img src="https://user-images.githubusercontent.com/76546008/199517363-ed363795-2dea-4610-9d3e-736b280c2274.png" width="640"/>

### 역할 : 머신러닝을 이용한 영상 제스처 분류

---

## 내용

- 프로젝트 소요기간 : 3개월(초기 버전) / 1개월(리뉴얼 버전)

&nbsp;본 프로젝트는 중간에 리뉴얼을 통해 시스템구조를 변경하였다. 따라서 내용은 초기 버전의 시스템과 리뉴얼 후 버전의 시스템으로 나누어 설명하겠다.

### __초기 버전__

&nbsp;처음에는 다양한 레퍼런스들을 참고하여 서버를 가지는 구조로 설계하였다.

모바일 기기로부터 카메라 입력 → 서버에서 클래스 분류 후 기기로 결과값 전송 → 기기에서 처리하여 동작 수행

&nbsp;서버에서 수행하는 동작이 Tensorflow를 이용하고, 이를 위해 우선적으로 머신러닝 모델을 학습시켜야 한다. 영상을 촬영하여 클래스별로 데이터 생성 및 수집을 하였다. 영상데이터를 처리하는 파이프라인 모듈로는 Mediapipe를 사용하였다. 데이터를 토대로 데이터셋을 생성 시계열 모델을 생성하였으나, 여러 시도에도 한계점 발생하였다. 이는 [시행착오](#시행착오)에서 서술하겠다. 이후, 시계열 모델이 아닌 이미지 분류 모델로 변경하였다.

&nbsp;위와 같이 서버를 구성하고, 모바일 기기와 서버간 통신에는 소켓과 Wowza서버를 사용하였다. 영상통신에는 RTSP, RTMP, SRT, NDI 등 다양한 프로토콜이 있으나 다른 조원이 담당했던 부분으로 프로젝트 기간과 비용을 고려했을때 Wowza서버를 이용한 RTMP가 적절하다고 전달받았다.

&nbsp;최종적으로, 모바일 기기에서 Wowza서버로 스트리밍 → 서버에서는 Wowza서버로부터 영상을 받아 클래스 분류 후 소켓 통신을 통해 기기로 결과값 전송 → 기기에서 처리의 구조로 작동되었다.

### __리뉴얼 버전__

&nbsp;기존 시스템의 딜레이 문제의 원인을 통신으로 생각하여, 안드로이드 Stand-alone으로 시스템 구조 변경을 시도하였다. 따라서, 기존 파이썬 서버에서 작동하던 Tensorflow와 Mediapipe를 안드로이드 버전으로 바꾸고 리팩토링을 진행하였다. Tensorflow는 Tensorflow Lite라는 버전을 사용하였다. Tensorflow Lite는 모델 학습은 불가하고, 클래스 분류 및 예측만 가능하며 학습의 경우에는 데스크탑 환경에서 학습을 한 뒤 tflite라는 모델로 변환하는 과정이 필요하다. Mediapipe도 안드로이드 솔루션을 이용해 진행하였다. 그러나, 기기간의 성능 편차가 큰 점, 적은 프로젝트 예산으로 인한 낮은 모바일 기기 성능, 그리고 리뉴얼의 목표가 딜레이의 최소화였기에 configuration을 저성능의 기기에 맞추어 설정하였다.

&nbsp;미리 설정한 제스처 동작을 이용해 모델을 학습, tflite모델로 변환하여 애플리케이션 패키지에 포함시켰다. 시스템에서는 카메라 입력을 Mediapipe를 통해 데이터화시키고 이 데이터를 tflite모델에 입력, 클래스 출력값에 따라 동작을 수행하도록 하였다. 딜레이가 1.5초내로 실시간 수준의 성능을 보였다.

---

## 시행착오

1. 시계열 모델 생성  
<img width="640" alt="image" src="https://user-images.githubusercontent.com/76546008/199625964-e801b0c7-04a8-407f-a7e1-660669f5ac45.png">  
&nbsp;초기 프로젝트 설계할때는 영상으로 된 시계열 데이터를 생각하였다. 이미지가 아닌 비디오말이다. 위의 그림에서의 45프레임은 임의로 정한 수치이다. 비디오는 이미지들의 연속, 프레임들로 구성되어있다. 각각의 프레임에서 Mediapipe를 통해 Hand Detection 및 데이터화를 수행하고, 45프레임의 데이터를 묶어 하나의 제스처로 분류한다. 이렇게 학습시킨 모델은 3개 정도의 클래스는 100%에 가깝게 분류하나, 클래스의 수가 늘어나면 정확도가 현저히 낮아졌다.  
&nbsp;분석하여 문제의 원인을 모델 구성, 데이터의 내용, 최적화 함수 총 3가지를 꼽았다.  
&nbsp; *첫째* , 머신러닝과 딥러닝에 대한 지식을 소화하는데 시간이 상당히 걸리기에 약식으로 익혀서 적절한 모델 구성을 하지 못하였다. 입력층과 출력층을 빼고, 은닉층의 레이어 개수와 종류에 대해, 역할과 관계에 대한 상관관계에 대한 이해가 부족하였다.  
&nbsp; *둘째* , 데이터의 내용이 상이하였다. 프레임 단위로 보면, 손이 떨리거나 너무 빨리 움직여 이미지에 손의 형태가 온전치 못한 프레임이 비디오에 끼어있는데, 이 때 Mediapipe에서 Hand Detection을 하지 못한다. 그러면, 데이터가 누락되고 이는 데이터의 손실을 초래한다. 이를 처리하기 위한 방법들로 중간값과 RMS를 사용하는 등의 방법을 고안해보았으나 크게 해결되지 않았다.  
&nbsp; *마지막으로* , 최적화 함수에 대한 이해 부족이다. 딥러닝 신경망 모델에는 다양한 최적화 함수가 존재한다. 특정 입력 벡터 혹은 변수에 가중치를 얼마나 두는지, 손실치를 얼마나 두는지 등의 알고리즘을 말한다. 대표적으로 Dense 레이어의 Sigmoid 함수, Softmax 함수, Relu 함수 등이 있다. 그러나, 이러한 함수의 이해를 위해서는 수치해석 및 확률 분포, 벡터 등에 대한 많은 이해가 필요하다. 이에 대한 부재로 적절한 함수로 계층을 구성하지 못하였다.  
&nbsp;결국 시계열 데이터를 통한 클래스 분류는 포기하고 단일 프레임에 대한 분류로 대체하게 되었다.

1. Data Augmentation  
&nbsp;적은 데이터로 모델을 생성하기 위해서는 Data Augmentation이 필수적이다. 회전, x축 이동, y축 이동, 비율 변화 등 다양한 이미지 변형이 존재하는데, 이를 변형하여도 Mediapipe를 통해 데이터화하면 동일 데이터가 출력되었다.  
&nbsp;원인을 분석하였더니, 이미지 변형을 하여도, Mediapipe에서 손의 데이터를 생성하는 알고리즘이 palm detection을 우선적으로하고 landmark를 추출한다. 이러한 방식은 손바닥을 기준으로 손가락과 마디의 상대적인 위치에 따른 계산이여서 이미지를 변형하여도 비율 보정으로 인해 동일한 데이터가 생성되는 것이다. 따라서, Data Augmentation은 불가했다.

1. 딜레이 발생  
&nbsp;기존 시스템 구조인 서버 구조의 경우 5 ~ 15초 가량의 딜레이가 발생했다. 원인은 Wowza서버로의 통신때문이라고 생각한다. 이 점 때문에 Wowza서버의 네트워크 환경과 로컬 기기의 네트워크 환경에 매우 의존적이고, 편차가 큰 딜레이가 발생한다.

1. 안드로이드 지식 부족  
&nbsp;리뉴얼 전에도 안드로이드를 담당하여 진행했던 조원이 있었으나, 그 조원도 안드로이드 프로그래밍을 능숙하게 하지는 못하였다. 리뉴얼 후에 모든 조원이 안드로이드 프로그래밍을 하게 되었는데 처음 접하였고 프로젝트 기간이 충분치 않아 최적화하지는 못했다. 그렇기에 애플리케이션의 안정성에 문제가 많았고 최종적으로 구글 플레이 스토어 배포에는 실패했다.

<!-- ---

## 파일 설명 -->

---

## Dependency

- [Mediapipe](https://google.github.io/mediapipe/)
- [Tensorflow](https://www.tensorflow.org/?hl=ko)
- [Wowza Server](https://www.wowza.com/)

---

## Contirubutors
- 김진만
- [이정연](https://github.com/happyyeon)
- [최동혁](https://github.com/donghyeok1)

<!-- ---

## 폴더 설명

+ **Solution1_Communication**

    <img width="209" alt="image" src="https://user-images.githubusercontent.com/88064555/167547070-464eddce-374d-4903-ab27-0f8b1b99894f.png">

    통신을 이용하여 Why를 구현한 모델이다. 안드로이드에서 동영상과 카메라/마이크가 병행적으로 실행된다. 사용자가 입력한 제스쳐/음성을 받아들여 RTMP 서버로 송출한다.(서버는 WOWZA를 사용하였다.) 동시에 Python 클라이언트에서 서버로 송출된 제스쳐/음성을 가져온다. 충분히 학습된 데이터셋을 기반으로 가져온 제스쳐/음성을 판단하고 이를 명령어로 변환한다. 이를 다시, 안드로이드 클라이언트로 명령어를 반환하여 동영상을 제어한다.


+ **Solution2_Only_Android**

    <img width="165" alt="image" src="https://user-images.githubusercontent.com/88064555/167548145-e85c7c54-2327-45aa-8d17-bce473b41dd2.png">

    오직 안드로이드 개발환경만을 이용하여 블록 다이어그램을 구현한 모델이다. [솔루션 1]과 비교하였을 때 장점은 통신부를 제거하여 Delay 문제를 개선할 수 있다. 사용자가 입력한 제스쳐/음성을 별도의 서버로 송출하지 않고 안드로이드 앱 자체에서 해당 입력을 판단하여 이를 명령어로 변환하여 즉각적으로 동영상을 제어한다. 

+ **not_use**

    _백업용 폴더_

    현재 사용하지 않지만 혹시 모를 상황에 대비하여 백업을 위한 폴더이다. 프로젝트를 진행하며 폴더가 많아지면 헷갈리므로 안 쓰는 파일들을 여기에 저장한다. -->

---

## 참고링크

1. [Mediapipe](https://google.github.io/mediapipe/) / [Mediapipe Github](https://github.com/google/mediapipe)
2. [Rock-Paper-Scissors-Machine](https://github.com/kairess/Rock-Paper-Scissors-Machine)
3. [gesture-recognition](https://github.com/kairess/gesture-recognition)
4. [gesture-user-interface](https://github.com/kairess/gesture-user-interface)
5. [음성인식으로 시작하는 딥러닝](https://wikidocs.net/book/2553)
6. [케라스 강좌](https://tykimos.github.io/lecture/)
7. [OpenCV 정보](https://bkshin.tistory.com/category/OpenCV)
8. [프로젝트 레퍼런스 모음](https://www.computervision.zone/projects/)
9. [프로젝트 레퍼런스](https://medium.com/@sunminlee89)

---

## 이미지

![03_03_문장_로고타입_조합_국영문_가로_1Color_typeB](https://user-images.githubusercontent.com/88064555/160678993-70372853-5ca5-42bf-85a6-de9f68d5f888.jpg)
