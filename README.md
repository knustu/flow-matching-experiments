# Flow Matching 독립 실험

작은 Flow Matching 모델에서 2D 분포를 생성하고, Euler 단계 수, 미니배치 Optimal Transport, 조건 정보가 결과에 미치는 영향을 실험했습니다.

## 2026-09-07 실험 기록

[원본 실험 보고서](Flow_Matching_Report.pdf)에 세 시드의 설정과 결과 그림을 담았습니다.

| 실험 | 관찰 |
| --- | --- |
| 2D 생성 | Two moons, 8 Gaussians, Spiral에서 큰 구조가 보였지만 세부 구조가 과도하게 평활화됐습니다. |
| Euler 단계 수 | 무작위 연결 모델의 Two moons Sliced W₂는 5단계 0.1982에서 100단계 0.1287로 줄었습니다. |
| 미니배치 OT | 5단계 Sliced W₂에서 무작위 연결 대비 Two moons 48.5%, 8 Gaussians 20.5%, Spiral 49.3% 감소를 관찰했습니다. |
| MNIST 조건부 생성 | 48차원 PCA 좌표에서 조건부 생성했습니다. 별도 숫자 분류기의 조건 일치율은 10단계 92.5%, 50단계 90.7%였습니다. |
| 합성 시계열 | 두 갈래 미래를 생성했습니다. FM의 Energy Score는 1.6894, MSE 기준점 예측의 점질량은 3.1813이었습니다. 평균 경로 MSE는 각각 0.5623, 0.5568이었습니다. |

세 시드는 고정된 데이터 분할에서 초기화·학습 샘플링·생성 잡음을 바꾼 반복입니다. 시계열 결과는 정답 생성식을 아는 합성 데이터에서 측정했습니다. 원본 실행기와 체크포인트의 보관 위치는 추가 확인이 필요합니다.

## 실행 가능한 2D 재현 예제

[toy_fm.py](toy_fm.py)는 보고서의 Two moons 설정을 참고해 이번에 새로 작성한 작은 예제입니다. 모델 초기화와 평가 표본을 고정하고, 무작위 연결과 미니배치 OT를 두고 다섯 가지 Euler 단계 수를 비교합니다.

~~~sh
python -m pip install -r requirements.txt
python toy_fm.py --steps 600 --seed 11
~~~

2026-11-25 실행에서 5단계 Sliced W₂는 무작위 연결 0.188756, 미니배치 OT 0.103457이었습니다. 이 수치는 재현 예제의 1시드 결과입니다.

## 참고

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Conditional Flow Matching with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)
