# Reference 1 Section 7.2 모델 1~6 실험

이 폴더는 Reference 1 논문의 Section 7.2에 제시된 Model 1~6 실험을 재현하는 스크립트를 담고 있다. 실행 중 진행 상황, 단계별 시간, 논문과 맞춘 metric을 기록한다.

## 실행 파일

- `sec72_models123_live.py`: Model 1~3 실행
- `sec72_models456_live.py`: Model 4~6 실행

## 실행 방법

```bash
python experiments/reference_1_section7_2/sec72_models123_live.py \
  --reps 20 \
  --seed 2026

python experiments/reference_1_section7_2/sec72_models456_live.py \
  --reps 20 \
  --seed 2026
```

## 결과 저장 위치

기본 출력은 아래 폴더에 저장된다.

- `experiments/reference_1_section7_2/results/exp72_models123_paper_aligned_live/`
- `experiments/reference_1_section7_2/results/exp72_models456_paper_aligned_live/`

주요 출력 파일은 raw per-rep CSV, mean/std summary CSV, metric figure, runtime figure다.

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. 논문 모델명, 파일명, metric 이름은 영어 표기를 유지할 수 있다.
