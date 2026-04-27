# Reference 2 Experiment 1: Figure 2 Runtime

이 폴더는 논문 "A Stochastic Block Hypergraph model"의 Figure 2 runtime benchmark를 재현한다. 공통 hypergraph 생성 로직은 `src/common.py`의 함수를 재사용한다.

## 실행 방법

```bash
python experiments/reference_2_exp1/run_figure2_runtime.py
```

환경에 따라 `python` 대신 프로젝트에서 사용하는 Python 실행 파일을 지정하면 된다.

## 기본 설정

- `strategy = weighted`
- `K = 4`
- `N = E`
- `p = 100 / N`
- `q = 0.4 * p`
- `reps = 5`

## 출력 파일

- `experiments/reference_2_exp1/figures/figure2_runtime.png`
- `experiments/reference_2_exp1/results/figure2_runtime.csv`
- `experiments/reference_2_exp1/results/figure2_fit.json`

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. Figure 번호, 변수명, strategy 이름은 영어 표기를 유지할 수 있다.
