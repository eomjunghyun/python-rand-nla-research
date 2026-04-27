# Reference 2 Experiment 2: Figure 3/4/5

이 폴더는 논문 "A Stochastic Block Hypergraph model"의 Experiment 2를 재현한다.

## 재현 대상

- Figure 3: degree distribution, hyperedge size distribution, binomial 근사
- Figure 4: effective parameter인 `E*/E`, `N*/N`과 `q/p`의 관계
- Figure 5: `min` strategy에서 hyperedge size distribution의 bimodal 경향

공통 hypergraph 생성 로직은 `src/common.py`의 함수를 재사용한다.

## 실행 방법

```bash
python experiments/reference_2_exp2/run_figure3_4_5.py
```

환경에 따라 `python` 대신 프로젝트에서 사용하는 Python 실행 파일을 지정하면 된다.

## 출력 파일

- `experiments/reference_2_exp2/figures/figure3_degree_and_size.png`
- `experiments/reference_2_exp2/figures/figure4_effective_parameters.png`
- `experiments/reference_2_exp2/figures/figure5_min_bimodal.png`
- `experiments/reference_2_exp2/results/figure3_summary.csv`
- `experiments/reference_2_exp2/results/figure4_fit_summary.csv`
- `experiments/reference_2_exp2/results/figure5_peak_check.json`

## 구현 가정

- Figure 3/4/5의 distribution은 각 설정에서 생성한 100개 hypergraph 샘플을 풀링해 empirical PMF로 계산한다.
- degree distribution은 `Binomial(E, theta_deg)`로 근사하고, `theta_deg = mean(degree) / E`를 MLE로 사용한다.
- hyperedge size distribution은 seed node 1개로 시작하는 생성 알고리즘을 반영해 `1 + Binomial(N-1, theta_size)`로 근사한다.
- `theta_size = (mean(size) - 1) / (N - 1)`를 MLE로 사용한다.
- effective ratio는 `q/p = 1`에서의 fitted theta를 기준으로 정규화한다.

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. Figure 번호, 분포 이름, 변수명은 영어를 섞어 쓴다.
