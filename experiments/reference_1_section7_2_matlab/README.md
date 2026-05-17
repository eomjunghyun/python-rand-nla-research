# Reference 1 Section 7.2 MATLAB 구현

이 폴더는 `experiments/reference_1_section7_2`의 Python 실험을 MATLAB 코드만으로 다시 구현한 버전입니다. Python `src.common`을 호출하지 않고, 모델 생성, Gaussian random projection, CountSketch random projection, random sampling, non-random baseline, metric 계산, CSV/PNG 저장을 `+sec72/Common.m` 안에 독립적으로 넣었습니다.

## 이 환경에서 MATLAB 실행

MATLAB은 `/Applications/MATLAB_R2026a.app/bin/matlab`에 설치되어 있습니다. Codex 샌드박스 안에서는 MathWorks 캐시 쓰기가 막혀 실패할 수 있으므로, 배치 실행은 샌드박스 밖 승인으로 실행해야 합니다.

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section7_2_matlab'); run_sec72_models123_matlab('reps',20,'seed',2026)"

/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section7_2_matlab'); run_sec72_models456_matlab('reps',20,'seed',2026)"
```

두 묶음을 한 번에 실행하려면:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section7_2_matlab'); run_all_sec72_matlab('reps',20,'seed',2026)"
```

간단한 smoke test:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section7_2_matlab'); run_all_sec72_matlab('reps',1,'n_values',[200],'seed',2026)"
```

## 결과 저장 위치

기본 출력은 새 MATLAB 폴더 아래에 저장됩니다.

- `experiments/reference_1_section7_2_matlab/results/exp72_models123_paper_aligned_live/`
- `experiments/reference_1_section7_2_matlab/results/exp72_models456_paper_aligned_live/`

파일명과 컬럼 형식은 Python 버전과 맞췄습니다.

- `sec72_models123_raw_per_rep.csv`
- `sec72_models123_summary_mean_std.csv`
- `sec72_models123_metrics_figure5_like.png`
- `sec72_models123_runtime.png`
- `sec72_models456_raw_per_rep.csv`
- `sec72_models456_summary_mean_std.csv`
- `sec72_models456_metrics_figure6_like.png`
- `sec72_models456_runtime.png`

현재 summary에는 네 방법이 포함됩니다.

- `Non-random`
- `Random Projection`
- `Random Sampling`
- `CountSketch`

보고서는 `section7_2_matlab_experiment_report.md`에 있습니다.

MATLAB과 NumPy의 RNG, eigensolver, k-means 초기화가 다르므로 수치 값이 Python 결과와 완전히 같지는 않습니다. 대신 실험 정의, metric, 출력 스키마, 파일 구조가 같은 형태가 되도록 맞췄습니다.
