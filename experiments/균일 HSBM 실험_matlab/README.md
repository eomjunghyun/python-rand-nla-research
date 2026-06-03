# 균일 HSBM 실험 MATLAB 구현

이 폴더는 `experiments/균일 HSBM 실험`의 Python 실험을 MATLAB 코드로 다시 실행하기 위한 별도 구현입니다. Python `src.common`을 호출하지 않고, 3-uniform HSBM 생성, normalized hypergraph operator 구성, 네 가지 spectral clustering 방법, metric 계산, CSV/PNG/Markdown 저장을 MATLAB 안에서 수행합니다.

## 실행

프로젝트 루트에서 실행합니다.

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/균일 HSBM 실험_matlab'); run_all_uniform_hsbm_matlab('reps',10,'seed',20260506,'no_progress',true)"
```

간단한 smoke test:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/균일 HSBM 실험_matlab'); run_all_uniform_hsbm_matlab('reps',1,'smoke',true)"
```

## 산출물

결과는 이 폴더 아래에 저장됩니다.

- `results/EXP-20260506-007_uniform_hsbm_n_rho16_eigsh_methods_matlab/`
- `results/EXP-20260506-008_uniform_hsbm_K_rho16_eigsh_methods_matlab/`
- `results/EXP-20260506-009_uniform_hsbm_rho_eigsh_methods_matlab/`
- `uniform_hsbm_matlab_experiment_report.md`

비교 방법은 Python 실험과 같은 네 가지입니다.

- `Non-random eigs`
- `Gaussian RP`
- `Random sampling`
- `CountSketch RP`

MATLAB과 Python/NumPy는 RNG, eigensolver, k-means 초기화가 다르므로 수치가 완전히 같지는 않습니다. 대신 생성식, metric, 출력 구조를 같은 실험으로 비교할 수 있게 맞췄습니다.
