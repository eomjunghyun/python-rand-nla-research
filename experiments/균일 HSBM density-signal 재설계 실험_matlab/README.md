# 균일 HSBM density-signal 재설계 실험 MATLAB 구현

이 폴더는 `experiments/균일 HSBM density-signal 재설계 실험`의 Python 실험을 MATLAB 코드로 다시 실행하기 위한 별도 구현입니다. 공통 MATLAB 구현은 `experiments/균일 HSBM 실험_matlab/+uhsbm/Common.m`을 사용합니다.

## 실행

프로젝트 루트에서 실행합니다.

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/균일 HSBM density-signal 재설계 실험_matlab'); run_all_redesigned_uniform_hsbm_matlab('reps',5,'seed',20260507,'no_progress',true)"
```

간단한 smoke test:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/균일 HSBM density-signal 재설계 실험_matlab'); run_all_redesigned_uniform_hsbm_matlab('reps',1,'smoke',true)"
```

## 산출물

결과는 이 폴더 아래에 저장됩니다.

- `results/*/*_raw.csv`
- `results/*/*_summary.csv`
- `results/*/*_config.json`
- `results/*/*_summary.png`
- `results/diagnostics/*.csv`
- `redesigned_uniform_hsbm_matlab_experiment_report.md`

Python 재설계 실험과 같이 strong-signal sweep과 weak-gap diagnostic sweep을 분리해 저장합니다.
