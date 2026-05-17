# Reference 1 Section 8.1 MATLAB 구현

이 폴더는 `experiments/reference_1_section8_1`의 Section 8.1 real network accuracy 실험을 MATLAB 코드로 다시 실행하는 버전입니다. 기존 Python `src`를 호출하지 않고, 데이터 준비, spectral embedding, KMeans, F1/NMI/ARI, pairwise ARI, CSV/Markdown 저장을 MATLAB 코드로 구현했습니다.

## 추가된 방법

기존 8.1의 세 방법에 두 방법을 추가했습니다.

- `CountSketch`: Gaussian random projection 대신 CountSketch sparse embedding으로 subspace를 잡습니다.
- `SIGN Bidirectional`: 첨부 논문 Wang et al. (2025)의 generalized Nyström method with subspace iteration(SIGN)을 그래프 adjacency의 spectral embedding용으로 적용합니다. 대칭 adjacency에서는 `A`와 `A'`가 같지만, 구현은 논문 구조대로 양방향 QR 갱신을 수행합니다.

## MATLAB 실행

MATLAB은 이 환경에서 `/Applications/MATLAB_R2026a.app/bin/matlab`로 실행할 수 있습니다. Codex 샌드박스에서는 MathWorks 캐시 쓰기 때문에 배치 실행을 승인된 외부 실행으로 돌려야 할 수 있습니다.

전체 8개 설정을 한 번에 실행:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section8_1_matlab'); run_all_sec81_matlab('reps',20,'seed',2026)"
```

빠른 smoke test:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section8_1_matlab'); run_all_sec81_matlab('reps',1,'seed',2026,'no_plot',true)"
```

European email만 실행:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section8_1_matlab'); run_sec81_email_matlab('reps',20,'seed',2026)"
```

## 데이터 준비

첫 실행 때 MATLAB이 원본 데이터를 읽어 `experiments/reference_1_section8_1_matlab/data/section8_1_matlab_inputs.mat` 캐시를 만듭니다.

- `data/email-Eu-core.txt`
- `data/email-Eu-core-department-labels.txt`
- `data/reference_1_section8_1/raw/polblogs/polblogs.gml`
- `data/reference_1_section8_1/raw/scc2016/SCC2016-with-abs/SCC2016/Data/authorPaperBiadj.txt`
- `data/reference_1_section8_1/raw/scc2016/SCC2016-with-abs/SCC2016/Data/paperCitAdj.txt`

캐시를 다시 만들려면:

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "addpath('experiments/reference_1_section8_1_matlab'); run_all_sec81_matlab('reps',1,'force_prepare',true,'no_plot',true)"
```

## 결과 저장 위치

출력 구조는 기존 Python 8.1 결과 폴더와 같은 형식으로 맞췄습니다.

- `results/exp8_1_email_eu_core_table2_like/`
- `results/exp8_1_email_eu_core_rank30_table2_like/`
- `results/exp8_1_political_blog_table2_like/`
- `results/exp8_1_political_blog_rank5_table2_like/`
- `results/exp8_1_statisticians_coauthor_table2_like/`
- `results/exp8_1_statisticians_coauthor_rank5_table2_like/`
- `results/exp8_1_statisticians_citation_table2_like/`
- `results/exp8_1_statisticians_citation_rank5_table2_like/`

각 폴더에는 다음 형식의 파일이 생성됩니다.

- `*_raw_per_rep.csv`
- `*_summary_mean_std.csv`
- `*_table2*_like.csv`
- `*_table2*_like.md`
- `*_pairwise_ari_raw.csv`
- `*_pairwise_ari_mean_matrix.csv`
- `*_pairwise_ari_heatmap.png`
- `*_meta.json`

종합 파일:

- `results/section8_1_matlab_rank_comparison.csv`
- `results/section8_1_matlab_rank_comparison.md`
- `section8_1_matlab_experiment_report.md`

MATLAB과 Python은 RNG, ARPACK/eigs, QR, KMeans 초기화 구현이 다르므로 숫자가 완전히 같지는 않습니다. 대신 데이터 정의, 반복 수, metric, 출력 스키마, rank 변경 설정을 맞췄습니다.
