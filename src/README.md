# `src` Modules

이 디렉터리는 실험 노트북들이 공통으로 사용하는 Python 유틸 코드를 담고 있습니다.

## 핵심 파일

- `common.py`
  - SBM 생성
  - Hypergraph SBM 생성/통계/라플라시안 유틸 (기존 `hypergraph_sbm.py` 통합)
  - Hayashi et al. EDVW random walk 기반 transition matrix와 Chung directed Laplacian
  - randomized / non-random spectral 알고리즘
  - 오차 metric 계산
  - 상세 timing breakdown 집계
  - Section 7.1 실험 실행/저장 유틸
  - Section 7.1 timing 시각화 유틸

## 파일 정리 메모

- 기존 `src/hypergraph_sbm.py`는 더 이상 사용하지 않으며, 관련 함수는 모두 `src/common.py`로 통합되었습니다.
- 실험 스크립트는 `from src.common import ...` 경로를 사용합니다.

## Hayashi et al. EDVW Laplacian 함수

논문 "Hypergraph Random Walks, Laplacians, and Clustering"의 Eq. (3), Eq. (5), Eq. (6), Algorithm 1에서 쓰는 행렬 구성을 `src/common.py`에 추가했습니다.

주요 함수:
- `edvw_transition_matrix`: hyperedge list와 optional EDVW weight에서 `P = D_V^{-1} W D_E^{-1} R` 생성
- `edvw_transition_matrix_from_incidence`: `|E| x |V|` EDVW incidence matrix `R`에서 transition matrix 생성
- `stationary_distribution_power`: `pi^T P = pi^T`를 만족하는 stationary distribution 계산
- `chung_directed_laplacian`: Chung directed combinatorial/normalized Laplacian 생성
- `chung_directed_similarity`: RDC-Spec에서 쓰는 `T = I - L` 행렬 생성
- `hayashi_edvw_laplacian`: hyperedge list에서 Hayashi EDVW Laplacian까지 한 번에 생성
- `hayashi_edvw_similarity_matrix`: hyperedge list에서 Algorithm 1의 spectral embedding용 `T` 행렬까지 한 번에 생성

## Section 7.2 / 8.x 연관 함수

아래는 `experiments`의 주요 실험이 `src/common.py`에서 공통으로 가져다 쓰는 함수 정리입니다.

### Section 7.2 (Models 1-6)

대상 스크립트:
- `experiments/reference_1_section7_2/sec72_models123_live.py`
- `experiments/reference_1_section7_2/sec72_models456_live.py`

주요 공통 함수:
- 진행/메타: `LiveProgress`, `METHODS`, `METHOD_COLORS`
- 알고리즘 실행: `run_non_random`, `run_random_projection`, `run_random_sampling`
- 성능평가: `evaluate_metrics`
- timing 병합/요약: `attach_timing_breakdown`, `extract_timing_breakdown`, `summarize_metrics`, `summarize_timing_breakdown`

참고:
- Model별 `B`, `Theta`, 라벨 생성 로직은 각 `sec72_*.py` 내부에 구현되어 있고,
  공통 spectral pipeline과 metric/timing 유틸만 `common.py`를 사용합니다.

### Section 8.1 (European Email Accuracy)

대상 스크립트:
- `experiments/reference_1_section8_1/exp8_1_email_eu_core_live.py`

주요 공통 함수:
- 데이터 로드/전처리 보조: `load_undirected_edgelist_csr`, `upper_triangle_edges`
- 고유벡터 계산: `eigvecs_eigsh_sparse`, `eigvecs_random_projection_sparse`, `eigvecs_random_sampling_sparse`
- 클러스터링/정렬: `kmeans_on_rows`, `align_labels_weighted_hungarian`
- 일치도 분석: `pairwise_ari`
- 진행 표시: `LiveProgress`

### Section 8.2 (Table 4 Runtime)

대상 스크립트:
- `experiments/reference_1_section8_2/exp8_2_live.py`

주요 공통 함수:
- 대규모 edge list 로더: `load_large_integer_edgelist_csr`
- Table 4 벤치마크: `benchmark_table4_methods_sparse`
- 요약/리포트 생성: `summarize_table4_median_times`, `format_table4_markdown`
- 시각화: `plot_table4_median_bars`, `plot_table4_runtime_boxplots`
- 진행 표시: `LiveProgress`

## Section 7.1 에 추가한 함수들

### 실험 설정/메타데이터

- `Exp1Config`
  - Experiment 1의 `n` 변화 실험 설정을 담는 dataclass입니다.

- `Exp2Config`
  - Experiment 2의 `alpha_n` 변화 실험 설정을 담는 dataclass입니다.

- `Exp3Config`
  - Experiment 3의 `K` 변화 실험 설정을 담는 dataclass입니다.

- `Exp4Config`
  - Experiment 4의 `n` 변화와 `alpha_n = 2 / sqrt(n)` 설정을 담는 dataclass입니다.

- `SavedExperimentOutputs`
  - 실험 실행 후 저장된 CSV/PNG 경로와 row 수를 묶어 반환하는 dataclass입니다.

- `TimingBreakdownResult`
  - 19개 timing-breakdown plot 렌더링 결과와 요약 테이블을 담는 dataclass입니다.

- `RuntimeCompositionResult`
  - 실제 시간 기준 stacked runtime composition figure 결과를 담는 dataclass입니다.

### 기본 설정 생성

- `default_exp1_config()`
  - Experiment 1 기본 설정을 반환합니다.

- `default_exp2_config()`
  - Experiment 2 기본 설정을 반환합니다.

- `default_exp3_config()`
  - Experiment 3 기본 설정을 반환합니다.

- `default_exp4_config()`
  - Experiment 4 기본 설정을 반환합니다.

- `default_output_dir(exp_key)`
  - `exp1`~`exp4`별 기본 결과 저장 디렉터리를 반환합니다.

- `parse_int_values(value_text)`
  - 쉼표로 구분된 정수 문자열을 리스트로 바꿉니다.

- `parse_float_values(value_text)`
  - 쉼표로 구분된 실수 문자열을 리스트로 바꿉니다.

### Section 7.1 실험 실행

- `_run_method_job(...)`
  - 한 반복 안에서 한 방법(`Non-random`, `Random Sampling`, `Random Projection`)의 실행, metric 계산, timing record 병합을 담당하는 내부 함수입니다.

- `run_experiment1(cfg, show_progress=True, theta_mode="exact", detailed_timing=False)`
  - Experiment 1 전체 반복을 실행하고 raw DataFrame을 반환합니다.

- `run_experiment2(cfg, show_progress=True, theta_mode="exact", detailed_timing=False)`
  - Experiment 2 전체 반복을 실행하고 raw DataFrame을 반환합니다.

- `run_experiment3(cfg, show_progress=True, theta_mode="hungarian", detailed_timing=False)`
  - Experiment 3 전체 반복을 실행하고 raw DataFrame을 반환합니다.

- `run_experiment4(cfg, show_progress=True, theta_mode="exact", detailed_timing=False)`
  - Experiment 4 전체 반복을 실행하고 raw DataFrame을 반환합니다.

### Section 7.1 요약/저장

- `summarize_experiment1(df_raw)`
  - Experiment 1 raw 결과를 `n` 기준으로 요약합니다.

- `summarize_experiment2(df_raw)`
  - Experiment 2 raw 결과를 `alpha_n` 기준으로 요약합니다.

- `summarize_experiment3(df_raw)`
  - Experiment 3 raw 결과를 `K` 기준으로 요약합니다.

- `summarize_experiment4(df_raw)`
  - Experiment 4 raw 결과를 `n` 기준으로 요약하고, `alpha_n_mean`도 함께 붙입니다.

- `save_experiment_outputs(exp_key, df_raw, df_summary, outdir=None, detailed_timing=False, plot_basics=True)`
  - raw CSV, summary CSV, timing breakdown CSV, 기본 metric/runtime PNG를 저장합니다.

- `run_and_save_experiment1(...)`
  - Experiment 1을 실행하고 저장까지 한 번에 처리합니다.

- `run_and_save_experiment2(...)`
  - Experiment 2를 실행하고 저장까지 한 번에 처리합니다.

- `run_and_save_experiment3(...)`
  - Experiment 3을 실행하고 저장까지 한 번에 처리합니다.

- `run_and_save_experiment4(...)`
  - Experiment 4를 실행하고 저장까지 한 번에 처리합니다.

### timing summary 로드/축 범위 계산

- `find_latest_summary(summary_filename, search_root=None)`
  - 주어진 summary CSV 이름과 일치하는 최신 파일을 찾습니다.

- `resolve_summary_path(exp_key, summary_path=None, search_root=None)`
  - 실험 키에 대응하는 timing summary CSV 경로를 확정합니다.

- `load_summary_frame(exp_key, summary_path=None, search_root=None)`
  - timing summary CSV를 읽어서 `(path, DataFrame)` 형태로 반환합니다.

- `build_timing_table(df, base_cols, metrics)`
  - timing summary DataFrame에서 필요한 컬럼만 뽑아 보기 좋은 표로 정리합니다.

- `compute_global_metric_limits(search_root=None, summary_paths=None)`
  - 모든 실험의 timing metric에 대해 공통 y축 범위를 계산합니다.

### timing-breakdown plot 생성

- `timing_method_slug(method_name)`
  - 방법 이름을 파일명에 쓰기 쉬운 slug로 바꿉니다.

- `timing_metric_title(method_name, metric_name)`
  - timing metric plot 제목 문자열을 만듭니다.

- `_plot_single_timing_metric(...)`
  - 한 개 timing metric line plot을 그리는 내부 함수입니다.

- `render_timing_breakdown_suite(exp_key, summary_path=None, output_dir=None, global_limits=None, search_root=None, save=True, show_plots=True)`
  - 전체 실행시간 3개, `Non-random` 2개, `Random Sampling` 6개, `Random Projection` 8개를 포함한 총 19개 timing-breakdown plot을 생성합니다.

### runtime composition plot 생성

- `numeric_series(df, column_name)`
  - 숫자 컬럼을 안전하게 float series로 변환합니다.

- `format_x_labels(values, x_col)`
  - `n`, `alpha_n`, `K` 축 라벨을 보기 좋게 문자열로 변환합니다.

- `build_method_runtime_table(df, method_name, x_col)`
  - 방법별 세부 단계 시간을 실제 시간과 퍼센트가 함께 있는 표로 정리합니다.

- `_format_percentage(pct)`
  - 퍼센트 라벨 문자열 포맷을 정합니다.

- `_use_inside_label(height, pct, shared_ymax)`
  - 퍼센트 라벨을 막대 내부에 넣을지 외부에 뺄지 결정합니다.

- `plot_total_runtime_comparison(ax, df, x_col, x_label)`
  - 세 방법의 전체 실행시간 비교 line plot을 그립니다.

- `_count_outside_labels(table, component_cols, shared_ymax)`
  - 바깥으로 빠질 퍼센트 라벨 개수를 세어 필요한 여백을 계산합니다.

- `annotate_stack_percentages(ax, table, component_cols, x_pos, shared_ymax)`
  - stacked bar 위에 퍼센트 라벨을 배치합니다.
  - 큰 조각은 내부에, 작은 조각은 바깥쪽 호출선 라벨로 배치해서 겹침을 줄입니다.

- `plot_method_runtime_stack(ax, table, method_name, x_col, x_label, shared_ymax)`
  - 한 방법의 실제 시간 기준 stacked runtime bar를 그립니다.

- `render_runtime_composition(exp_key, summary_path=None, output_path=None, search_root=None, save=True, show_plot=True)`
  - 맨 위 총 실행시간 비교 + 아래 3개 방법별 stacked runtime composition figure를 생성합니다.

- `render_all_section7_visualizations(exp_key, summary_path=None, breakdown_output_dir=None, composition_output_path=None, global_limits=None, search_root=None, save=True, show_plots=True)`
  - timing-breakdown suite와 runtime composition figure를 한 번에 생성합니다.

## 현재 Section 7.1 노트북 사용 흐름

`experiments/reference_1_section7_1/exp1_live.ipynb` 부터 `exp4_live.ipynb` 까지는 모두 `src/common.py`만 import 합니다.

일반적인 순서는 다음과 같습니다.

1. `default_exp*_config()`로 기본 실험 설정 생성
2. `run_and_save_experiment*()`로 실험 실행 및 CSV 저장
3. `compute_global_metric_limits()`로 공통 y축 범위 계산
4. `render_all_section7_visualizations()`로 19개 timing plot과 runtime composition figure 생성
