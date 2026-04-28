# Reference 1 Section 8.1 정확도 실험

이 폴더는 Reference 1 논문의 Section 8.1 accuracy experiment를 재현한다.

## 비교 방법

- Random Projection (`q=2`, `r=10`)
- Random Sampling (`p=0.7`, `p=0.8`)
- Non-random spectral clustering

## 현재 구현 상태

- `exp8_1_email_eu_core_live.py`: European Email network
- `exp8_1_email_eu_core_rank30_live.py`: European Email network, embedding rank 30
- `exp8_1_political_blog_live.ipynb`: Political blog network
- `exp8_1_statisticians_coauthor_live.ipynb`: Statisticians coauthor network
- `exp8_1_statisticians_citation_live.ipynb`: Statisticians citation network
- `exp8_1_political_blog_rank5_live.ipynb`: Political blog network, embedding rank 5
- `exp8_1_statisticians_coauthor_rank5_live.ipynb`: Statisticians coauthor network, embedding rank 5
- `exp8_1_statisticians_citation_rank5_live.ipynb`: Statisticians citation network, embedding rank 5
- `exp8_1_remaining_real_networks_live.py`: 위 세 노트북이 공통으로 사용하는 실행 엔진
- `prepare_exp8_1_real_data.py`: Political blog / statisticians 원자료 전처리

## 입력 데이터 형식

- edge list 텍스트 파일을 입력으로 사용한다.
- 각 줄은 `node_u node_v` 형식이다.
- label 파일은 각 줄이 `node_id class_id` 형식이다.
- 스크립트 내부에서 directed edge list를 undirected graph로 바꾸고 largest connected component를 사용한다.

## European Email 실행 방법

```bash
python experiments/reference_1_section8_1/exp8_1_email_eu_core_live.py \
  --edge-path data/email-Eu-core.txt \
  --label-path data/email-Eu-core-department-labels.txt \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p-values 0.7,0.8
```

## European Email rank 30 실행 방법

`target_rank=30`은 spectral embedding 차원만 바꾸며, 최종 KMeans 군집 수는 정답 class 수(`K=42`)를 그대로 사용한다.

```bash
python experiments/reference_1_section8_1/exp8_1_email_eu_core_rank30_live.py \
  --edge-path data/email-Eu-core.txt \
  --label-path data/email-Eu-core-department-labels.txt \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --target-rank 30 \
  --p-values 0.7,0.8
```

## 나머지 세 데이터셋 준비

원자료를 받은 뒤 다음 전처리를 실행한다.

```bash
python experiments/reference_1_section8_1/prepare_exp8_1_real_data.py
```

생성되는 processed 데이터는 다음 논문 Table 1 수치와 맞아야 한다.

| Dataset | Nodes | Edges | Target rank |
|---|---:|---:|---:|
| Political blog | 1222 | 16714 | 2 |
| Statisticians coauthor | 2263 | 4388 | 3 |
| Statisticians citation | 2654 | 20049 | 3 |

데이터셋별 실험은 각 노트북을 실행한다.

- `exp8_1_political_blog_live.ipynb`
- `exp8_1_statisticians_coauthor_live.ipynb`
- `exp8_1_statisticians_citation_live.ipynb`
- `exp8_1_political_blog_rank5_live.ipynb`
- `exp8_1_statisticians_coauthor_rank5_live.ipynb`
- `exp8_1_statisticians_citation_rank5_live.ipynb`

Rank 5 노트북은 최종 군집 수는 논문값 그대로 유지하고, spectral embedding에 사용하는 eigenvector 수만 5로 늘린다.

CLI로 한 번에 실행하려면 다음 명령을 사용할 수 있다.

```bash
python experiments/reference_1_section8_1/exp8_1_remaining_real_networks_live.py \
  --datasets political_blog,statisticians_coauthor,statisticians_citation \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p-values 0.7,0.8
```

## 출력 파일

종합 보고서:
- `section8_1_experiment_report.md`

European Email 기본 rank 결과:
- `results/exp8_1_email_eu_core_table2_like/email_eu_raw_per_rep.csv`
- `results/exp8_1_email_eu_core_table2_like/email_eu_summary_mean_std.csv`
- `results/exp8_1_email_eu_core_table2_like/email_eu_table2a_like.csv`
- `results/exp8_1_email_eu_core_table2_like/email_eu_table2a_like.md`
- `results/exp8_1_email_eu_core_table2_like/email_eu_pairwise_ari_raw.csv`
- `results/exp8_1_email_eu_core_table2_like/email_eu_pairwise_ari_mean_matrix.csv`
- `results/exp8_1_email_eu_core_table2_like/email_eu_pairwise_ari_heatmap.png`
- `results/exp8_1_email_eu_core_table2_like/email_eu_meta.json`

European Email rank 30 결과:
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_raw_per_rep.csv`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_summary_mean_std.csv`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_table2a_like.csv`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_table2a_like.md`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_pairwise_ari_raw.csv`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_pairwise_ari_mean_matrix.csv`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_pairwise_ari_heatmap.png`
- `results/exp8_1_email_eu_core_rank30_table2_like/email_eu_meta.json`

공통 파일명:
- `email_eu_raw_per_rep.csv`
- `email_eu_summary_mean_std.csv`
- `email_eu_table2a_like.csv`
- `email_eu_table2a_like.md`
- `email_eu_pairwise_ari_raw.csv`
- `email_eu_pairwise_ari_mean_matrix.csv`
- `email_eu_pairwise_ari_heatmap.png`
- `email_eu_meta.json`

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. 데이터셋 이름, 알고리즘 이름, 파일명은 영어 표기를 유지할 수 있다.
