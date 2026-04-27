# Reference 1 Section 8.1 European Email 정확도 실험

이 폴더는 European Email 네트워크를 사용해 Reference 1 논문의 Section 8.1 accuracy experiment를 재현한다.

## 비교 방법

- Random Projection (`q=2`, `r=10`)
- Random Sampling (`p=0.7`, `p=0.8`)
- Non-random spectral clustering

## 입력 데이터 형식

- edge list 텍스트 파일을 입력으로 사용한다.
- 각 줄은 `node_u node_v` 형식이다.
- label 파일은 각 줄이 `node_id class_id` 형식이다.
- 스크립트 내부에서 directed edge list를 undirected graph로 바꾸고 largest connected component를 사용한다.

## 실행 방법

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

## 출력 파일

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
