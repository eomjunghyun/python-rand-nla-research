# Reference 1 Section 8.2 Table 4 스타일 효율성 실험

이 폴더는 Reference 1 논문의 Section 8.2와 Table 4에 맞춘 대규모 real network timing benchmark를 재현한다.

## 대상 네트워크

- DBLP collaboration network
- Youtube social network
- Internet topology graph

## 비교 방법

- Random Projection
- CountSketch
- Random Sampling
- Non-random: `scipy.sparse.linalg.eigsh`를 사용하는 Python proxy
- `SIGN Bidirectional`: Wang et al. (2025)의 generalized Nystrom with subspace iteration을 sparse graph timing benchmark에 맞춘 양방향 변형

## 논문과 맞춘 측정 기준

- Table 4의 기준에 맞춰 eigenvector computation 단계의 시간만 주요 timing으로 본다.
- `k-means`, ARI, clustering post-processing 시간은 제외한다.
- Random Sampling은 sampling 포함 시간과 sampling 제외 시간을 모두 기록한다. 논문 표의 괄호 표기와 맞추기 위한 구성이다.
- `SIGN Bidirectional`은 baseline raw timing을 만든 뒤 같은 데이터셋/반복 수로 별도 실행하고, 최종 Table 4-style CSV/plot/report에 병합한다.

## 필요한 데이터 파일

아래 파일들이 로컬 `data/` 폴더에 있다고 가정한다.

- `data/com-dblp.ungraph.txt` 또는 `data/com-dblp.ungraph.txt.gz`
- `data/com-youtube.ungraph.txt` 또는 `data/com-youtube.ungraph.txt.gz`
- `data/as-skitter.txt` 또는 `data/as-skitter.txt.gz`

## 실행 방법

```bash
python experiments/reference_1_section8_2/exp8_2_live.py \
  --dblp-edgelist data/com-dblp.ungraph.txt.gz \
  --youtube-edgelist data/com-youtube.ungraph.txt.gz \
  --internet-edgelist data/as-skitter.txt.gz \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p 0.7
```

Wang et al. (2025) SIGN 추가 비교는 아래 스크립트로 실행한다.

```bash
python experiments/reference_1_section8_2/run_sign_section8_2.py \
  --baseline-raw-csv experiments/reference_1_section8_2/results/exp8_2_table4_paper_aligned/table4_time_raw.csv \
  --dblp-edgelist data/com-dblp.ungraph.txt.gz \
  --youtube-edgelist data/com-youtube.ungraph.txt.gz \
  --internet-edgelist data/as-skitter.txt.gz \
  --reps 20 \
  --seed 2026
```

## 출력 파일

- `table4_time_raw.csv`
- `table4_like_median_time.csv`
- `table4_like_median_time.md`
- timing 관련 figure와 meta 파일
- `results/sign_section8_2_wang2025/table4_with_sign_time_raw.csv`
- `results/sign_section8_2_wang2025/table4_with_sign_median_time.csv`
- `results/sign_section8_2_wang2025/table4_with_sign_median_time.md`
- `results/sign_section8_2_wang2025/sign_step_time_summary.csv`
- `results/sign_section8_2_wang2025/sign_section8_2_report.md`

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. 데이터셋 이름, 알고리즘 이름, 논문 표기와 직접 대응되는 용어는 영어를 섞어 쓴다.
