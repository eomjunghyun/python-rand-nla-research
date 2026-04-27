# Reference 1 Section 8.2 Table 4 스타일 효율성 실험

이 폴더는 Reference 1 논문의 Section 8.2와 Table 4에 맞춘 대규모 real network timing benchmark를 재현한다.

## 대상 네트워크

- DBLP collaboration network
- Youtube social network
- Internet topology graph

## 비교 방법

- Random Projection
- Random Sampling
- `partial_eigen`: `scipy.sparse.linalg.eigsh`를 사용하는 Python proxy

## 논문과 맞춘 측정 기준

- Table 4의 기준에 맞춰 eigenvector computation 단계의 시간만 주요 timing으로 본다.
- `k-means`, ARI, clustering post-processing 시간은 제외한다.
- Random Sampling은 sampling 포함 시간과 sampling 제외 시간을 모두 기록한다. 논문 표의 괄호 표기와 맞추기 위한 구성이다.

## 필요한 데이터 파일

아래 파일들이 로컬 `data/` 폴더에 있다고 가정한다.

- `data/com-dblp.ungraph.txt`
- `data/com-youtube.ungraph.txt`
- `data/as-skitter.txt`

## 실행 방법

```bash
python experiments/reference_1_section8_2/exp8_2_live.py \
  --dblp-edgelist data/com-dblp.ungraph.txt \
  --youtube-edgelist data/com-youtube.ungraph.txt \
  --internet-edgelist data/as-skitter.txt \
  --reps 20 \
  --seed 2026 \
  --q 2 \
  --r 10 \
  --p 0.7
```

## 출력 파일

- `table4_time_raw.csv`
- `table4_like_median_time.csv`
- `table4_like_median_time.md`
- `table4_time_breakdown.csv`
- timing 관련 figure와 meta 파일

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. 데이터셋 이름, 알고리즘 이름, 논문 표기와 직접 대응되는 용어는 영어를 섞어 쓴다.
