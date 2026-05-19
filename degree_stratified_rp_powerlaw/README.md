# Degree-Stratified RP Power-Law 그래프 실험

이 폴더는 기존 HSBM/하이퍼그래프 실험과 분리해서 만든 일반 그래프용 독립 실험 폴더입니다.

목표는 power-law degree 분포가 뚜렷한 실데이터 그래프에서 다음 방법들을 비교하는 것입니다.

- `General eigensolver`: dense full symmetric eigendecomposition, `numpy.linalg.eigh`
- `Gaussian RP`: 표준 Gaussian randomized projection
- `Random sampling`: Bernoulli edge sampling 후 rescaling
- `Degree-stratified RP`: degree bucket별 Gaussian sketch를 만들고 bucket mass의 제곱근 비율로 sketch dimension을 배분하는 방법

기본 실행 예시는 다음과 같습니다.

```powershell
cd C:\Users\WWindows10\Documents\github_project\python-rand-nla-research\degree_stratified_rp_powerlaw
py .\degree_stratified_rp_powerlaw.py `
  --edgelist ..\data\com-dblp.ungraph.txt `
  --dataset-name com-dblp `
  --max-n 1500 `
  --k 8 `
  --r 20 `
  --q 2 `
  --reps 5 `
  --outdir results\com_dblp_demo
```

주요 결과 파일은 `results/` 아래에 저장됩니다.

- `degree_stratified_rp_raw.csv`: 반복별 원자료
- `degree_stratified_rp_summary.csv`: 방법별 평균/표준편차 요약
- `degree_stratified_bucket_allocations.csv`: Degree-stratified RP의 bucket별 sketch dimension 배분
- `degree_stratified_clustering_stability.csv`: 반복 간 clustering 안정성
- `degree_stratified_rp_meta.json`: 실행 설정과 그래프 degree 진단 정보

현재 smoke test 결과 요약은 [결과보고서.md](./결과보고서.md)에 정리되어 있습니다.

## LastFM Asia alpha/tau/r sweep

LastFM Asia 실험은 별도 스크립트로 실행합니다.

데이터셋 정보:

- SNAP LastFM Asia Social Network
- Undirected graph
- 7,624 nodes
- 27,806 edges
- Multi-class node labels
- 공식 페이지: https://snap.stanford.edu/data/feather-lastfm-social.html

데이터 다운로드:

```powershell
cd C:\Users\WWindows10\Documents\github_project\python-rand-nla-research
New-Item -ItemType Directory -Force -Path data\lastfm_asia | Out-Null
Invoke-WebRequest `
  -Uri https://snap.stanford.edu/data/lastfm_asia.zip `
  -OutFile data\lastfm_asia\lastfm_asia.zip
```

alpha/tau/r sweep 실행:

```powershell
cd C:\Users\WWindows10\Documents\github_project\python-rand-nla-research\degree_stratified_rp_powerlaw

py .\lastfm_operator_r_sweep.py `
  --dataset-path ..\data\lastfm_asia\lastfm_asia.zip `
  --dataset-name lastfm-asia `
  --alpha-values 0,0.25,0.5 `
  --tau-values 0,mean `
  --r-values 5,10,20,40 `
  --q 1 `
  --reps 5 `
  --outdir results\lastfm_alpha_tau_r_sweep
```

이 스윕은 다음 operator를 사용합니다.

```text
S_{alpha,tau} = D_tau^{-alpha} A D_tau^{-alpha}
D_tau = D + tau I
```

`tau=mean`은 `tau`를 평균 degree로 설정합니다. `k`를 명시하지 않으면 LastFM label class 수를 자동으로 사용합니다.

실행된 LastFM sweep 결과는 [LastFM_결과보고서.md](./LastFM_결과보고서.md)에 정리되어 있습니다.

## LastFM A+B degree-stratum 실험

Gaussian RP가 약하고 Degree-stratified RP가 좋아지는 구간을 확인하기 위한 focused experiment입니다.

```powershell
cd C:\Users\WWindows10\Documents\github_project\python-rand-nla-research\degree_stratified_rp_powerlaw

py .\lastfm_degree_stratum_experiment.py `
  --dataset-path ..\data\lastfm_asia\lastfm_asia.zip `
  --dataset-name lastfm-asia `
  --r-values 0,2,5,10,20,40 `
  --alpha 0.5 `
  --tau 0 `
  --q 1 `
  --reps 10 `
  --outdir results\lastfm_degree_stratum_alpha05_tau0
```

결과는 [LastFM_A_B_결과보고서.md](./LastFM_A_B_결과보고서.md)에 정리되어 있습니다.
