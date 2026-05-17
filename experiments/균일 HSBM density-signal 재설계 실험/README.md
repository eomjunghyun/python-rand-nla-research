# 균일 HSBM density-signal 재설계 실험

이 폴더는 기존 `experiments/균일 HSBM 실험` 결과를 덮어쓰지 않고, density와 signal 효과를 분리해 다시 실행하는 실험을 담는다.

핵심 목적은 다음과 같다.

- `rho_n`을 올릴 때 정확도만 즉시 포화되지 않도록 signal gap을 함께 조절한다.
- `K`를 올릴 때 3-uniform HSBM의 within 후보 비율 감소를 `rho_n` 보정으로 완화한다.
- background density를 키우되 signal gap을 충분히 유지하는 추가 density sweep으로, 정확도와 계산량이 동시에 관측되는 regime을 만든다.
- randomized eigensolver의 계산 이점을 보기 위해 oversampling과 power iteration을 speed regime으로 낮춘다.
- 대표 인스턴스의 spectral gap과 randomized parameter tradeoff를 따로 진단해, speedup이 정확도를 잃은 결과인지 확인한다.
- 결과 CSV, 설정 JSON, 요약 그림, 보고서를 모두 이 폴더 아래 `results/`와 `결과보고서.md`에 저장한다.

실행:

```bash
python "experiments/균일 HSBM density-signal 재설계 실험/run_redesigned_uniform_hsbm.py" all
```

개별 실행:

```bash
python "experiments/균일 HSBM density-signal 재설계 실험/run_redesigned_uniform_hsbm.py" rho
python "experiments/균일 HSBM density-signal 재설계 실험/run_redesigned_uniform_hsbm.py" K
python "experiments/균일 HSBM density-signal 재설계 실험/run_redesigned_uniform_hsbm.py" n
```

보고서만 다시 생성:

```bash
python "experiments/균일 HSBM density-signal 재설계 실험/run_redesigned_uniform_hsbm.py" report
```

진단 표만 다시 계산하고 보고서를 갱신:

```bash
python "experiments/균일 HSBM density-signal 재설계 실험/run_redesigned_uniform_hsbm.py" diagnostics
```
