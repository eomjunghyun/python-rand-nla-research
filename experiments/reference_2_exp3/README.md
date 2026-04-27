# Reference 2 Experiment 3: Figure 6~12

이 폴더는 논문 "A Stochastic Block Hypergraph model"의 Experiment 3을 재현한다.

## 재현 대상

- Figure 6: majority 조건과 `q` 변화에 따른 bipartite hypergraph visualization
- Figure 7: strategy별 `G` 분포 `P(G)`와 `q/p`의 관계
- Figure 8: node traversal order의 영향
- Figure 9: weighted strategy에서 `N/E` 변화에 따른 mean `G`와 `q/p`의 관계
- Figure 10: `q/p = 1`에서 scaling 관계, 즉 mean `G`와 `1/sqrt(N/E)`의 관계
- Figure 11: strategy별 mean `G` 비교
- Figure 12: strategy별 `Delta`와 `q/p`의 관계

## 논문 규모 실행

```bash
python experiments/reference_2_exp3/run_figure6_7_8_9_10_11_12.py \
  --seed 2026 \
  --fig7-realizations 100 \
  --fig8-realizations 100 \
  --fig9-12-realizations 100
```

## 빠른 점검 실행

```bash
python experiments/reference_2_exp3/run_figure6_7_8_9_10_11_12.py --quick
```

환경에 따라 `python` 대신 프로젝트에서 사용하는 Python 실행 파일을 지정하면 된다.

## 출력 파일

- `figures/figure6_bipartite_majority.png`
- `figures/figure7_gini_distribution.png`
- `figures/figure8_order_effect.png`
- `figures/figure9_mean_g_weighted.png`
- `figures/figure10_scaling.png`
- `figures/figure11_strategy_comparison.png`
- `figures/figure12_delta.png`
- `results/composition_summary.json`
- `results/exp3_report.md`

## 참고 사항

- Figure 6의 force-directed layout은 Python 환경에서 D3를 직접 사용하지 않고 `networkx.spring_layout`으로 대체한다.
- 논문과 완전히 같은 random layout을 보장하기보다는, 같은 생성 모델과 요약 통계를 확인하는 데 초점을 둔다.

## 작성 규칙

README와 실험 설명은 기본적으로 한글로 작성한다. Figure 번호, strategy 이름, 변수명은 영어 표기를 유지할 수 있다.
