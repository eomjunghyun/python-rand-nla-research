# Reference 2 - Experiment 3 (Figure 6~12)

논문 **"A Stochastic Block Hypergraph model"**의 실험 3을 재현합니다.

재현 대상:
- Figure 6: bipartite hypergraph visualization (majority, q 변화)
- Figure 7: strategy별 G 분포 P(G) vs q/p
- Figure 8: node traversal order 영향
- Figure 9: weighted 전략에서 mean G vs q/p (N/E 변화)
- Figure 10: q/p=1에서 scaling (mean G vs 1/sqrt(N/E))
- Figure 11: strategy별 mean G 비교
- Figure 12: strategy별 Delta vs q/p (N/E 변화)

## Run (paper-scale)

```bash
python3 experiments/reference_2_exp3/run_figure6_7_8_9_10_11_12.py \
  --seed 2026 \
  --fig7-realizations 100 \
  --fig8-realizations 100 \
  --fig9-12-realizations 100
```

## Run (quick smoke test)

```bash
python3 experiments/reference_2_exp3/run_figure6_7_8_9_10_11_12.py --quick
```

## Outputs

- `figures/figure6_bipartite_majority.png`
- `figures/figure7_gini_distribution.png`
- `figures/figure8_order_effect.png`
- `figures/figure9_mean_g_weighted.png`
- `figures/figure10_scaling.png`
- `figures/figure11_strategy_comparison.png`
- `figures/figure12_delta.png`
- `results/composition_summary.json`
- `results/exp3_report.md`

## Notes

- Figure 6의 force-directed layout은 Python 환경에서 D3를 직접 사용하지 않고 `networkx.spring_layout`으로 대체했습니다.
- majority 전략 동률은 **가장 작은 community index**를 선택하도록 구현되어 있습니다.
- `results/exp3_report.md`에 아래 항목을 자동 요약합니다:
  - low q/p에서 purity 증가 여부
  - q/p→1에서 G 감소 여부
  - majority/weighted의 dominant community 강화 경향
  - max 전략의 interior peak(q/p≈0.2) 탐지
  - 논문과 차이 원인(시드, tie-breaking, layout 근사, histogram 설정)
