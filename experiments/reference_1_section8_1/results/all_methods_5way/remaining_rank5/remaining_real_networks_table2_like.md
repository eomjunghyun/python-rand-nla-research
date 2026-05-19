Table 2 (b-d): Section 8.1 remaining real network accuracy experiments.

### Political blog network

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.435(0.004) | 0.042(0.006) | 0.002(0.001) |
| Random Sampling (p= 0.7) | 0.501(0.008) | 0.125(0.006) | 0.037(0.004) |
| Random Sampling (p= 0.8) | 0.499(0.007) | 0.125(0.005) | 0.037(0.003) |
| CountSketch | 0.434(0.007) | 0.041(0.007) | 0.002(0.001) |
| SIGN Bidirectional | 0.431(0.010) | 0.039(0.008) | 0.002(0.001) |
| Non-Random | 0.502(0.000) | 0.127(0.000) | 0.038(0.000) |

### Statisticians citation network (No true labels)

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.898(0.124) | 0.752(0.267) | 0.794(0.267) |
| Random Sampling (p= 0.7) | 0.854(0.116) | 0.634(0.218) | 0.732(0.232) |
| Random Sampling (p= 0.8) | 0.835(0.132) | 0.602(0.261) | 0.685(0.266) |
| CountSketch | 0.921(0.102) | 0.797(0.225) | 0.848(0.220) |
| SIGN Bidirectional | 0.805(0.131) | 0.548(0.263) | 0.610(0.264) |

### Statisticians coauthor network (No true labels)

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.991(0.023) | 0.978(0.051) | 0.985(0.036) |
| Random Sampling (p= 0.7) | 0.816(0.245) | 0.766(0.151) | 0.810(0.172) |
| Random Sampling (p= 0.8) | 0.781(0.298) | 0.788(0.209) | 0.809(0.226) |
| CountSketch | 0.992(0.023) | 0.982(0.049) | 0.988(0.035) |
| SIGN Bidirectional | 0.994(0.012) | 0.982(0.034) | 0.989(0.020) |

Note: Values are mean(std) over 20 replications.
For the two statisticians networks, scores are relative to non-random spectral clustering.
