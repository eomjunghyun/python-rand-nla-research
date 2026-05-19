Table 2 (b-d): Section 8.1 remaining real network accuracy experiments.

### Political blog network

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.576(0.000) | 0.178(0.000) | 0.080(0.000) |
| Random Sampling (p= 0.7) | 0.572(0.009) | 0.178(0.007) | 0.077(0.006) |
| Random Sampling (p= 0.8) | 0.572(0.004) | 0.178(0.005) | 0.077(0.003) |
| CountSketch | 0.576(0.000) | 0.178(0.000) | 0.080(0.000) |
| SIGN Bidirectional | 0.576(0.000) | 0.178(0.000) | 0.080(0.000) |
| Non-Random | 0.576(0.000) | 0.178(0.000) | 0.080(0.000) |

### Statisticians citation network (No true labels)

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.984(0.005) | 0.936(0.021) | 0.973(0.011) |
| Random Sampling (p= 0.7) | 0.937(0.010) | 0.797(0.024) | 0.899(0.016) |
| Random Sampling (p= 0.8) | 0.950(0.011) | 0.830(0.025) | 0.917(0.014) |
| CountSketch | 0.984(0.005) | 0.935(0.018) | 0.972(0.009) |
| SIGN Bidirectional | 0.971(0.014) | 0.896(0.038) | 0.953(0.021) |

### Statisticians coauthor network (No true labels)

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 1.000(0.000) | 1.000(0.000) | 1.000(0.000) |
| Random Sampling (p= 0.7) | 0.873(0.205) | 0.786(0.190) | 0.835(0.198) |
| Random Sampling (p= 0.8) | 0.969(0.023) | 0.914(0.055) | 0.948(0.038) |
| CountSketch | 0.998(0.007) | 0.996(0.019) | 0.997(0.011) |
| SIGN Bidirectional | 0.998(0.007) | 0.994(0.020) | 0.997(0.012) |

Note: Values are mean(std) over 20 replications.
For the two statisticians networks, scores are relative to non-random spectral clustering.
