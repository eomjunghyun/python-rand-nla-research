table2d: The clustering performance on the Statisticians citation network (No true labels).

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.856(0.141) | 0.665(0.303) | 0.711(0.292) |
| Random Sampling (p= 0.7) | 0.884(0.100) | 0.692(0.196) | 0.791(0.203) |
| Random Sampling (p= 0.8) | 0.840(0.122) | 0.613(0.238) | 0.700(0.247) |
| CountSketch | 0.910(0.124) | 0.783(0.264) | 0.823(0.259) |
| SIGN Bidirectional | 0.680(0.164) | 0.435(0.177) | 0.532(0.194) |

Note: Values are mean(std) over 20 MATLAB replications.
For this dataset, scores are relative to non-random spectral clustering.
