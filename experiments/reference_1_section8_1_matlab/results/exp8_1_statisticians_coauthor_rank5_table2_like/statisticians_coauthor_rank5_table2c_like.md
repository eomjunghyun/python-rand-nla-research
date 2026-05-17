table2c: The clustering performance on the Statisticians coauthor network (No true labels).

| Methods | F 1 | NMI | ARI |
|---|---:|---:|---:|
| Random Projection | 0.847(0.211) | 0.741(0.309) | 0.735(0.331) |
| Random Sampling (p= 0.7) | 0.758(0.260) | 0.684(0.204) | 0.725(0.231) |
| Random Sampling (p= 0.8) | 0.837(0.223) | 0.761(0.215) | 0.787(0.241) |
| CountSketch | 0.876(0.173) | 0.776(0.275) | 0.779(0.296) |
| SIGN Bidirectional | 0.810(0.158) | 0.622(0.255) | 0.649(0.277) |

Note: Values are mean(std) over 20 MATLAB replications.
For this dataset, scores are relative to non-random spectral clustering.
