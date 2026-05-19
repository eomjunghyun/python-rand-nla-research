Table 4-like median time (seconds) over replications, with Wang 2025 SIGN Bidirectional added.

| Networks | Random projection | CountSketch | SIGN Bidirectional | Random sampling | Non-random | SIGN / RP | SIGN / Non-random |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBLP | 0.618 | 0.635 | 1.253 | 3.119(0.391) | 0.398 | 2.03x | 3.15x |
| Youtube | 5.517 | 5.498 | 10.265 | 11.660(1.819) | 1.446 | 1.86x | 7.10x |
| Internet | 4.172 | 4.121 | 9.447 | 12.229(1.684) | 1.822 | 2.26x | 5.19x |

Note: Random Sampling values outside parentheses include sampling time; values inside parentheses exclude sampling time.