# Data

## Reference 1 Section 8.1

현재 포함된 원자료/전처리 데이터:

- `email-Eu-core.txt`
- `email-Eu-core-department-labels.txt`
- `reference_1_section8_1/raw/polblogs.zip`
- `reference_1_section8_1/raw/SCC2016-with-abs.zip`
- `reference_1_section8_1/processed/political_blog_*`
- `reference_1_section8_1/processed/statisticians_coauthor_*`
- `reference_1_section8_1/processed/statisticians_citation_*`

Political blog 데이터는 Newman network data의 `polblogs.gml`을 무방향화한 뒤 giant component를 사용한다.
Statisticians 데이터는 Ji & Jin (2016)의 `SCC2016-with-abs` 원자료에서 author-level coauthor/citation 네트워크를 구성한 뒤 giant component를 사용한다.
