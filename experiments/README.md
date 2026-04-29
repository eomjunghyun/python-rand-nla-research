# 수치 실험

- `메인 실험/`: 과거 비균일 HSBM의 `n` 변화, 큰 `n` 변화, `K` 변화 실험 기록. 현재 `src.common`에서는 비균일 HSBM 생성 함수가 제거되어 새 실험 기준 API는 아니다.
- `균일 HSBM 실험/`: planted `d`-uniform HSBM의 `n` 변화 실험
- `nonuniform_hsbm_scaling/`: 초기 비균일 HSBM 스케일링 작업 폴더. 현재는 legacy 기록으로만 둔다.
- `reference_1_section7_1/`: 논문 Section 7.1 Experiments 1-4 재현 노트북
- `reference_1_section7_2/`: 논문 Section 7.2 Models 1-6 재현 스크립트
- `reference_1_section8_1/`: European Email 네트워크 accuracy 실험 및 target rank 변형
- `reference_1_section8_2/`: DBLP, Youtube, Internet 대상 Table 4 스타일 효율성 benchmark

앞으로 새로 작성하는 실험 README, 마크다운 설명, 코드 주석은 기본적으로 한글로 작성한다. 함수명, 지표명, 알고리즘명처럼 영어가 자연스러운 용어는 그대로 섞어 쓴다.
