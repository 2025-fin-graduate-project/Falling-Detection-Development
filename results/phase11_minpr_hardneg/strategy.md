# Phase 11 Strategy

**핵심 가설**: P9O-v01의 병목은 비낙상 영상 FP. val_video_min_pr로 올바른 epoch을 선택하고, val FP 영상을 hard-negative로 오버샘플해 NFall 방어력을 높인다.

**이전 Phase 실패 원인**: Phase 9/10에서 val_loss 기준 early stopping → MinPR 최적 epoch을 놓침. 단순 postprocess sweep으로는 FP 영상의 sustained high-score 패턴을 해결하지 못함.

**주요 변경점**:
- `--checkpoint-monitor val_video_min_pr`: 매 epoch val 전체를 추론해 best MinPR 가중치 보존
- `--hard-negative-video-ids`: val FP 영상의 negative 윈도우 stride를 1로 줄여 2배 노출
- `--min-consecutive-values 1,2,3,4,5,7,9`: 더 세밀한 postprocess sweep

**후보 실험**:
| ID | 체크포인팅 | 변경점 | 결과 |
|----|-----------|--------|------|
| v01 | val_video_min_pr | 체크포인팅만 | test MinPR=0.8694 (val↑ test↓, thr=0.475 과적합) |
| v02 | val_video_min_pr | +hard-neg stride=1 | test MinPR=0.8939 (FP 32→26, thr=0.475 여전) |
| v03 | val_video_min_pr | v02 + consecutive 1~9 | test MinPR=0.8612 (thr=0.525/mc=4 → FN=27↑, 역효과) |
| v04 | **val_loss** | hard-neg stride=1 (hard-neg 단독 효과) | test MinPR=0.8880 (thr=0.550↑, FP=23, FN=28, mc=1) |
| v05 | val_loss | hard-neg + focal α=0.10 | test MinPR=0.9061 (thr=0.675↑, FP=23, FN=22↓, 큰 개선) |
| v06 | val_loss | hard-neg + 전체 neg stride=1 | - |
| v07 | val_video_min_pr | hard-neg + **thr_floor=0.50** | - |
| v08 | val_loss | hard-neg + kp12 | test MinPR=0.8735 (thr=0.425↓, FP=31↑ — α=0.25 없이 역효과) |
| v09 | val_loss | hard-neg + GRU(256,128) | - |

**중간 분석 (v01-v02)**:
- val_video_min_pr 체크포인팅: val에서 thr=0.475 선택 → test FP 증가 (val/test 분포 갭)
- hard-negative는 효과 있음: test FP 32→26
- v04부터 val_loss + hard-neg 조합으로 효과 분리

**베이스라인**: P9O-v01 test MinPR=0.9187 (NFallP=0.9187, NFallR=0.9224)

**종료 기준**: test MinPR ≥ 0.92
