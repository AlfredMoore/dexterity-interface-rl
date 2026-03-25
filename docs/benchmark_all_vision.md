# Vision Benchmark Unified Report

- Dataset: `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807`
- Total merged runs: `16`
- Success: `15`
- Failed: `1`

## Sorting Rule
- Order by tool: `DA3 -> PromptDA -> SAM3 -> SAM2_w_prompt`.
- Within each tool: checkpoint size from small to large (`ckpt_size_mb` ascending).
- If checkpoint size is unavailable, fallback to model name and place after sized entries.

## TR;DL
DA3:
 * small: 50hz
 * base: 42hz
 * metric-large: 34hz
 * giant: 11hz

PromptDA:
 * small: 87hz
 * large: 18hz

SAM3:
 * normal: 11hz

SAM2_w_prompt
 * tiny: 89hz
 * small: 87hz
 * base_plus: 67hz
 * large: 40hz

## DA3
| id | model | status | ckpt_size_mb | mean_ms | mean_hz | min_ms | max_ms | std_ms | source_run | video | log |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| da3_small | depth-anything/DA3-SMALL | ok | 130.89 | 19.68 | 50.80 | 16.54 | 47.13 | 2.53 | `bench_vision_da3_base_small_20260318_230916` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_da3_base_small_20260318_230916/outputs/da3_small.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_da3_base_small_20260318_230916/logs/da3_small.log` |
| da3_base | depth-anything/DA3-BASE | ok | 516.43 | 23.50 | 42.60 | 21.78 | 55.98 | 1.88 | `bench_vision_da3_base_small_20260318_230916` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_da3_base_small_20260318_230916/outputs/da3_base.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_da3_base_small_20260318_230916/logs/da3_base.log` |
| da3_metric_large | depth-anything/DA3METRIC-LARGE | ok | 1274.81 | 28.70 | 34.80 | 27.34 | 108.61 | 3.33 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/da3_metric_large.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/da3_metric_large.log` |
| da3_nested_giant_large_1_1 | depth-anything/DA3NESTED-GIANT-LARGE-1.1 | ok | 6446.42 | 90.63 | 11.00 | 89.42 | 101.35 | 0.75 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/da3_nested_giant_large_1_1.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/da3_nested_giant_large_1_1.log` |

## PromptDA
| id | model | status | ckpt_size_mb | mean_ms | mean_hz | min_ms | max_ms | std_ms | source_run | video | log |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| promptda_s_transparent_vits | PromptDA-s-transparent.ckpt | ok | 95.79 | 11.47 | 87.20 | 11.01 | 85.84 | 3.08 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/promptda_s_transparent_vits.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/promptda_s_transparent_vits.log` |
| promptda_s_vits | PromptDA-s.ckpt | ok | 95.79 | 11.40 | 87.70 | 11.01 | 84.30 | 3.01 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/promptda_s_vits.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/promptda_s_vits.log` |
| promptda_l_vitl | PromptDA-l.ckpt | ok | 1297.35 | 54.04 | 18.50 | 52.62 | 55.72 | 0.62 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/promptda_l_vitl.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/promptda_l_vitl.log` |

## SAM3
| id | model | status | ckpt_size_mb | mean_ms | mean_hz | min_ms | max_ms | std_ms | source_run | video | log |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| sam3_efficient_efficientvit_b0 | efficient_sam3_efficientvit_s.pt | failed | 1618.17 |  |  |  |  |  | `bench_vision_20260318_220448` | `` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam3_efficient_efficientvit_b0.log` |
| sam3_litetext_s0 | efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt | ok | 2103.38 | 193.14 | 5.20 | 192.23 | 203.29 | 0.58 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam3_litetext_s0.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam3_litetext_s0.log` |
| sam3_litetext_s1 | efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt | ok | 2183.42 | 194.40 | 5.10 | 193.54 | 195.99 | 0.39 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam3_litetext_s1.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam3_litetext_s1.log` |
| sam3_litetext_mobileclip2_l | efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt | ok | 2413.40 | 193.99 | 5.20 | 193.08 | 195.69 | 0.40 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam3_litetext_mobileclip2_l.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam3_litetext_mobileclip2_l.log` |
| sam3_original | sam3.pt | ok | 3290.24 | 196.27 | 5.10 | 195.10 | 199.05 | 0.51 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam3_original.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam3_original.log` |

## SAM2_w_prompt
| id | model | status | ckpt_size_mb | mean_ms | mean_hz | min_ms | max_ms | std_ms | source_run | video | log |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| sam2_tiny | sam2.1_hiera_tiny.pt | ok | 148.78 | 11.15 | 89.70 | 10.11 | 18.51 | 0.99 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam2_tiny.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam2_tiny.log` |
| sam2_small | sam2.1_hiera_small.pt | ok | 175.87 | 11.42 | 87.50 | 10.60 | 16.43 | 1.04 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam2_small.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam2_small.log` |
| sam2_base_plus | sam2.1_hiera_base_plus.pt | ok | 308.62 | 14.76 | 67.80 | 14.26 | 20.57 | 0.77 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam2_base_plus.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam2_base_plus.log` |
| sam2_large | sam2.1_hiera_large.pt | ok | 856.48 | 25.53 | 39.20 | 25.28 | 27.92 | 0.18 | `bench_vision_20260318_220448` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/outputs/sam2_large.mp4` | `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam2_large.log` |

## Failure Note
- `sam3_efficient_efficientvit_b0` failed, log: `/workspace/models/data_examples/realsense/rs_record_distant_20260316_053807/bench_vision_20260318_220448/logs/sam3_efficient_efficientvit_b0.log`
