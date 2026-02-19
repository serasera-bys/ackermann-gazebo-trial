# Hazard Score VHDL Companion

This folder is a companion digital-design artifact for CV coverage.
It is not wired into ROS2 runtime yet.

## What it implements

- `hazard_score.vhd`
  - Input: `front/left/right` obstacle range (mm)
  - Output:
    - `hazard_score_o` (higher score means more risk)
    - `danger_flag_o` (asserted when any critical threshold is crossed)
- `tb_hazard_score.vhd`
  - Testbench covering safe/warn/critical cases.

## Quick test with GHDL

```bash
cd /home/bernard/ros2_ws/src/hybrid_nav_robot/fpga/hazard_score_vhdl
ghdl -a --std=08 hazard_score.vhd tb_hazard_score.vhd
ghdl -e --std=08 tb_hazard_score
ghdl -r --std=08 tb_hazard_score --assert-level=error
```

Expected: simulation ends with `tb_hazard_score PASSED` note and no assertion errors.

## Planned integration path

- Use module output as a candidate FPGA offload for safety pre-scoring.
- Feed `hazard_score_o` into software safety layer as an additional signal.
- Keep ROS2 safety hard-stop as the final authority.
