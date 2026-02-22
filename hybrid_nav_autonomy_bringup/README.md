# Hybrid Nav Semantic Autonomy Bringup

Launch full semantic exploration stack:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select \
  hybrid_nav_semantic_perception \
  hybrid_nav_frontier_explorer \
  hybrid_nav_semantic_rl \
  hybrid_nav_autonomy_bringup \
  hybrid_nav_safety_layer \
  hybrid_nav_bringup \
  ackermann_bringup \
  ackermann_control
source install/setup.bash

ros2 launch hybrid_nav_autonomy_bringup semantic_autonomy.launch.py
```

If you want real ONNX detection (not mock fallback), install ONNX Runtime into the ROS Python environment:

```bash
python3 -m pip install --user --break-system-packages onnxruntime
```

Optional arguments:

- `model_path:=<path_to_yolov8n.onnx>`
- `policy_file:=<path_to_semantic_rl_policy.json>`
- `enable_dataset_collection:=true`
- `use_rviz:=false`

Key topics:

- Input sensors: `/scan`, `/camera/rgb/image_raw`, `/camera/depth/image_raw`
- Semantic outputs: `/semantic/detections`, `/semantic/debug_image`, `/semantic/grid`, `/semantic/object_markers`
- Exploration outputs: `/exploration/frontier_candidates_markers`, `/semantic_rl/selected_goal`
- Run metrics artifact: `experiments/results/semantic_explorer/run_<timestamp>/metrics.json`

Export semantic map:

```bash
ros2 service call /semantic_map/export std_srvs/srv/Trigger
```
