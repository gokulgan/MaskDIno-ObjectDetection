# MaskDIno-ObjectDetection

This repository can be used to train and test data on the **MaskDINO** network architecture for object detection and pose estimation.



## Docker Setup

Build the Docker image using the provided `build.yml` file:
```bash
docker compose -f docker/build.yml build
```

---

## Training

Start training using `11_obj_train_swinL.py`:
```bash
python3 11_obj_train_swinL.py \
  --config /workspace/maskdino_ros_pkg/maskdino_ros_pkg/maskdino/config_swinL.yaml \
  --data_dir /<your_data_directory> \
  --data_yaml /<your_data_yaml> \
  --output_dir /<your_output_directory> \
  --epochs 120
```

| Argument | Description |
|----------|-------------|
| `--config` | Path to the SwinL config YAML file |
| `--data_dir` | Path to the training data directory |
| `--data_yaml` | Path to the data YAML file |
| `--output_dir` | Path where trained weights will be saved |
| `--epochs` | Number of training epochs (default: 120) |

After training completes, the model weights will be saved to the specified `--output_dir`.

---

## Inference (ROS2)

Two ROS2 nodes are available for running the model:

| Script | Description |
|--------|-------------|
| `maskdino_ros_g_n.py` | Object detection **with** pose estimation |
| `maskdino_ros_g.py` | Object detection **only** |

### Run Detection with Pose Estimation
```bash
ros2 run maskdino_ros_pkg maskdino_ros_g_n \
  --ros-args \
  -p weights_path:=/<path_to_weights> \
  -p config_path:=/<path_to_config> \
  -p labels_path:=/<path_to_labels> \
  -p img_topic:=/<image_topic> \
  -p depth_topic:=/<depth_topic> \
  -p out_topic:=/maskdino_ros2/detection \
  -p conf_thresh:=0.5 \
  -p visualize:=true \
  -p target_frame_id:=base_link
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `weights_path` | Path to the trained model weights |
| `config_path` | Path to the model config file |
| `labels_path` | Path to the labels file |
| `img_topic` | Input RGB image topic |
| `depth_topic` | Input depth image topic |
| `out_topic` | Output detection topic (default: `/maskdino_ros2/detection`) |
| `conf_thresh` | Confidence threshold for detections (default: `0.5`) |
| `visualize` | Enable visualization output (default: `true`) |
| `target_frame_id` | Target TF frame for pose estimation (default: `base_link`) |

---
