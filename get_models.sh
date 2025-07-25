cd examples
mkdir -p models
cd models

wget https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_voc_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_eye_buffer_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_vww_i8_w4_a4.h5

find . -maxdepth 1 -type f \( -name "*.h5" \) -exec cnn2snn convert -m  {} \;
rm *.h5


# Get datasets for examples
cd ..
mkdir -p datasets
cd datasets

wget https://data.brainchip.com/dataset-mirror/voc/test_20_classes.tfrecord
wget https://data.brainchip.com/dataset-mirror/coco/coco_anchors.pkl
wget https://data.brainchip.com/dataset-mirror/eye_tracking_ais2024_cvpr/eye_tracking_preprocessed_400frames_test.npz
wget https://data.brainchip.com/dataset-mirror/samples/eye_tracking/eye_tracking_bs100.npz