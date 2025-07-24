cd examples
mkdir -p models
cd models

wget https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_voc_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_eye_buffer_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5
wget https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_vww_i8_w4_a4.h5

find . -maxdepth 1 -type f \( -name "*.h5" \) -exec cnn2snn convert -m  {} \;
rm *.h5