mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
rm -r modelnet40_normal_resampled.zip
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm -r shapenetcore_partanno_segmentation_benchmark_v0.zip
cd ..
wget https://drive.google.com/file/d/1L25i0l6L_b1Vw504WQR8-Z0oh2FJA0G9/view?usp=sharing
unzip checkpoint.zip
rm -r checkpoint.zip