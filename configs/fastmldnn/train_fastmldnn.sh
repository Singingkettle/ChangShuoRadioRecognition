echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.1_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2900 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.1_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.2_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2901 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.2_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.3_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2902 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.3_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.4_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2903 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.4_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.6_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2904 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.6_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.7_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2905 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.7_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.8_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2906 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.8_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-0.9_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2907 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-0.9_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-balance-1.0_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2908 tools/train.py ./configs\fastmldnn\fastmldnn_abl-balance-1.0_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-last_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2909 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-last_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-logsumexp_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2910 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-logsumexp_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-max_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2911 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-max_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-mean_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2912 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-mean_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-median_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2913 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-median_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-min_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2914 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-min_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-quantile_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2915 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-quantile_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-merge-std_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2916 tools/train.py ./configs\fastmldnn\fastmldnn_abl-merge-std_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-no-class-distance-expansion_-iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2917 tools/train.py ./configs\fastmldnn\fastmldnn_abl-no-class-distance-expansion_-iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-without-group-conv-use-bigru2_iq-ap-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2918 tools/train.py ./configs\fastmldnn\fastmldnn_abl-without-group-conv-use-bigru2_iq-ap-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-without-group-conv-use-lstm2_iq-ap-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2919 tools/train.py ./configs\fastmldnn\fastmldnn_abl-without-group-conv-use-lstm2_iq-ap-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-without-group-conv-use-rnn2_iq-ap-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2920 tools/train.py ./configs\fastmldnn\fastmldnn_abl-without-group-conv-use-rnn2_iq-ap-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_abl-without-group-conv_iq-ap-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2921 tools/train.py ./configs\fastmldnn\fastmldnn_abl-without-group-conv_iq-ap-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_iq-ap-channel-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2922 tools/train.py ./configs\fastmldnn\fastmldnn_iq-ap-channel-deepsig201610A.py --seed 0 --launcher pytorch

echo "Start Test: ./configs\fastmldnn\fastmldnn_iq-ap-deepsig201610A.py"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2923 tools/train.py ./configs\fastmldnn\fastmldnn_iq-ap-deepsig201610A.py --seed 0 --launcher pytorch

