python3 utils/launch_train.py --workspace /home/ubuntu/workspace/dgl-bench \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 2 \
   --part_config /home/ubuntu/workspace/partition_dataset/ogb-paper100M-4p/ogb-paper100M.json \
   --ip_config utils/ip_config4.txt \
   "/home/ubuntu/workspace/venv/bin/python3 graphsage/train_dist_graphsage_node_classification.py --graph_name ogb-paper100M --ip_config utils/ip_config4.txt --num_trainers 8 --num_epochs 7 --eval_every 21 --breakdown"

python3 utils/launch_train.py --workspace /home/ubuntu/workspace/dgl-bench \
   --num_trainers 8 \
   --num_samplers 1 \
   --num_servers 2 \
   --part_config /home/ubuntu/workspace/partition_dataset/ogb-paper100M-4p/ogb-paper100M.json \
   --ip_config utils/ip_config4.txt \
   "/home/ubuntu/workspace/venv/bin/python3 graphsage/train_dist_gat_node_classification.py --graph_name ogb-paper100M --ip_config utils/ip_config4.txt --num_trainers 8 --num_epochs 7 --eval_every 21 --breakdown"