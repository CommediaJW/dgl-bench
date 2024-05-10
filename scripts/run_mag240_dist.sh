for i in 4 2 1
do
   python3 utils/launch_train.py --workspace /home/ubuntu/dgl_workspace/dgl-bench \
      --num_trainers 8 \
      --num_samplers 1 \
      --num_servers 1 \
      --part_config /home/ubuntu/dgl_workspace/partition_dataset/mag240m_${i}p/mag240m.json \
      --ip_config utils/ip_config${i}.txt \
      "/home/ubuntu/miniconda3/envs/raft/bin/python3 training/train_dist_node_classification.py \
         --graph_name mag240m \
         --ip_config utils/ip_config${i}.txt \
         --num_trainers 8 \
         --num_epochs 4 --eval_every 999 \
         --model sage --num_hidden 128 --num_heads 8 \
         --fan_out 12,12,12 --batch_size 1536 \
         --breakdown"
done

for i in 4 2 1
do
   python3 utils/launch_train.py --workspace /home/ubuntu/dgl_workspace/dgl-bench \
      --num_trainers 8 \
      --num_samplers 1 \
      --num_servers 1 \
      --part_config /home/ubuntu/dgl_workspace/partition_dataset/mag240m_${i}p/mag240m.json \
      --ip_config utils/ip_config${i}.txt \
      "/home/ubuntu/miniconda3/envs/raft/bin/python3 training/train_dist_node_classification.py \
         --graph_name mag240m \
         --ip_config utils/ip_config${i}.txt \
         --num_trainers 8 \
         --num_epochs 4 --eval_every 999 \
         --model gat --num_hidden 128 --num_heads 8 \
         --fan_out 12,12,12 --batch_size 1536 \
         --breakdown"
done
