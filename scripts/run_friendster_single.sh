torchrun --nproc_per_node 8 graphsage/train_gat_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 8 --num-epochs 7 --fan-out 12,12,12 --num-hidden 8 --heads 4,4,1 --breakdown --eval-every 21
torchrun --nproc_per_node 4 graphsage/train_gat_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 4 --num-epochs 7 --fan-out 12,12,12 --num-hidden 8 --heads 4,4,1 --breakdown --eval-every 21
torchrun --nproc_per_node 2 graphsage/train_gat_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 2 --num-epochs 7 --fan-out 12,12,12 --num-hidden 8 --heads 4,4,1 --breakdown --eval-every 21
torchrun --nproc_per_node 1 graphsage/train_gat_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 1 --num-epochs 7 --fan-out 12,12,12 --num-hidden 8 --heads 4,4,1 --breakdown --eval-every 21
torchrun --nproc_per_node 8 graphsage/train_graphsage_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 8 --num-epochs 7 --fan-out 12,12,12 --num-hidden 32 --breakdown --eval-every 21
torchrun --nproc_per_node 4 graphsage/train_graphsage_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 4 --num-epochs 7 --fan-out 12,12,12 --num-hidden 32 --breakdown --eval-every 21
torchrun --nproc_per_node 2 graphsage/train_graphsage_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 2 --num-epochs 7 --fan-out 12,12,12 --num-hidden 32 --breakdown --eval-every 21
torchrun --nproc_per_node 1 graphsage/train_graphsage_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster --num-trainers 1 --num-epochs 7 --fan-out 12,12,12 --num-hidden 32 --breakdown --eval-every 21