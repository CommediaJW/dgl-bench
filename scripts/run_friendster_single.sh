python3 graphsage/train_gat_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster/ --num-trainers 1,2,4,8 --num-epochs 10 --fan-out 12,12,12 --num-hidden 8 --heads 4,4,1 --breakdown --eval-every 21
python3 graphsage/train_graphsage_node_classification.py --dataset friendster --root /home/ubuntu/workspace/processed_dataset/friendster/ --num-trainers 1,2,4,8 --num-epochs 10 --fan-out 12,12,12 --num-hidden 32 --breakdown --eval-every 21