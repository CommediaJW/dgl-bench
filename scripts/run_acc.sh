# torchrun --nproc_per_node 2 training/train_node_classification_gpu.py --dataset ogbn-products --root /home/ubuntu/dgl_workspace/dataset/ogbn-products --num-trainers 2 --num-epochs 100 --fan-out 20,20,20 --batch-size 1536 --num-hidden 128 --breakdown --eval-every 10 --model sage
torchrun --nproc_per_node 8 training/train_node_classification.py --dataset ogbn-papers100M --root /home/ubuntu/dgl_workspace/dataset/ogbn-papers100M --num-trainers 8 --num-epochs 100 --fan-out 20,20,20 --batch-size 1000 --num-hidden 128 --breakdown --eval-every 2 --model sage
# torchrun --nproc_per_node 2 training/train_node_classification_gpu.py --dataset ogbn-products --root /home/ubuntu/dgl_workspace/dataset/ogbn-products --num-trainers 2 --num-epochs 100 --fan-out 20,20,20 --batch-size 1536 --num-hidden 128 --breakdown --eval-every 10 --model gat --num-heads 8
# torchrun --nproc_per_node 2 training/train_node_classification.py --dataset ogbn-papers100M --root /home/ubuntu/dgl_workspace/dataset/ogbn-papers100M --num-trainers 2 --num-epochs 100 --fan-out 20,20,20 --batch-size 1000 --num-hidden 128 --breakdown --eval-every 10 --model gat --num-heads 8