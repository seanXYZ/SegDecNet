
Pytorch Implement of the paper "Segmentation Based Deep-Learning Approach for Surface DefectDetection"
https://arxiv.org/abs/1903.08536

Modified little things without changing the accuracy

BUT, Much Faster than the paper!!!

Achieve less 3ms per image of the size 704x256, by a single Nvidia 1080Ti. This speed can rival Cognex's VIDI ~ 



step1: train segment net

	python train_segment.py

step2: train decision net

	python train_decision.py

step3: test 

	python test.py

