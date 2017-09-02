all:
	process pickle train
process: 
	python LFW_img_transform.py
	python auto_occlude.py
pickle:
	python pickle_images.py
train:
	python train_model$(num).py