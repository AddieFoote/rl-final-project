.PHONY: board

board:
	python reorganize_logs.py
	tensorboard --logdir=clean_logs