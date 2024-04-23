.PHONY: board
board:
	python reorganize_logs.py $(env)
	tensorboard --logdir=clean_logs