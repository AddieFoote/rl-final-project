
NUM_STATE_CLASSES = 10
SOMETHING_IDK_WHAT = 200
NUM_DIRECTIONS = 4
EPISODES_PER_BATCH = 16
CHANGE_REWARD = False
FAKE_GREEDY = True

from wrappers import OneHotImageZerothIndexWrapper

ENV_WRAPPER = OneHotImageZerothIndexWrapper