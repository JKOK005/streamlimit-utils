import numpy as np

# Types of generator classes
class RandomArrCreator(object):
	@classmethod
	def create(cls, arr_size: np.ndarray) -> np.ndarray:
		"""
		Generates random images specified by img_size. 
		All image values are constrained between 0 - 1
		"""
		return np.random.rand(*arr_size)

# Class to generate images or labels
class ArrGenerator(object):
	def __init__(self, img_size: np.ndarray, gen_cls):
		self.img_size 	= img_size
		self.gen_cls 	= gen_cls
		return

	def generate(self):
		return self.gen_cls.create(arr_size = self.img_size)

# Class to generate training data with labels for ML training
class DataGenerator(object):
	@staticmethod
	def generate(img_gen, label_gen):
		while True:
			yield (img_gen.generate(), label_gen.generate())

if __name__ == "__main__":
	# Simulate generating an (img, label) pair
	BATCH_SIZE 	= 10 
	IMG_SIZE 	= np.array([BATCH_SIZE, 32, 32, 3])
	LABEL_SIZE 	= np.array([BATCH_SIZE, 10])

	img_gen 	= ArrGenerator(img_size = IMG_SIZE, gen_cls = RandomArrCreator)
	label_gen 	= ArrGenerator(img_size = LABEL_SIZE, gen_cls = RandomArrCreator)
	data_gen 	= DataGenerator.generate(img_gen = img_gen, label_gen = label_gen)

	(smpl_img, smpl_label) = next(data_gen)
	assert(np.array_equal(smpl_img.shape, IMG_SIZE) and np.array_equal(smpl_label.shape, LABEL_SIZE))