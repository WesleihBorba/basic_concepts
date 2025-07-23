# The goal is get the image and identify what is
import numpy as np


# Inheritance
class IdentifyImage:

    def __init__(self):
        self.image = np.array([2, 5, 6, 7])

    def __len__(self):
        print(self.image)
        print('Size of Image: ', len(self.image))
        return len(self.image)


class TransformData(IdentifyImage):
    def resize_image(self):
        self.image = np.resize(self.image,(3, 4))
        print(self.image)


class AnalyzeImage(TransformData):
    def analyze(self):
        TransformData.resize_image(self)
        size_image = TransformData.__len__(self)


inheritance_data = AnalyzeImage()
inheritance_data.analyze()

# Polymorphism (There are other ways to do this)
class Meteor(AnalyzeImage):

    def discover(self):
        size = AnalyzeImage.analyze(self)
        print(size)


polymorphism_data = Meteor()
polymorphism_data.discover()

# Abstraction


# Encapsulation (Public, Protected, Private)
