# The goal is get the image and identify what is
import numpy as np

# Input data
data_value = np.array([2, 5])


# Inheritance
class IdentifyImage:

    def __init__(self, data):
        self.image = data

    def __len__(self):
        print('Size of Image: ', len(self.image))
        return len(self.image)


class TransformData(IdentifyImage):
    def resize_image(self):
        print(self.image)
        self.image = np.resize(self.image, (3, 4))
        print(self.image)


class AnalyzeImage(TransformData):
    def analyze(self, data):
        TransformData.resize_image(self)
        size_image = TransformData.__len__(self)


inheritance_data = AnalyzeImage(data_value)
inheritance_data.analyze(data_value)


# Polymorphism (There are other ways to do this)
class Meteor(IdentifyImage):

    def __repr__(self):
        print(f'Its a Meteor')

    def discover(self, data):
        IdentifyImage.__init__(self, data)
        size = IdentifyImage.__len__(self)

        if size == 4:
            self.__repr__()
        else:
            print('Its not a Meteor')


class LunarDebris(IdentifyImage):
    def __repr__(self):
        print(f'Its a Lunar Debris')

    def discover(self, data):
        IdentifyImage.__init__(self, data)
        size = IdentifyImage.__len__(self)

        if size == 2:
            self.__repr__()
        else:
            print('Its not a Lunar Debris')


class MinimalObject(IdentifyImage):
    def __repr__(self):
        print(f'Its a Minimal Object in space')

    def discover(self, data):
        IdentifyImage.__init__(self, data)
        size = IdentifyImage.__len__(self)

        if size == 1:
            self.__repr__()
        else:
            print('Nothing Relevant')


polymorphism_data_1 = Meteor(data_value)
polymorphism_data_2 = LunarDebris(data_value)
polymorphism_data_3 = LunarDebris(data_value)

polymorphism_data_1.discover(data=data_value)
polymorphism_data_2.discover(data=data_value)
polymorphism_data_3.discover(data=data_value)

# Abstraction


# Encapsulation (Public, Protected, Private)
