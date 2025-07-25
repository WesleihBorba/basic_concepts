# The goal is get the image and identify what is
import numpy as np
from abc import ABC, abstractmethod

# Input data
data_value = np.array([2, 5, 7, 9])


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
    def analyze(self):
        TransformData.resize_image(self)
        size_image = TransformData.__len__(self)
        print(size_image)


# Polymorphism (There are other ways to do this)

class Meteor(IdentifyImage):

    def __repr__(self):
        return 'Its a Meteor'

    def discover(self):
        if len(self) >= 3:
            return 'Meteor'
        else:
            return 'Its not a Meteor'


class LunarDebris(IdentifyImage):
    def __repr__(self):
        return 'Its a Lunar Debris'

    def discover(self):
        if len(self) == 2:
            return 'Lunar Debris'
        else:
            return 'Its not a Lunar Debris'


class MinimalObject(IdentifyImage):
    def __repr__(self):
        return 'Its a Minimal Object in space'

    def discover(self):
        if len(self) <= 1:
            return 'Minimal Object'
        else:
            return 'Nothing Relevant'


polymorphism_loop = [Meteor(data_value), LunarDebris(data_value), MinimalObject(data_value)]

list_object = []
for o in polymorphism_loop:
    o.discover()
    list_object.append(o)

print(list_object)

# Abstraction
#class GettingObjectResponse():


# Encapsulation (Public, Protected, Private)
