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
        self.image = np.resize(self.image, (3, 4))
        print(self.image)


class AnalyzeImage(TransformData):
    def analyze(self):
        self.resize_image()
        size_image = len(self)
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
    result = o.discover()
    list_object.append(result)


# Abstraction
class GettingObjectResponse(ABC):
    def __init__(self, list_values):
        self.object_list = list_values

    @abstractmethod
    def world_response(self):
        pass


class WhatWillHappen(GettingObjectResponse):

    def world_response(self):
        for index_data in self.object_list:
            if index_data == 'Meteor':
                print('We will gonna die')
                break
            elif index_data == 'Lunar Debris':
                print('We can use spatial laser')
                break
            else:
                print('We are saved')
                break


testing = WhatWillHappen(list_object)
testing.world_response()


# Encapsulation (Public, Protected, Private)
# Just one example (but we won't elaborate on anything relevant)
class Names:
    def __init__(self):
        self.id = None
        # Write your code below
        self._id = 0
        self.__id = 1


e = Names()