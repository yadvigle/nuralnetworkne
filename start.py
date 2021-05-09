import pickle
from matplotlib import pyplot as plt
from neuralntework import thinker


def main():
    image_path = 'test_X.pickle'
    with open(image_path, 'rb') as file:
        data = pickle.load(file)

    y = thinker(data[2])
    with open('testanswer1.pickle', "wb") as file:
        pickle.dump(y, file)



main()