# code from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
import random

def getSampleIndices(ds, frac, shuffle=False):
    counts = 0
    k = int(len(ds)*frac)
    data_number = [i for i in range(len(ds))]

    if shuffle:
        random.shuffle(data_number)

    SampleIndices = data_number[:k]
    
    print('We sample {:d} ({:.2f} %) training data'.format(len(SampleIndices),frac*100))

    return SampleIndices
