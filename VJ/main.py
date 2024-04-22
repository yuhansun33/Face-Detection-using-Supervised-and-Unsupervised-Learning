import dataset
import adaboost
import utils
import detection
import matplotlib.pyplot as plt
import argparse

def main():
    # Part 1: Implement load_data_FDDB function in dataset.py and test the following code.
    print('Loading images')
    trainData, testData = dataset.create_dataset()
    print(f'The number of training samples loaded: {len(trainData)}')
    print(f'The number of test samples loaded: {len(testData)}')

    print('Show the first and last images of training dataset')
    fig, ax = plt.subplots(1, 2)
    ax[0].axis('off')
    ax[0].set_title('Face')
    ax[0].imshow(trainData[0][0], cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Non face')
    ax[1].imshow(trainData[-1][0], cmap='gray')
    plt.show()

    # Part 2: Implement selectBest function in adaboost.py and test the following code.
    # Part 3: Modify difference values at parameter T of the Adaboost algorithm.
    # And find better results. Please test value 1~10 at least.
    # print('Start training your classifier')
    # with open('experiment.txt', 'a') as f: 
    #     f.write('T = {},\t False Positive Rate\tFalse Negative Rate\tAccuracy\n'.format(T))
        
    # clf = adaboost.Adaboost(T=10)
    for T in range(1, 11):
        clf = adaboost.Adaboost(T)
        clf.train(trainData)

        clf.save('clf_200_1_10')
        clf = adaboost.Adaboost.load('clf_200_1_10')

        with open('experiment.txt', 'a') as f: 
            f.write('TrainData : ')
        print('\nEvaluate your classifier with training dataset')
        utils.evaluate(clf, trainData)

        with open('experiment.txt', 'a') as f: 
            f.write('TestData : ')
        print('\nEvaluate your classifier with test dataset')
        utils.evaluate(clf, testData)

        # Part 4: Implement detect function in detection.py and test the following code.
        print('\nDetect faces at the assigned location using your classifier')
        detection.detect('detect/detectData.txt', clf)

        # Part 5: Test classifier on your own images
        print('\nDetect faces on your own images')
        detection.detect('detect/yourOwnImages.txt', clf)


if __name__ == "__main__":
    
    main()
