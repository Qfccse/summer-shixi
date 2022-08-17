import random
import matplotlib.pyplot as plt


def coin_flip(minX, maxX):
    ratios = []
    x = range(minX, maxX + 1)
    for number_Flips in x:
        numHeads = 0
        for n in range(number_Flips):
            if random.random() < 0.5:
                numHeads += 1
        numTails = number_Flips - numHeads
        ratios.append(numHeads / float(numTails))
    plt.title("Heads/Tails Ratios")
    plt.xlabel("number of flips")
    plt.ylabel("Heads/Tails")
    plt.plot(x, ratios)
    plt.hlines(1, 2, x[-1], linestyles="dashed", colors="y")
    plt.show()


coin_flip(2, 5000)
