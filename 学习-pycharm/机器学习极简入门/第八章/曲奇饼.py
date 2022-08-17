class Bayes(object):
    def __init__(self):
        self._container = dict()

    def Set(self, hypothis, prob):
        self._container[hypothis] = prob

    def Mult(self, hypothis, prob):
        old_prob = self._container[hypothis]
        self._container[hypothis] = old_prob * prob

    def Normalize(self):
        count = 0
        for hypothis in self._container.values():
            count = count + hypothis
        for hypothis, prob in self._container.items():
            self._container[hypothis] = self._container[hypothis] / count

    def Prob(self, hypothis):
        Prob = self._container[hypothis]
        return Prob


if __name__ == '__main__':
    bayes = Bayes()

    bayes.Set('BowA', 0.5)
    bayes.Set('BowB', 0.5)

    bayes.Mult('BowA', 0.75)
    bayes.Mult('BowB', 0.5)

    bayes.Normalize()
    prob = bayes.Prob('BowA')
    print(prob)
