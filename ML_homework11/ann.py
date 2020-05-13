import numpy as np

# If I was a cat - should I go to sleep
# energy 0-100
# Has ate 0-1
# Played with owner  0-1
# Satisfaction form scratching the sofa 0-100
# Is my bed comfy 0-1


class GoToSleepANN:
    def __init__(self, energy=20, hasAte=0, play=1, scratchingSatisfaction=80, comfyBed=1):
        self.energy = energy
        self.has_ate = hasAte
        self.play = play
        self.scratching_satisfaction = scratchingSatisfaction
        self.comfy_bed = comfyBed

    def activate(self, signal):
        if signal < 50:
            return 0
        else:
            return 1

    def predict(self):
        inputs = np.array([self.energy, self.has_ate, self.play, self.scratching_satisfaction, self.comfy_bed])

        weights_input_hiddenOne = [0.8, 1, 0.2, 0.5, 0.7]
        weights_input_hiddenTwo = [0.6, 0.9, 0.3, 1, 0.1]
        weights_input_hiddenThree = [0.1, 1, 0.8, 0.4, 0.3]
        weights_input_hiddenFour = [1, 0.7, 0.3, 0.9, 0.2]

        weights_input_to_hidden = np.array([weights_input_hiddenOne, weights_input_hiddenTwo, weights_input_hiddenThree,
                                            weights_input_hiddenFour])
        weights_hidden_to_output = np.array([0.5, 0.2, 1, 0.7])

        from_input_to_hidden = np.dot(weights_input_to_hidden, inputs)
        hidden_activation = np.array([self.activate(s) for s in from_input_to_hidden])

        from_hidden_to_output = np.dot(weights_hidden_to_output, hidden_activation)
        output_activation = self.activate(from_hidden_to_output)

        return output_activation


model = GoToSleepANN(95, 1, 0, 50, 0)
print(model.predict())
