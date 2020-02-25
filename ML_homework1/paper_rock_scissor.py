import itertools
from random import choice

poss_choices = ['paper', 'rock', 'scissors']

poss_combs = [combination for combination in itertools.product(poss_choices, repeat=3)]
combs_dict = dict.fromkeys(poss_combs, 0)
#print(combs_dict)

attempts = 0
score = 0
user_inputs = []

while True:
    computer_choice = 0
    if attempts < 5:
        computer_choice = choice(['paper', 'rock', 'scissors'])
    else:
        #  paper > rock and paper > scissors
        if combs_dict[user_inputs[-2], user_inputs[-1], 'paper'] > combs_dict[user_inputs[-2], user_inputs[-1], 'rock'] and combs_dict[user_inputs[-2], user_inputs[-1], 'paper'] > combs_dict[user_inputs[-2], user_inputs[-1], 'scissors']:
            computer_choice = 'paper'
        #  rock > paper and rock > scissors
        elif combs_dict[user_inputs[-2], user_inputs[-1], 'rock'] > combs_dict[user_inputs[-2], user_inputs[-1], 'paper'] and combs_dict[user_inputs[-2], user_inputs[-1], 'rock'] > combs_dict[user_inputs[-2], user_inputs[-1], 'scissors']:
            computer_choice = 'rock'
        #  scissors > rock and scissors > paper
        else:
            computer_choice = 'scissors'

    print("What do you choose? ('paper', 'rock', 'scissors')")
    user_inp = input()
    print("Computer chooses: ", computer_choice)
    if computer_choice == user_inp:
        print("Computer wins!")
        score += 1
    else:
        print("Computer looses!")
        score -= 1
    attempts += 1

    user_inputs.append(user_inp)
    if len(user_inputs) > 3:
        combs_dict[tuple(user_inputs[-3:])] += 1

