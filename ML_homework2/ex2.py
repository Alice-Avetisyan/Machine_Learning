# Regression
# Weather prediction (fahrenheit)

week1 = {'Monday': 15.8, 'Tuesday': 26.6, 'Wednesday': 32}  # Thursday ?
week2 = {'Monday': 28.4, 'Tuesday': 33.8, 'Wednesday': 50}
week3 = {'Monday': -4, 'Tuesday': -22, 'Wednesday': -13}


def weather(days):
    warmth = (days['Monday']+days['Tuesday']+days['Wednesday'])/len(days)
    if warmth > 32:
        warmth += 3
        days['Thursday'] = warmth
    else:
        warmth -= 3
        days['Thursday'] = warmth

    print(days)


weather(week1)
weather(week2)
weather(week3)

