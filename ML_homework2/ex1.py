# Classification

# average weight for a cat is 3.5-4.5kg
# average weight for a big dog is 15.5-150.4kg
animal1 = {'whiskers': 'Y', 'ears': 'pointy', 'weight': 3.5}
animal2 = {'whiskers': 'N', 'ears': 'pointy', 'weight': 15.8}
animal3 = {'whiskers': 'Y', 'ears': 'pointy', 'weight': 4.2}
animal4 = {'whiskers': 'N', 'ears': 'floppy', 'weight': 45.3}


def animal_clscation(animal):
    if animal['whiskers'] == 'Y' and animal['ears'] == 'pointy' and animal['weight'] < 15:
        print("This is a cat")
    else:
        print("This is a dog")


animal_clscation(animal1)
animal_clscation(animal2)
animal_clscation(animal3)
animal_clscation(animal4)
