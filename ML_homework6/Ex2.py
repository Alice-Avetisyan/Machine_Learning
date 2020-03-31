import matplotlib.pyplot as plt

fig = plt.figure()
axis1 = fig.add_subplot(2, 2, 1)
axis1.plot([2, 5, 6, 1, 3, 7], '>m')

axis2 = fig.add_subplot(2, 2, 2)
axis2.plot([200, 463, 34, 763, 23], [20, 46, 87, 12, 42], 'r--2')

x = [23, 57, 97, 23, 64]
axis3 = fig.add_subplot(223)
axis3.plot(x, '-.*b')

y = [24, 456, 87, 12, 75]
axis4 = fig.add_subplot(224)
axis4.plot(x, y, 'go')

plt.show()

