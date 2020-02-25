# Clustering
patient1 = {'alcohol': 'yes', 'headache': 'high', 'nausea': 'high', 'fever': 36.5}
patient2 = {'alcohol': 'no', 'headache': 'high', 'nausea': 'high', 'fever': 38.2}
patient3 = {'alcohol': 'no', 'headache': 'over_the_top', 'nausea': 'over_the_top', 'fever': 40.2}
patient4 = {'alcohol': 'yes', 'headache': 'low', 'nausea': 'none', 'fever': 36}

patients_list = [patient1, patient2, patient3, patient4]


def dividing_by_groups(patients):
    group1 = []
    group2 = []
    group3 = []

    for patient in patients:
        if patient['alcohol'] == 'yes' and patient['fever'] < 38.0:
            group1.append(patient)
        elif patient['fever'] > 39:
            group3.append(patient)
        else:
            group2.append(patient)

    print("Hangover group: ", group1)
    print("Sick group: ", group2)
    print("Probably dead group: ", group3)


dividing_by_groups(patients_list)
