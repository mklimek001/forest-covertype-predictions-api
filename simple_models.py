# Heuristic use connection between tree type and elevation
def heuristic(sample):
    elevation = sample['Elevation']

    if elevation < 0.2:
        return 4
    elif 0.2 <= elevation < 0.3:
        return 5
    elif 0.3 <= elevation < 0.4:
        return 6
    elif 0.4 <= elevation < 0.5:
        return 3
    elif 0.5 <= elevation < 0.6:
        return 1
    elif 0.6 <= elevation < 0.7:
        return 2
    else:
        return 7
