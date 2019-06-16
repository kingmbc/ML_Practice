from sklearn import tree

def main():

    features = [[140, 1],
                [130, 1],
                [150, 0],
                [170, 0]]
    lables = [1, 1, 0, 0]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, lables)
    print(clf.predict([[150, 0]]))


if __name__ == "__main__":
    main()
