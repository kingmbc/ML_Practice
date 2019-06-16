from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def main():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.5)

    classfier_dt = DecisionTreeClassifier()
    classfier_dt.fit(x_train, y_train)

    classfier_knn = KNeighborsClassifier()
    classfier_knn.fit(x_train, y_train)

    dt_prediction = classfier_dt.predict(x_test)
    knn_prediction = classfier_knn.predict(x_test)

    print(accuracy_score(y_test, dt_prediction))
    print(accuracy_score(y_test, knn_prediction))

if __name__ == '__main__':
    main()
