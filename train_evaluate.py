from sklearn.metrics import confusion_matrix


def evaluate(pipe, X_test, y_test):
    # score
    rdmf_score = pipe.score(X_test, y_test)
    print(
        f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation"
    )

    print(20 * "-")
    print("matrice de confusion")
    print(confusion_matrix(y_test, pipe.predict(X_test)))