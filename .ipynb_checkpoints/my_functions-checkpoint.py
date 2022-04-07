import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from sklearn.metrics import accuracy_score


def plot_train_test(X_train, X_test, y_train, y_test):
   fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, dpi=100)

   for header, ax, data, color in zip(["Train", "Test"], axs, [X_train, X_test], [y_train, y_test]):
       ax.set_title(header)
       sns.scatterplot(x="alcohol", y="malic_acid", data=data, c=color, s=100, edgecolor="k", ax=ax)
       ax.grid(color="k", alpha=0.1, zorder=0) 
   
   fig.tight_layout()
   plt.show()


def plot_true_predicted(X_test, y_test, clf):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, dpi=100)
    
    for header, ax, y_i, sct in zip(["True", "Predicted"], axs, [y_test, clf.predict(X_test)], ["Training", "Test"]):
        ax.set_title("{}: {}".format(header, accuracy_score(y_test, y_i).round(3)))
        sns.scatterplot(x="alcohol", y="malic_acid", data=X_test, c=y_i, s=100, edgecolor="k", ax=ax)
        ax.grid(color="k", alpha=0.1, zorder=0)
    
    fig.tight_layout()
    plt.show()

def plot_validation_curve(cv_results):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, dpi=100)
    
    for header, ax, y_name in zip(["score", "time"], axs, ["mean_test_score", "mean_fit_time"]):
        ax.set_title(header)
        p_name = cv_results["params"].keys()
        sns.lineplot(x=cv_results.iloc[:, 4].name, y=y_name, data=cv_results, ax=ax)
        ax.grid(color="k", alpha=0.1, zorder=0)
        
    
    fig.tight_layout()
    plt.show()