import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


def plot_train_test(X_train, X_test, y_train, y_test):
   fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, dpi=100)

   for header, ax, data, color in zip(["Train", "Test"], axs, [X_train, X_test], [y_train, y_test]):
       ax.set_title(header)
       sns.scatterplot(x="alcohol", y="malic_acid", data=data, c=color, s=100, ax=ax)
       #ax.set_xlabel("alcohol")
       # #ax.set_ylabel("malic_acid")
       ax.grid(color="k", alpha=0.1, zorder=0) 
   
   fig.tight_layout()
   plt.show()
