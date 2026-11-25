import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Decision Tree imports
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. LOAD + INSPECT
# ------------------------------------------------------------

df = pd.read_csv(r"C:\Users\User\Downloads\archive (8)\heart.csv")

print(df.shape)
print(df.head())
print(df.info())
print("\nMissing values:", df.isnull().sum().sum())

print("\nClass balance (1 = heart disease):\n",
      df["HeartDisease"].value_counts())


# ------------------------------------------------------------
# 2. PREPARE DATA
# ------------------------------------------------------------

cat_columns = ["Sex", "ChestPainType", "RestingECG",
               "ExerciseAngina", "ST_Slope"]

# Text columns (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
# must become numbers. One-hot encoding makes one 0/1 column per category.

df = pd.get_dummies(df, drop_first=True, dtype=int)

X = df.drop(columns="HeartDisease")
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ------------------------------------------------------------
# 3. TREE 1: NO LIMITS (grows until every leaf is pure -> overfits)
# ------------------------------------------------------------

full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

print("\n=== UNRESTRICTED TREE ===")
print("Depth       :", full_tree.get_depth())
print("Leaves      :", full_tree.get_n_leaves())
print("Train acc   :", round(full_tree.score(X_train, y_train), 4))
print("Test acc    :", round(full_tree.score(X_test, y_test), 4))

# -----------------------------------------------
# 4. FIND A GOOD max_depth (train vs test accuracy curve)
# -----------------------------------------------

depths = range(1, 16)
train_acc, test_acc = [], []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(t.score(X_train, y_train))
    test_acc.append(t.score(X_test, y_test))

plt.figure(figsize=(7, 4))
plt.plot(depths, train_acc, marker="o", label="Train accuracy")
plt.plot(depths, test_acc, marker="o", label="Test accuracy")

plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Overfitting check: train vs test accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("depth_vs_accuracy.png")
plt.show()

# -----------------------------------------------
#5. TREE 2: PRUNED TREE (controlled complexity)
# -----------------------------------------------
tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_leaf=10, # at most 3 questions deep
    random_state=42# every leaf must contain at least 10 patients
)
tree.fit(X_train, y_train)

# -----------------------------------------------
# 6. EVALUATE
# -----------------------------------------------

y_pred = tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== PRUNED TREE (max_depth=3) ===")
print("Confusion matrix:\n", cm)
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

print("Train acc  :", round(tree.score(X_train, y_train), 4))
print("Test acc   :", round(accuracy_score(y_test, y_pred), 4))
print("Precision  :", round(precision_score(y_test, y_pred), 4))
print("Recall     :", round(recall_score(y_test, y_pred), 4))
print("F1-score   :", round(f1_score(y_test, y_pred), 4))

print("\n", classification_report(
    y_test,
    y_pred,
    target_names=["No disease", "Disease"]
))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
    xticklabels=["Pred No", "Pred Yes"],
    yticklabels=["Actual No", "Actual Yes"]
)

plt.title("Confusion Matrix (pruned tree)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# -----------------------------------------------
# 7. VISUALIZE THE TREE
# -----------------------------------------------

plt.figure(figsize=(20, 9))

plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No disease", "Disease"],
    filled=True,       # colour by majority class
    rounded=True,
    fontsize=10
)

plt.title("Pruned Decision Tree")
plt.tight_layout()
plt.savefig("pruned_tree.png")
plt.show()

# The same tree as plain-text if/else rules

print("\nTree as text rules:\n")
print(export_text(tree, feature_names=list(X.columns)))

# -----------------------------------------------
# 8. FEATURE IMPORTANCE
# -----------------------------------------------

imp = pd.Series(
    tree.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature importances:\n", imp[imp > 0])

plt.figure(figsize=(7, 4))
imp[imp > 0].sort_values().plot(kind="barh")

plt.title("Feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
