import os
import time
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from ml_features import extract_features
DATASET_DIR = "dataset"
CLASSES     = ["with_mask", "without_mask"]
MODEL_PATH  = "mask_detector_ml.joblib"
print("[INFO] Extracting features (this takes a few minutes)...")
X, y = [], []
for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_DIR, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(label)
X = np.array(X)
y = np.array(y)
print(f"[INFO] {len(X)} images | feature vector length: {X.shape[1]}")
plt.figure(figsize=(5, 3))
plt.bar(CLASSES, [(y == 0).sum(), (y == 1).sum()], color=["#4CAF50", "#F44336"])
plt.title("Class Distribution")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("ml_class_distribution.png", dpi=150)
plt.close()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, stratify=y_train, random_state=42)
print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42),
    "SVM (RBF)": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
}
results = {}
for name, model in models.items():
    print(f"[INFO] Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    results[name] = val_acc
    print(f"       val accuracy = {val_acc*100:.2f}%  ({time.time()-t0:.0f}s)")

best_name  = max(results, key=results.get)
best_model = models[best_name]
print(f"\n[INFO] Best model on validation set: {best_name}")
plt.figure(figsize=(6, 3.5))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()],
               color=["#2E75B6", "#4CAF50", "#F4A62A"])
plt.ylim(80, 100)
plt.ylabel("Validation accuracy (%)")
plt.title("Model Comparison")
for b, v in zip(bars, results.values()):
    plt.text(b.get_x() + b.get_width() / 2, v * 100 + 0.2,
             f"{v*100:.2f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("ml_model_comparison.png", dpi=150)
plt.close()
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("\n── Classification Report (test set) ──")
print(classification_report(y_test, y_pred, target_names=CLASSES))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("ml_confusion_matrix.png", dpi=150)
plt.close()
joblib.dump(best_model, MODEL_PATH)
print(f"\n[RESULT] Best model : {best_name}")
print(f"[RESULT] Test accuracy: {test_acc*100:.2f}%")
print(f"[INFO]   Saved to {MODEL_PATH}")