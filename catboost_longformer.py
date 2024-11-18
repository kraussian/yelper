import json
import pandas as pd
from   glob import glob
from   catboost import CatBoostClassifier, Pool
from   sklearn.model_selection import train_test_split
from   sklearn.decomposition import PCA

# Load embeddings from parquet file
idx = 1
print("Available Longformer Embedding files:")
files = glob("longformer_embeddings_*.parquet")
for file in files:
    print(f"[{idx}] {file}")
userchoice = int(input("\nSelect the Longformer Embedding file to load: "))
X = pd.read_parquet(files[userchoice-1])
del idx, files, file, userchoice

# Load labels from JSON file
filename = 'reviews_sample_25000.json'
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)
y = [int(rec['stars'])-1 for rec in data]
del filename, file, data

def train_catboost(
    X, y, iterations=500, learning_rate=0.1, depth=6,
    test_size=0.2, random_state=42, task_type='GPU', pool=False
):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize CatBoost model
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=100,
        loss_function='MultiClass',
        task_type=task_type
    )

    if pool:
        # Optional: Use CatBoost Pooling
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)
        model.fit(train_pool)
    else:
        # Train the model normally
        model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Return model
    return model

# Train model normally
model = train_catboost(X, y)
"""
0:      learn: 1.5522732        total: 24.1ms   remaining: 12s
100:    learn: 0.8653434        total: 1.19s    remaining: 4.71s
200:    learn: 0.7406242        total: 2.32s    remaining: 3.44s
300:    learn: 0.6545676        total: 3.44s    remaining: 2.27s
400:    learn: 0.5860685        total: 4.55s    remaining: 1.12s
499:    learn: 0.5262706        total: 5.67s    remaining: 0us
Test Accuracy: 0.5768
"""

# Train model using 1k its & 0.05 LR
model = train_catboost(X, y, iterations=1000, learning_rate=0.05)
"""
0:      learn: 1.5801091        total: 12.7ms   remaining: 12.7s
100:    learn: 0.9830431        total: 1.18s    remaining: 10.5s
200:    learn: 0.8644613        total: 2.37s    remaining: 9.43s
300:    learn: 0.7952683        total: 3.56s    remaining: 8.27s
400:    learn: 0.7404948        total: 4.69s    remaining: 7s
500:    learn: 0.6942478        total: 5.82s    remaining: 5.8s
600:    learn: 0.6517177        total: 6.94s    remaining: 4.6s
700:    learn: 0.6145193        total: 8.07s    remaining: 3.44s
800:    learn: 0.5798404        total: 9.21s    remaining: 2.29s
900:    learn: 0.5483491        total: 10.3s    remaining: 1.14s
999:    learn: 0.5202302        total: 11.5s    remaining: 0us
Test Accuracy: 0.5868
"""

# Train model using CatBoost Pooling
model_pool = train_catboost(X, y, pool=True)

# Optional: Use PCA to reduce embeddings
target_dim = 50
pca = PCA(n_components=target_dim)
X_reduced = pca.fit_transform(X)
model_pca = train_catboost(X_reduced, y)
"""
0:      learn: 1.5566618        total: 6ms      remaining: 3s
100:    learn: 0.9471062        total: 520ms    remaining: 2.05s
200:    learn: 0.8215244        total: 1.03s    remaining: 1.54s
300:    learn: 0.7414304        total: 1.56s    remaining: 1.03s
400:    learn: 0.6778344        total: 2.07s    remaining: 512ms
499:    learn: 0.6244050        total: 2.58s    remaining: 0us
Test Accuracy: 0.5704
"""