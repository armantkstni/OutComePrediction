from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers.LastStateTransformer import LastStateTransformer
from bucketers.BaseBucketer import BaseBucketer

class KNNBucketer(BaseBucketer):
    def __init__(self, n_neighbors=5, case_id_col=None, cat_cols=None, num_cols=None,
                 random_state=None, encoding_method="last", n_components=None):
        super().__init__(case_id_col, cat_cols, num_cols, random_state)
        self.n_neighbors = n_neighbors
        self.encoding_method = encoding_method
        self.n_components = n_components

    def fit(self, X, y=None):
        # استفاده از آخرین وضعیت برای encoding پیشوندها
        self.encoder = LastStateTransformer(case_id_col=self.case_id_col,
                                            cat_cols=self.cat_cols,
                                            num_cols=self.num_cols,
                                            fillna=True)
        X_enc = self.encoder.fit_transform(X)

        steps = []
        steps.append(('scaler', StandardScaler()))
        if self.n_components:
            steps.append(('pca', PCA(n_components=self.n_components)))
        steps.append(('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors)))

        self.pipeline = Pipeline(steps)
        # استفاده از طول پیشوند به عنوان label برای باکت‌بندی
        self.pipeline.fit(X_enc, X[self.case_id_col])

        self.X_enc = X_enc
        return self

    def predict(self, X):
        X_enc = self.encoder.transform(X)
        return self.pipeline.named_steps['knn'].predict(X_enc)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)
