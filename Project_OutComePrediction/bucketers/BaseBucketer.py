class BaseBucketer:
    def __init__(self, case_id_col=None, cat_cols=None, num_cols=None, random_state=None):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        متد آموزش مدل باکت‌بندی.
        باید در کلاس‌های فرزند پیاده‌سازی شود.
        """
        raise NotImplementedError("متد fit باید در کلاس فرزند پیاده‌سازی شود.")

    def predict(self, X):
        """
        پیش‌بینی bucket مربوط به نمونه‌ها.
        باید در کلاس‌های فرزند پیاده‌سازی شود.
        """
        raise NotImplementedError("متد predict باید در کلاس فرزند پیاده‌سازی شود.")

    def fit_predict(self, X, y=None):
        """
        ابتدا مدل را آموزش می‌دهد و سپس پیش‌بینی می‌کند.
        """
        self.fit(X, y)
        return self.predict(X)
