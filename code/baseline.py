import numpy as np
from sklearn import *

class Baseline:
    def __init__(self, train, val, test, components=100, truth_col="date"):
        self.train = train
        self.test = test
        selt.val = val
        pca = PCA(n_components=components)
        X_processed = process(train)
        pca.fit(X_processed)
        self.X = pca.transform(X_processed)
        self.X_val = pca.transform(self.process(val))
        self.X_test = pca.transform(self.process(test))
        self.truth_col = truth_col

    def process(self, dataset, input_col="img", h=50, w=300, center=True):
        images = dataset[input_col]
        if center:
            crop = lambda x: x[(x.shape[0]//2)-(h//2):(x.shape[0]//2)+(h//2),(x.shape[1]//2)-(w//2):(x.shape[1]//2)+(w//2)]
            images = images.apply(crop) # center cropping
            images = images.apply(lambda img: skimage.transform.resize(img, (h, w)))
        return np.concatenate(images.to_numpy()).reshape(dataset.shape[0], h*w)


    def majority():
        self.blm = lambda images, major=2: [major for img in images]
        self.test["blm"] = self.blm(self.test.img.values)
        print(f"MAE: {mean_absolute_error(self.test[self.truth_col].values, self.test.blm.values):g}")
        print(f"MSE: {mean_squared_error(self.test[self.truth_col].values, self.test.blm.values):g}")


    def xgb():
        self.gxb = xgb.XGBRegressor(objective="reg:squarederror")
        self.xgb.fit(self.X, self.train[self.truth_col].values)
        self.test["xgb"] = self.xgb.predict(X_test)
        print(f"MAE: {mean_absolute_error(self.test[self.truth_col].values, self.test.xgb.values):g}")
        print(f"MSE: {mean_squared_error(self.test[self.truth_col].values, self.test.xgb.values):g}")


    def xtr():
        self.xtr = ExtraTreesRegressor()
        self.xtr.fit(self.X, self.train[self.truth_col].values)
        self.test["xtr"] = self.xtr.predict(X_test)
        print(f"MAE: {mean_absolute_error(self.test[self.truth_col].values, self.test.xgb.values):g}")
        print(f"MSE: {mean_squared_error(self.test[self.truth_col].values, self.test.xgb.values):g}")
