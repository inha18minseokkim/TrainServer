import numpy as np
from loguru import logger

import PriceLoader
from scipy.optimize import minimize
from Strategies.Strategy import Strategy
def objfunc(w,rtn_cov):
    return np.sqrt(w.T@rtn_cov@w)

class OptimizerStrategy(Strategy):
    def __init__(self, codelist: list[str], rtnstdlist: list):
        self.codelist = codelist
        # rtnstdlist에서 수익률만 빼와
        self.rtnlist = [rtn for [rtn, std] in rtnstdlist]
    def execute(self):
        loader: PriceLoader.StkPrice = PriceLoader.StkPrice()
        # codelist에 있는 종목들의 공분산행렬을 가져옴
        self.covlist = loader.getCov(self.codelist)
        weights = np.array([1.0 / len(self.covlist)] * len(self.covlist))
        bound = [(0, 1) for i in range(len(self.covlist))]
        params = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = minimize(objfunc, weights, (self.covlist), method='SLSQP', bounds=bound, constraints=params)
        logger.debug(res)
        logger.debug(res.x)
        return res.x