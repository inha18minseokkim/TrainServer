import numpy as np
import PriceLoader
from Strategies.Strategy import Strategy


class BruteForceStrategy(Strategy):
    def __init__(self, codelist: list[str], rtnstdlist: list):
        self.codelist = codelist
        # rtnstdlist에서 수익률만 빼와
        self.rtnlist = [rtn for [rtn,std] in rtnstdlist]
    def execute(self):
        loader: PriceLoader.StkPrice = PriceLoader.StkPrice()
        # 1. codelist에 있는 종목들의 공분산행렬을 가져옴
        self.covlist = loader.getCov(self.codelist)
        #2. 점을 10만개 찍어
        epoch = 100000
        krxrtn = loader.getKoreaBondRtn() #연수익률을 가져와서 월수익률로 바꿈
        krxrtn = np.exp(krxrtn/12) - 1 #문제는 우리는 로그수익률을 사용하기 때문에 여기도 로그수익률로 변환해줌

        rtnlist = []
        vollist = []
        weightlist = []
        for i in range(epoch):
            weight = np.random.random(len(self.codelist))
            weight /= np.sum(weight)  # 가중치 합 1로 만듬
            ret = np.dot(weight, self.rtnlist)
            vol = np.sqrt(np.dot(weight.T, np.dot(self.covlist, weight)))
            rtnlist.append(ret)
            vollist.append(vol)
            weightlist.append(weight)
        #3. 찍은 점 중 sharpe ratio가 최대인 인덱스를 갖고와서 weight를 찍어
        rtnlist = np.array(rtnlist)
        vollist = np.array(vollist)
        sharpelist = (rtnlist - krxrtn) / vollist  # 국고채 1년을 1개월 기준으로 바꾼걸 ratio 구함
        res = weightlist[sharpelist.argmax()]

        return res