from loguru import logger

import Container
import DBManager
import Declaration
from Strategies.BruteForceStrategy import BruteForceStrategy
from Strategies import Strategy
from Train import Trainer


class Executor:
    def __init__(self, serverdb: DBManager.ServerDBManager):
        self.serverdb = serverdb

    def decideModel(self,code: str):
        modelname = self.serverdb.getStockModel(code)
        if modelname == Declaration.DefaultModel:
            return Container.TrainContainer.defaultPredict()
    def decideStrategy(self,strategyName: str, codelist: list, rtnstd: list) -> Strategy.Strategy:
        if strategyName == 'BruteForceStrategy':
            return BruteForceStrategy(codelist, rtnstd)

        return None
    def execute(self,kakaoid: str):#kakaoid 유저의 주식정보를 받아서 해당 모델에 넣고 수익률 도출->도출된 수익률로 전략-> 최종 비율 db에 반영
        curUserStockList: list = self.serverdb.getUserStockList(kakaoid) #현재 유저가 고른 종목 리스트
        #해당 종목 코드로 할 일
        #종목에 맵핑되어있는 모델을 깨움
        #모델에 넣고 예측값을 가져옴-> 여기까지는 Trainer 클래스가 관여하는것
        curUserModelList: list(Trainer.Trainer) = [self.decideModel(c) for c in curUserStockList]
        print(curUserModelList)
        curUserRTNSTD: list = [curUserModelList[i].pred(curUserStockList[i]) for i in range(len(curUserStockList))]
        #curUserRTNSTD -> 각 종목의 예상수익률, 표준편차 들어있음. Trainer 클래스로 캐스팅 되어있기 때문에 간단하게 통일성있는 한줄로 작성가능.
        print(curUserRTNSTD)
        #예측값가지고 뭘 함 -> 여기서부터는 Strategy가 관여함
        curUserStrategyName = self.serverdb.getStrategy(kakaoid)
        curstrategy = self.decideStrategy(curUserStrategyName,curUserStockList,curUserRTNSTD)
        curratio = curstrategy.execute()
        res: dict = {curUserStockList[i]: curratio[i] for i in range(len(curUserStockList))}
        logger.debug(f'{curratio}   {res}')
        self.serverdb.setStockRatio(kakaoid,res)

