import Container
import DBManager
import Declaration


class Executor:
    def __init__(self, serverdb: DBManager.ServerDBManager):
        self.serverdb = serverdb

    def decideModel(self,code: str):
        modelname = self.serverdb.getStockModel(code)
        if modelname == Declaration.DefaultModel:
            return Container.TrainContainer.defaultPredict()

        return None
    def execute(self,kakaoid: str):
        curUserStockList: list = self.serverdb.getUserStockList(kakaoid) #현재 유저가 고른 종목 리스트
        #해당 종목 코드로 할 일
        #종목에 맵핑되어있는 모델을 깨움
        #모델에 넣고 예측값을 가져옴-> 여기까지는 Trainer 클래스가 관여하는것
        curUserModelList: list = [self.decideModel(c) for c in curUserStockList]
        print(curUserModelList)

        #예측값가지고 뭘 함 -> 여기서부터는 Strategy 클래스가 관여함

