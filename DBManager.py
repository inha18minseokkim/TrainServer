import pymongo
from loguru import logger
from ast import literal_eval

import Declaration

KAKAOID = 'kakaoid'
SESSIONID = 'sessionid'
UUID = 'uuid'
LOGINTOKEN = 'logintoken'
NICKNAME = 'nickname'
APIKEY = 'apikey'
SECRET = 'secret'
QUANTITY = 'quantity'
MODEL = 'model'
TOKEN = 'token'
CANO = 'cano'
ACNT = 'acnt_prdt_cd'
STKLST = 'stocklist'
PERIOD = 'period'
FAVLST = 'favlist'
class ServerDBManager:
    def __init__(self):
        self.client = pymongo.MongoClient(  # 실제 배포에서는 아래거 써야됨.
            f"mongodb+srv://admin1:admin@cluster0.qbpab.mongodb.net/?retryWrites=true&w=majority")
        # client = pymongo.MongoClient(
        #     f"mongodb+srv://admin1:{Declaration.serverDBPW}@cluster0.qbpab.mongodb.net/?retryWrites=true&w=majority")
        self.serverdb = self.client.TradingDB
        logger.debug("serverdb 초기화 완료", self.serverdb)

    def getUserInfoFromServer(self, kakaoid: str):
        # {'_id': ObjectId('62c3ed0a991191142d3d56fc'), 'kakaoid': '12181577',
        # 'nickname': '김민석', 'apikey': 'asdf', 'secret': 'sec', 'quantity': 1000000, 'code': 1} ->kakaoid가 있을 경우
        # {'code': 0} -> kakaoid가 없을 경우
        cursor = self.serverdb.user.find({KAKAOID: kakaoid})
        res = list(cursor)
        if len(res) == 0:  # 정보가 없으면 0을 리턴
            logger.debug(f'{kakaoid} 에 해당하는 정보가 없음')
            return {'code': 0 ,'msg' :  f'{kakaoid} 에 해당하는 정보가 없음'}
        res = res[0]
        res['code'] = 1
        return res
    def setStockRatio(self,kakaoid: str, target: dict):
        idquery = {KAKAOID : kakaoid}
        value = {'$set' : { STKLST : target}}
        self.serverdb.user.update_one(idquery,value)

    def getStockRatio(self, kakaoid: str):  # kakaoid유저가 설정해놓은 주가 비율을 가져옴
        cursor = self.serverdb.user.find({KAKAOID: kakaoid})
        res = list(cursor)
        if len(res) == 0:  # 정보가 없으면 0을 리턴
            return {'code': 0}
        try:
            tmp = res[0][STKLST]
        except:
            logger.debug('아직 비율이 설정되지 않음. 빈 리스트를 만듦')
            idquery = {KAKAOID : kakaoid}
            values = {"$set" : {STKLST : {}}}
            self.serverdb.user.update_one(idquery,values)
            return {}
        res = tmp
        res['code'] = 1
        logger.debug('kakaoid에 대한 주가 비율 정보를 요청함', res)
        return res
    def getUserStockList(self, kakaoid: str) -> list:
        tmp = self.getStockRatio(kakaoid)
        del tmp['code']
        return [k for k in tmp.keys()]

    def getScheduler(self): #Scheduler의 정보 가져옴
        cursor = self.serverdb.scheduler.find()
        res: list[list[(str, int)]] = list(cursor)
        return res
    def setSchedulerIdx(self, _idx):
        idquery = {'idx' : 'idx'}
        values = {'$set' : {'value' : _idx}}
        self.serverdb.scheduleridx.update_one(idquery,values)
        return
    def getSchedulerIdx(self):
        cursor = self.serverdb.scheduleridx.find({'idx' : 'idx'})
        return list(cursor)[0]['value']

    def getStockModel(self,code: str):
        try:
            cursor = self.serverdb.stockmodel.find({'code' : code})
            res = list(cursor)
            return res[0]['modelname']
        except: #db안에 아직 모델에 대한 정보가 없으면 db에 default로 넣어
            self.serverdb.stockmodel.insert_one({'code':code,'modelname':Declaration.DefaultModel})
            return Declaration.DefaultModel



    def getModelInfo(self):
        cursor: list[(str,float,float,str)] = list(self.serverdb.modelinfo.find())
        return cursor