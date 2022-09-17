import time
import datetime
from dependency_injector import containers
from loguru import logger
import DBManager
import Executor



class PredictScheduler: #싱글톤 패턴으로 구현 routine은 스레드로 돌려
    def __init__(self, _serverdb: DBManager.ServerDBManager):
        self.queue: list(list(str,datetime.datetime)) = [] #카카오아이디가 들어가있는 큐
        #카카오아이디와 맨 마지막에 execute를 했떤 날짜를 넣어. 만약 head의 시간이 아직 하루가 지난게 아니면 예측하지말고 기다려.
        self.serverdb: DBManager.ServerDBManager = _serverdb
        self.curexecutor: Executor.Executor = Executor.Executor(self.serverdb)
    def logqueue(self):
        logger.debug("스케쥴러 큐의 내용 출력")
        for id,time in self.queue:
            logger.debug(f"{id} {time}")
        logger.debug("출력완료")
    def load_queue(self): #서버 재시작되면 큐를 복원
        pass
    def save_queue(self): #서버가 날아갔을 때를 대비해서 현재 큐 상태를 db에 저장해놔야됨
        pass #뭐 일단..당분간은 필요없을듯
    def routine(self): #주기적으로 실행하는 함수.
        while True:
            if len(self.queue) == 0:
                logger.debug('큐가 비어서 그냥 지나침')
                time.sleep(10)
                continue
            kakaoid, settime = self.queue[0]
            curtime = datetime.datetime.now()
            dt: datetime.timedelta = curtime - settime
            logger.debug(f'curtime{curtime}, dt {dt}')
            stddelta: datetime.timedelta = datetime.timedelta(days=1) #기준일은 1일
            if dt < stddelta: #현재 시간 간극이 기준 간격보다(stddelta) 짧다 -> 트레이딩 한지 얼마 안됐다.
                logger.debug(f'헤더 가 아직 하루가 안되었기 때문에 {stddelta}만큼 쉰다')
                time.sleep(stddelta.total_seconds()) #dt만큼 일단 스레드 비활성화.
            #하루 지났으면 실행해
            else:
                self.queue.pop(0)
                self.curexecutor.execute(kakaoid)
                self.queue.append([kakaoid,datetime.datetime.now()]) #큐 맨끝에 넣어
    def searchElement(self,kakaoid: str) -> int:#선형 탐색으로 인덱스 반환
        for i in range(len(self.queue)):
            if self.queue[i][0] == kakaoid:
                logger.debug(f'{kakaoid} : {i} 인덱스 찾음 반환')
                return i
        logger.debug(f'{kakaoid} : 인덱스 못찾음')
        return -1
    def delElement(self,kakaoid: str) -> int:
        idx: int = self.searchElement(kakaoid)
        if idx == -1:
            logger.debug(f"찾는 인덱스 없음 : {kakaoid}")
            return -1
        del self.queue[idx]
        logger.debug(f"{kakaoid} 큐에서 삭제 완료")
        self.save_queue()
        logger.debug(f"{kakaoid} 큐에서 삭제 후 db저장 완료")
        return 0
    def newElement(self, kakaoid: str, stklist: list, strategy: str, quantity: int): #프론트에서 리퀘스트를 받으면 얘로 큐에 넣어.
        #serverdb에 일단 정보를 셋팅해야됨.
        req: dict = {i : 0 for i in stklist}
        self.serverdb.setStockRatio(kakaoid,req)
        self.serverdb.setStrategy(kakaoid,strategy)
        self.serverdb.setQuantity(kakaoid,quantity)
        #서버에 필요한 정보가 다 반영됨. 이제 execute하고 큐에 넣으면됨
        res = self.curexecutor.execute(kakaoid)
        #큐에 넣는 과정. 현재 큐에 중복된 아이디가 없으면 그냥 넣고 있으면 넣지마
        if self.searchElement(kakaoid) == -1:
            logger.debug(f"{kakaoid} 삽입 시도")
            self.queue.append([kakaoid, datetime.datetime.now()])

        return res