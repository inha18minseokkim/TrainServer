import threading
import nest_asyncio
from fastapi import FastAPI, Request, requests
from loguru import logger

import DBManager
import Scheduler
from Container import MainContainer
app = FastAPI()
mc : MainContainer = MainContainer()
scheduler: Scheduler.PredictScheduler = None
nest_asyncio.apply()
@app.on_event("startup")
async def on_app_start() -> None:
    global trc,scheduler
    trc = MainContainer()
    serverdb : DBManager.ServerDBManager = trc.serverdb()
    scheduler = trc.scheduler()
    schedulerThread = threading.Thread(target = scheduler.routine)
    schedulerThread.start()

@app.post("/submituserInfo")
async def insertIntoqueue(request: Request):
    kakaoid = request.headers.get('kakaoid')
    strategy = request.headers.get('strategy')
    stklist = request.headers.get('stklist')
    quantity = request.headers.get('quantity')
    if kakaoid == None or strategy == None or stklist == None or quantity == None:
        return {'code' : 0, 'msg' : 'None값을 받음'}
    logger.debug(f'{kakaoid} {strategy} {stklist} {quantity}')
    stklist = stklist.split(',')
    quantity = int(quantity)
    res = scheduler.newElement(kakaoid,stklist,strategy,quantity)
    logger.debug(f"{kakaoid}")
    scheduler.logqueue()
    #메인서버로 post 리퀘스트 보내면됨 아래는 메인서버 리퀘스트
    # @router.post('/onTrainComplete')
    # async def onTradeComplete(request: Request):
    #     kakaoid = request.get('kakaoid')
    #     logger.debug(kakaoid)
    #     scheduler = await get_scheduler()
    #     res = scheduler.addNewAccount(kakaoid)
    #     return res

    #header = {"kakaoid" : kakaoid}
    #requests.post(url="http://haniumproject.com:8000/onTrainComplete",headers = header)
    return res
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/getQueueInfo")
async def getQ():
    scheduler.logqueue()

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
