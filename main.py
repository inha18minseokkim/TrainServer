import threading
import nest_asyncio
from fastapi import FastAPI,Request

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
    stklist = request.headers.get('stklist').split(',')
    quantity = int(request.headers.get('quantity'))
    res = scheduler.newElement(kakaoid,stklist,strategy,quantity)
    return res
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
