from dependency_injector import containers, providers
from loguru import logger

import DBManager
import Scheduler
from Train import DefaultTimeSeriesPredict


class MainContainer(containers.DeclarativeContainer):
    def __new__(cls):
        if not hasattr(cls,'instance'):
            logger.debug("Container 싱글톤 객체 만듬")
            cls.instance = super(MainContainer,cls).__new__(cls)
        else:
            logger.debug('객체 만들어져있어서 주솟값만 리턴함')
        return cls.instance

    serverdb : DBManager.ServerDBManager = providers.Singleton(DBManager.ServerDBManager)
    scheduler : Scheduler.PredictScheduler = providers.Singleton(Scheduler.PredictScheduler, _serverdb = serverdb)
