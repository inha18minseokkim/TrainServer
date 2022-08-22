from dependency_injector import containers, providers
from Train import DefaultTimeSeriesPredict


class TrainContainer(containers.DeclarativeContainer):
    defaultPredict : DefaultTimeSeriesPredict.DefaultPredict \
        = providers.Singleton(DefaultTimeSeriesPredict.DefaultPredict, batch_size=512, load=True)