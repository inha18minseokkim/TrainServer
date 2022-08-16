import FinanceDataReader as fdr
import asyncio
import numpy as np
import pandas


class StkPrice:
    async def getPrice(self, code: str):
        res = fdr.DataReader(code)['Close']
        return res
    async def getPriceList(self,li : list):
        res = await asyncio.gather(*[self.getPrice(code) for code in li])
        #print(res)
        return res
    async def getMeanStd(self, code : str):
        price: pandas.Series = await self.getPrice(code)
        rateofreturn = price.apply(np.log) - price.apply(np.log).shift(20)
        #print(rateofreturn)
        rateofreturn.dropna(inplace=True)
        rateofreturn = np.array(rateofreturn)
        return [np.mean(rateofreturn),np.std(rateofreturn)]

    async def getMeanStdList(self, code : list):
        price: list(pandas.Series) = await self.getPriceList(code)
        li = []
        for df in price:
            tmpreturn = df.apply(np.log) - df.apply(np.log).shift(20)
            tmpreturn.dropna(inplace=True)
            tmpreturn = np.array(tmpreturn)
            li.append([np.mean(tmpreturn),np.std(tmpreturn)])
        return li

if __name__ == "__main__":
    loader: StkPrice = StkPrice()
    res = asyncio.run(loader.getMeanStdList(['005930','091160','091170']))
    print(res)