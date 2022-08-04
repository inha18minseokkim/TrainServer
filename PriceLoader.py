import FinanceDataReader as fdr
import asyncio

class StkPrice:
    async def getPrice(self, code: str):
        res = fdr.DataReader(code)
        return res
    async def getPriceList(self,li : list):
        res = await asyncio.gather(*[self.getPrice(code) for code in li])
        print(res)
        return res

if __name__ == "__main__":
    loader: StkPrice = StkPrice()
    print("X")
    res = asyncio.run(loader.getPriceList(['005930','003550','091160']))
    print("A")
    print(res)