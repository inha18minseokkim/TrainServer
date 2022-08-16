import asyncio
import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from loguru import logger
import FinanceDataReader as fdr

import Declaration
import PriceLoader
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import torch.nn.functional as F

import Trainer

CURPATH = Declaration.ModelPATH + '/DefaultModel.pth'
class fund_GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(fund_GRU, self).__init__()

        self.hidden_size = hidden_size
        self.daily_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.weekly_rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.monthly_rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, daily_input, weekly_input, monthly_input):
        # packed_input = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        out, daily_hidden = self.daily_rnn(daily_input, self.daily_hidden)
        out, weekly_hidden = self.weekly_rnn(weekly_input, self.weekly_hidden)
        out, monthly_hidden = self.monthly_rnn(monthly_input, self.monthly_hidden)
        merged_hidden = torch.cat((daily_hidden[0], weekly_hidden[0], monthly_hidden[0]), 1)
        # logits = self.classifier(hidden[0].view(hidden[0].size(0), -1))
        logits = self.classifier(merged_hidden)

        return logits

    def init_hidden(self, curr_batch):
        self.daily_hidden = torch.zeros(1, curr_batch, self.hidden_size).to('cuda:0')
        self.weekly_hidden = torch.zeros(1, curr_batch, self.hidden_size).to('cuda:0')
        self.monthly_hidden = torch.zeros(1, curr_batch, self.hidden_size).to('cuda:0')


class DefaultPredict(Trainer.Trainer):
    def __init__(self,batch_size: int, load: bool = False):
        self.priceloader: PriceLoader.StkPrice = PriceLoader.StkPrice()
        self.batch_size = batch_size
        self.model = fund_GRU(1,128)
        try:
            if load == True: #모델을 로드하는거면 로딩
                self.load()
                self.model.eval()
        except FileNotFoundError: #처음 시작하는거면 파일이 없을 수도 있음.
            pass
        #self.dataloader = self.data_prepro()

    def data_prepro(self, traincode: list = []): #raw data를 batchsize에 맞게 변경
        self.pricelist: list = asyncio.run(self.priceloader.getPriceList(traincode))
        self.traincode = traincode
        resdaily = []
        resweekly = []
        resmonthly = []
        reslabel = []
        for df in self.pricelist:
            daily = df.apply(np.log) - df.apply(np.log).shift(1)
            weekly = df.apply(np.log) - df.apply(np.log).shift(5)
            monthly = df.apply(np.log) - df.apply(np.log).shift(20)
            print(len(daily))
            label = monthly.shift(-1)
            tmp:pd.DataFrame = pd.concat([daily,weekly,monthly,label],axis=1)
            tmp.dropna(inplace=True)
            tmp.columns = ['daily','weekly','monthly','label']
            # data를 15영업일 단위로 windowing 함
            for i in range(len(tmp) - 15):
                resdaily.append(tmp.iloc[i:i + 15].loc[:, 'daily'])
                resweekly.append(tmp.iloc[i:i + 15].loc[:, 'weekly'])
                resmonthly.append(tmp.iloc[i:i + 15].loc[:, 'monthly'])
                reslabel.append(tmp.iloc[i:i + 15].loc[:, 'label'])
        resdaily = torch.FloatTensor(resdaily)
        resweekly = torch.FloatTensor(resweekly)
        resmonthly = torch.FloatTensor(resmonthly)
        reslabel = torch.FloatTensor(reslabel)
        dataset = torch.utils.data.TensorDataset(resdaily, resweekly, resmonthly, reslabel)
        #batchsize로 dataloader 사용
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    def train(self):
        self.model.train() #트레인모드로
        self.model.to('cuda:0') #gpu 사용
        dataloader = self.data_prepro(['005930','213500','091160'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        epochs = 10
        lossfunc = torch.nn.MSELoss()
        for e in range(epochs + 1):
            avg_loss = []
            print('epoch', e)
            for idx, (daily_input, weekly_input, monthly_input, label) in enumerate(dataloader):
                #logger.debug(idx)
                # input data의 차원이 (256,15)라서 이걸 (256,15,1)로 바꿔주는 unsqueeze(2)
                # to('cuda:0') -> cpu에 있던 텐서를 gpu로 옮김
                daily_input = daily_input.to('cuda:0').unsqueeze(2)
                weekly_input = weekly_input.to('cuda:0').unsqueeze(2)
                monthly_input = monthly_input.to('cuda:0').unsqueeze(2)
                # epoch마다 batch size가 달라질 수 있기 때문에(특히 마지막) 항상 초기화해줘야됨
                self.model.init_hidden(daily_input.shape[0])
                optimizer.zero_grad()
                label = label.to('cuda:0')
                # print(daily_input.unsqueeze(2).shape)
                tmp = self.model(daily_input, weekly_input, monthly_input)
                loss = lossfunc(tmp, label)
                # backpropagation
                loss.backward()
                optimizer.step()
                # mseloss를 epoch별로 plotting하기 위해 리스트에 저장
                avg_loss.append(loss.item())
        print(avg_loss)
        self.save()

    def save(self):
        try:
            torch.save(self.model,CURPATH)
        except:
            os.mkdir('./Model')
            torch.save(self.model,CURPATH,map_loaction=torch.device('cuda:0'))
    def load(self):
        self.model = torch.load(CURPATH,map_loaction=torch.device('cuda:0'))

    def test_dataload(self, code: str) -> torch.utils.data.TensorDataset:
        #financeDataReader에서 학습할 주식종목에 대한 데이터 로드해서 가져옴
        dftest = fdr.DataReader(code).iloc[-60:] #최근 60영업일 데이터 가져옴
        dftest = dftest['Close']

        dailytest = dftest.apply(np.log) - dftest.apply(np.log).shift(1)
        weeklytest = dftest.apply(np.log) - dftest.apply(np.log).shift(5)
        monthlytest = dftest.apply(np.log) - dftest.apply(np.log).shift(20)
        labeltest = monthlytest.shift(-1)

        samsungtest = pd.concat([dailytest, weeklytest, monthlytest, labeltest], axis=1)
        samsungtest.dropna(inplace=True)
        samsungtest.columns = ['daily', 'weekly', 'monthly', 'label']
        testdaily = []
        testweekly = []
        testmonthly = []
        testlabel = []
        for i in range(len(samsungtest) - 15):
            testdaily.append(samsungtest.iloc[i:i + 15].loc[:, 'daily'])
            testweekly.append(samsungtest.iloc[i:i + 15].loc[:, 'weekly'])
            testmonthly.append(samsungtest.iloc[i:i + 15].loc[:, 'monthly'])
            testlabel.append(samsungtest.iloc[i:i + 15].loc[:, 'label'])

        testdaily = torch.FloatTensor(testdaily).to('cuda:0')
        testweekly = torch.FloatTensor(testweekly).to('cuda:0')
        testmonthly = torch.FloatTensor(testmonthly).to('cuda:0')
        testlabel = torch.FloatTensor(testlabel).to('cuda:0')
        dataset = torch.utils.data.TensorDataset(testdaily, testweekly, testmonthly, testlabel)
        return dataset

    def pred(self,code: str): #한 가지 종목에 대해 예측, 평균과 표준편차 넘겨줌
        self.model.to(torch.device('cuda:0'))
        self.model.eval() #backpropagation 없이 평가만
        testset = self.test_dataload(code)
        res = []
        with torch.no_grad():
            for daily_input, weekly_input, monthly_input, label in testset:
                daily_input = daily_input.to('cuda:0').unsqueeze(0).unsqueeze(2)
                weekly_input = weekly_input.to('cuda:0').unsqueeze(0).unsqueeze(2)
                monthly_input = monthly_input.to('cuda:0').unsqueeze(0).unsqueeze(2)
                label = label.to('cuda:0')
                # testset은 batchsize가 1이기 때문에 hidden state 초기화 안해주면 차원이 안맞음
                self.model.init_hidden(daily_input.shape[0])
                # print(daily_input.shape)
                pred = self.model(daily_input, weekly_input, monthly_input)
                print(pred[0])
                res.append(pred.to('cpu').squeeze().item())
        res = np.array(res).reshape(-1)
        print(res)
        return [np.mean(res),np.std(res)]

if __name__ == '__main__':
    pred: DefaultPredict = DefaultPredict(512)
    print(pred.pred('005930'))
    #pred.train()
