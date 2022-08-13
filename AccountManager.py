import DBManager
from loguru import logger
import requests,json
import threading
import uuid
Base_URL = "https://openapivts.koreainvestment.com:29443"

class Account:
    def __init__(self, kakaoid: str, _serverdb):
        self.state = 0
        self.serverdb = _serverdb
        self.userinfo = self.serverdb.getUserInfoFromServer(kakaoid)
        if self.userinfo['code'] == 0:
            self.state = 0
            logger.debug(f'Account : {kakaoid} 에 대한 계정정보 서버에서 불러오기 실패')
            return
        logger.debug(self.userinfo)
        self.kakaoid = self.userinfo[DBManager.KAKAOID]
        self.nickname = self.userinfo[DBManager.NICKNAME]
        self.apikey = self.userinfo[DBManager.APIKEY]
        self.secret = self.userinfo[DBManager.SECRET]
        self.quantity = self.userinfo[DBManager.QUANTITY]
        self.cano = self.userinfo[DBManager.CANO]
        self.acnt = self.userinfo[DBManager.ACNT]
        self.period = self.userinfo[DBManager.PERIOD]
        self.favlist = self.userinfo[DBManager.FAVLST]
        self.curpricedic: dict = {}
        #ratio는 내가 설정해놓은 비율
        self.ratio: dict = self.serverdb.getStockRatio(self.kakaoid)
        logger.debug(self.kakaoid)
        logger.debug(self.ratio)
        self.total = 0 #총평가금액
        self.deposit = 0 #예수금총금액
        self.eval = 0 #유가평가금액
        self.sumofprch = 0 #매입금액합계금액
        self.sumofern = 0 #평가손익합계금액
        self.assticdc = 0 #자산증감액
        self.assticdcrt = 0.0 #자산증감수익률
        self.curaccount: list = []


        #kakaoid에 해당하는 token을 한국투자에서 가져오기 위한 주문
        headers = {"content-type": "application/json"}
        body = {"grant_type": "client_credentials",
                "appkey": self.apikey,
                "appsecret": self.secret}
        path = "oauth2/tokenP"
        url = f"{Base_URL}/{path}"
        logger.debug(f"{url}로 보안인증 키 요청")
        tokenres = requests.post(url, headers=headers, data=json.dumps(body)).json()
        self.token = tokenres['access_token']
        logger.debug(f"token 생성 완료, 현재 계정 정보 {self.kakaoid},{self.token}")
#       token을 가져왔으니 한국투자 api에 연결해서 세부 잔고정보도 가져옴.
        self.getcurAccountInfo()

        logger.debug(self.curpricedic)
        logger.debug('AccountManager 인스턴스 생성완료')
        logger.debug(self.curaccount)
        # logger.debug(self.total,self.deposit,self.eval,self.sumofprch,self.sumofern,self.assticdc,self.assticdcrt)
        self.state = 1

    def getcurAccountInfo(self): #현재 내 잔고 현황 가져와서 딕셔너리 형태로
        #잔고 현황을 가져오는 것
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt,
            "AFHR_FLPR_YN": "N", #시간외단일가:  ㄴㄴ 그냥 현재가로
            "OFL_YN": "N", #오프라인 여부
            "INQR_DVSN": "02", #조회구분 종목별
            "UNPR_DVSN": "01", #단가구분
            "FUND_STTL_ICLD_YN": "N", #펀드결제분 포함하지 않음
            "FNCG_AMT_AUTO_RDPT_YN": "N", #융자금액자동상환여부
            "PRCS_DVSN": "00", #전일매매포함
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        headers = {
            "content-type": 'application/json',
            "authorization": f"Bearer {self.token}",
            "appKey": self.apikey,
            "appSecret": self.secret,
            "tr_id": "VTTC8434R",
            "custtype": "P",
        }
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{Base_URL}/{path}"
        res = requests.get(url, headers=headers, params=params).json()
        res1 = res['output1']
        res2 = res['output2'][0]
        logger.debug(res2)
        self.curaccount = []
        logger.debug(self.curpricedic)
        for column in res1:
            node = {'pdno': column['pdno'], 'prdt_name':column['prdt_name'], 'hldg_qty':int(column['hldg_qty']),'pchs_avg_pric':float(column['pchs_avg_pric']),
                    'pchs_amt' : int(column['pchs_amt']),
                    'prpr': int(column['prpr']), 'evlu_amt' : int(column['evlu_amt']),'evlu_pfls_amt': int(column['evlu_pfls_amt']),'evlu_pfls_rt': float(column['evlu_pfls_rt'])}
            self.curaccount.append(node)
            self.curpricedic[column['pdno']] = int(column['prpr'])
        #pdno 종목코드
        #prdt_name 종목명
        #hldg_qty 보유수량
        #pchs_avg_pric 매입평균가격
        #pchs_amt 매입금액
        #prpr 현재가
        #evlu_amt 평가금액
        #evlu_pfls_amt 평가손익금액
        #evlu_pfls_rt 평가손익율
        self.total = int(res2['tot_evlu_amt'])  # 총평가금액
        self.deposit = int(res2['dnca_tot_amt'])  # 예수금총금액
        self.eval = int(res2['scts_evlu_amt']) # 유가평가금액
        self.sumofprch = int(res2['pchs_amt_smtl_amt'])  # 매입금액합계금액
        self.sumofern = int(res2['evlu_pfls_smtl_amt'])  # 평가손익합계금액
        self.assticdc = int(res2['asst_icdc_amt'])  # 자산증감액
        self.assticdcrt = float(res2['asst_icdc_erng_rt'])  # 자산증감수익률
    def getAccountInfoDictionary(self) -> dict:
        return {
            'state': self.state,
            'kakaoid': self.kakaoid,
            'nickname': self.nickname,
            'apikey': self.apikey,
            'secret': self.secret,
            'quantity': self.quantity,
            'cano': self.cano,
            'acnt': self.acnt,
            'curpricedic': self.curpricedic,
            'ratio': self.ratio,
            'total': self.total,
            'deposit': self.deposit,
            'eval': self.eval,
            'sumofprch': self.sumofprch,
            'sumofern': self.sumofern,
            'assticdc': self.assticdc,
            'assticdcrt': self.assticdcrt,
            'curaccount': self.curaccount,
            'token': self.token,
            'period' : self.period,
            'favlist' : self.favlist
        }

if __name__ == "__main__":
    serdb: DBManager.ServerDBManager = DBManager.ServerDBManager()
    ac: Account = Account('12181577',serdb)
    ac.getAccountInfoDictionary()