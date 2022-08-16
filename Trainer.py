

#주가를 예측하는 인공지능 모듈들의 최상위 클래스. 사실상 인터페이스
class Trainer:
    def __init__(self): #생성자 별거없음
        pass
    def data_prepro(selfself,traincode: list): #훈련데이터를 전처리하는 함수
        pass
    def train(self): #훈련을 진행하는 함수
        pass
    def save(self): #모델을 저장하는 함수
        pass
    def load(self): #모델을 저장공간에서 로딩하는 함수
        pass
    def test_dataload(self, code: str): #테스트데이터를 전처리하는 함수
        pass
    def pred(self, code: str): #code에 대한 예측을 진행하는 함수
        pass
