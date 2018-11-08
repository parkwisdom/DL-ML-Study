########################################## recommender system(추천 시스템) ##############################################
#by collaborative filtering(협업 필터링)
#목적: 유사한 사람 찾기, 추천하기...
#구분 :
# 1)사용자 기반 필터링
# 2)아이템 기반 필터링:

import matplotlib as mpl
#1. 한글 깨짐 방지
mpl.rcParams['axes.unicode_minus']=False
from matplotlib import font_manager, rc
###################################
import matplotlib.pyplot as plt
from math import sqrt #제곱근  #유클리디안 거리
font_name=font_manager.FontProperties(fname='C:\Windows\Fonts/malgun.ttf').get_name() #malgun.ttf 폰트 설정하기
rc('font',family=font_name)

#데이터 가져와 재구성하는 부분 생략함

#2. 데이터 가져오기
critics={
    'BTS':{'암수살인':5, '바울':4, '할로윈':1.5},
    '손흥민':{'바울':5, '할로윈':2},
    '조용필':{'암수살인':2.5, '바울':2, '할로윈':1},
    '나훈아':{'암수살인':3.5,'바울':4, '할로윈':5}
}
print(critics.get('BTS').get('바울')) #{중요}

#3. 유클리디안 거리
# print(pow(3,2)) #3의 제곱
def sim(i,j): #유사도 #전달된 i와 의 2 데이터의 유사도를 리턴하는 함수-i와 j의 거리
    #i:x2-x1, j:y2-y1이 전달됨
    return sqrt(pow(i,2)+pow(j,2)) #1번째 데이터의 제곱값 + 2번째 데이터의 제곱값
#손흥민과 나훈아 사이의 거리를 구하고 싶다
#피타고라스 정리->거리가 가까울수록 유사도가 높다.
# print(critics['손흥민'])

#4. 유사도 측정하기
var1=critics['손흥민']['바울']-critics['나훈아']['바울']
var2=critics['손흥민']['할로윈']-critics['나훈아']['할로윈']
print(sim(var1,var2)) #3.1622776601683795 # 이 수치가 낮을 수록 유사도 높다

print("손흥민 기준으로 다른 사람과의 유사도 측정")
for i in critics:
    if i !='손흥민':
        var1=critics['손흥민']['바울']-critics['나훈아']['바울']
        var2=critics['손흥민']['할로윈']-critics['나훈아']['할로윈']
        print(i,"와 손흥민의 유사도:",1/(1+sim(var1,var2)))


#두점 사이의 거리
#항목(영화) 데이터가 2종류(두 편)인 경우 -> 피타고라스 공식
#항목(영화) 데이터가 여러종류(여러편)->유클리디안 거리

print('-'*40)
#유클리디안 거리 기반 두 데이터 사이의 거리
def sim_distance(data,name1,name2):
    sum=0
    for i in data[name1]:
        if i in data[name2]: # 같은 영화를 봤다면
            sum+=pow(data[name1][i]-data[name2][i],2)
    return 1/(1+sqrt(sum))
        # print(i)

print(sim_distance(critics,'손흥민','나훈아'))

# 손흥민과 나머지 전체 관객과의 평점간 거리(유클리디안)
def matchf(data,name, idx=3,sim=sim_distance): #sim=sim_distance 만들어둔 함수 불러옴
    myList=[]
    for i in data:
        if i !=name:#본인이 아닌경우라면
            myList.append((sim(data, name,i),i)) #유사도, 상대이름
            myList.sort()
            print("정렬: ",myList)
            myList.reverse()
            print("역순: ",myList)
    return myList[:idx]

print(matchf(critics,'손흥민'))
#손흥민과 나머지 전체 관객과의 평점간 거리를 내림차순 정렬
li=matchf(critics,'손흥민')
print(li)

def barchart(data,labels): #유사도(손흥민과의), 이름(손흥민을 제외)
    positions=range(len(data))
    plt.barh(positions,data,height=0.5,color='r') #barh:수평바(y축, x축)
    plt.yticks(positions,labels)
    plt.xlabel('simlarity')
    plt.ylabel('name')
    # plt.show()
    # print(positions)
score=[]
names=[]
for i in li:
    score.append(i[0])
    names.append(i[1])
barchart(score,names)

##----------------------------------------------

# plt.figure(figsize=(14,8)) #인치
# plt.plot([1,2,3],[1,2,3],'g^')
# plt.text(1,1,'자동차')
# plt.text(2,2,'버스')
# plt.text(3,3,'열차')
# plt.axis([0,6,0,6]) #축크기 재설정
# plt.show()

## --------------------------------------


critics = {
    '조용필': {
        '택시운전사': 2.5,
        '겨울왕국': 3.5,
        '리빙라스베가스': 3.0,
        '넘버3': 3.5,
        '사랑과전쟁': 2.5,
        '세계대전': 3.0,
    },
    'BTS': {
        '택시운전사': 1.0,
        '겨울왕국': 4.5,
        '리빙라스베가스': 0.5,
        '넘버3': 1.5,
        '사랑과전쟁': 4.5,
        '세계대전': 5.0,
    },
    '강감찬': {
        '택시운전사': 3.0,
        '겨울왕국': 3.5,
        '리빙라스베가스': 1.5,
        '넘버3': 5.0,
        '세계대전': 3.0,
        '사랑과전쟁': 3.5,
    },
    '을지문덕': {
        '택시운전사': 2.5,
        '겨울왕국': 3.0,
        '넘버3': 3.5,
        '세계대전': 4.0,
    },
    '김유신': {
        '겨울왕국': 3.5,
        '리빙라스베가스': 3.0,
        '세계대전': 4.5,
        '넘버3': 4.0,
        '사랑과전쟁': 2.5,
    },
    '유성룡': {
        '택시운전사': 3.0,
        '겨울왕국': 4.0,
        '리빙라스베가스': 2.0,
        '넘버3': 3.0,
        '세계대전': 3.5,
        '사랑과전쟁': 2.0,
    },
    '이황': {
        '택시운전사': 3.0,
        '겨울왕국': 4.0,
        '세계대전': 3.0,
        '넘버3': 5.0,
        '사랑과전쟁': 3.5,
    },
    '이이': {'겨울왕국': 4.5, '사랑과전쟁': 1.0,
             '넘버3': 4.0},
}

#삼관분석
"""
유클리디안 거리 공식의 한계점:
특정인의 점수가 극단적으로 높거나 낮다면 제대로 된 결과를 도출해 내기가 어렵다.
"""
def drawGraph(data,name1,name2):
    plt.figure(figsize=(14,8))
    #plot하기 위한 좌표를 저장하는 list정의
    li=[] #name1의 평점을 저장
    li2=[] #name2의 평점을 저장

    for i in critics[name1]:
        if i in data[name2]: #같은 영화에 대한 평점이 있다면
            li.append(critics[name1][i]) #name1의 i에 대한 영화 평점
            li2.append(critics[name2][i]) #name2의 i에 대한 영화 평점
            plt.text(critics[name1][i],critics[name2][i],i)
    plt.plot(li,li2,'ro')
    plt.axis([0,6,0,6])
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.show()

drawGraph(critics,'BTS','유성룡')
drawGraph(critics,'이황','조용필')


#피어슨 상관계수
#x와 y의 변화하는 정도를 -1~1 사이로 기술한 통계치

def sim_pearson(data , name1,name2):
#피어슨 상관계수 :x와 y가 함께 변화하는 정도(공분산) / (x가 변화하는정도 * y가 변화하는 정도)

    sumX=0 #x의 합
    sumY=0 #y의 합
    sumPowX=0 #x 제곱의합
    sumPowY=0 #y 제곱의합
    sumXY=0 #x*y의 합
    count=0 #영화의 개수

    for i in data[name1]:
        if i in data[name2]:# BTS와 유서룡이 모두 본 영화
            sumX+=data[name1][i]
            sumY+=data[name2][i]
            sumPowX+=pow(data[name1][i],2)
            sumPowY+=pow(data[name2][i],2)
            sumXY+=data[name1][i]*data[name2][i]
            count+=1

    return (sumXY-((sumX*sumY)/count)) / sqrt((sumPowX-(pow(sumX,2)/count))*(sumPowY-(pow(sumY,2)/count)))


print("BTS와 유성룡 피어슨 상관계수: ",sim_pearson(critics,'BTS','유성룡'))
print("이황과 조용필 피어슨 상관계수:",sim_pearson(critics,'이황','조용필'))

#딕셔너리를 수행하면서 기준(BTS)과 다른 데이터(사람)외의 상관계수를 구해보자.=> 내림차순 정렬
def top_match(data,name,index=2,sim_function=sim_pearson):
#data:영화평점딕셔너리 , name:기준이 되는 사람의 이름, index:피어슨 상관계수에서 상위(가장 가까운)몇명을 추출

#피어슨 함수 호출 지정
    li=[]
    for i in data: #전체 영화를 돌겠다
        if name != i: #자신이(BTS) 아니라면
            li.append((sim_function(data, name,i),i))
        li.sort()
        li.reverse()
        return li[:index]


#BTS와 성향이 비슷한 3명 추출
print(top_match(critics,'BTS',3))

#영화를 추천하는 시스템 구성, 예상되는 평점 출력

"""**추천시스넴 구성 순서**
1) 자신을 제외한 나머지 사람들과의 평점에 대한 유사도를 서함
BTS와 강감찬의 추축되는 평점= 유사도*(다른사람의)영화평점
0.7*(강감찬)4
2) 추측되는 평점들의 총합을 구함
3) 추측되는 평점들의 총합/유사도의 총합=> 모든 사람들을 근거로 했을때 예상되는 평점이 추출됨.
4) 아직 안본 영화를 대상으로 예상되는 평점을 구하여, 예상되는 평점이 가장 높은 영화를 추천하자.
"""

def getRecommendataion(data,person,sim_function=sim_pearson):
    li=[] #최종 결과 리턴
    score=0
    score_dic={}
    sim_dic={}
    result=top_match(data,person,len(data))
    print("중간:", result)
    for sim,name in result:  #유사도, 이름
        if sim<0:continue # 유사도 0보다 작으면 빼자
        for movie in data[name]:
            if movie not in data[person]: #이이가 안본영화
                score+=sim*data[name][movie]
                score_dic.setdefault(movie,0)
                score_dic[movie]+=score

                sim_dic.setdefault(movie,0)
                sim_dic[movie]+=sim
            score=0



        #     print(name,"movie:", movie)
        # print("====================")
    return li
print(getRecommendataion(critics,'이이'))
#기준이 되는 사람이 '이이'가 안본 영화를 추출
#안본 영화 각각에 대한 예상 평점을 준다
#예상 평점이 가장 큰 영화를 추천하자