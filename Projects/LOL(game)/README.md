# LOL 승패 예측 모델링 및 주요 변수 탐색

![lol](https://user-images.githubusercontent.com/54128055/129048246-a184de5e-0eab-40f2-8e56-5743970e380e.png)

**팀 β** 

참여 : 윤나요 최벽문

기간 : 2021. 7. 21 ~ 8. 2 

## 목차

1. 주제 설정 및 목표
2. 데이터
    1. 데이터 설명
    2. 전처리
    3. 최종 데이터 셋
3. EDA
    1. 수치형 변수 시각화
    2. 범주형 변수 시각화
    3. 상관 분석
4. 승부예측 모델링
    1. LightGBM Model
    2. Linear Regression Model
    3. 변수 축소 후 2차 모델링
5. 부록
    1. Champion 선호도 테이블
    2. Silver Rank 승부 예측 모델링

---

### League of Legends 란?

리그 오브 레전드는 5명의 강력한 챔피언으로 구성된 양 팀이 서로의 기지를 파괴하기 위해 치열한 사투를 벌이는 전략 게임입니다.

## 1. 주제 설정 및 목표

### 주제 : LOL 경기의 데이터를 바탕으로 해당 경기의 승부를 예측할 수 있다.

- 승패 예측이 가능할 때, 주효한 변수를 구분해낼 수 있다.
- 주요 변수의 분산이 승패에 영향력이 큰 변수인 지 확인할 수 있다. (팀 구성원 중 유독 튀거나 못하는 플레이어가 있다면 승리에 지장을 줄 것인가?)
- (추가) 경기 시간에 따라 승패 예측 모델을 다르게 구성할 수 있는 지 확인할 수 있다.

## 2. 데이터

### a. 데이터 설명

**League of Legends(LOL) - Ranked Games 2020**

challenger,grandmaster,master 108,000 game data (Riot, korea, 2020)

출처 : [https://www.kaggle.com/gyejr95/league-of-legendslol-ranked-games-2020-ver1](https://www.kaggle.com/gyejr95/league-of-legendslol-ranked-games-2020-ver1)

분석에 사용한 데이터는 2020년 Lol 한국 서버의 Chanllenger ~ GrandMaster 등급의 랭크전 108,000 게임의 데이터를 사용하였다.

해당 랭크 구간은 플레이어 상위 0.09% 이상의 최상위 리그이다.

### Match Data

- ***Column***
    - gameCreation : 게임 생성번호
    - gameDuration : 게임 진행시간 (단위 : 초)
    - gameId : 게임(매치) 고유값
    - gameMode : 게임 모드
    - gameType : 게임 종류
    - gameVersion : 클라이언트의 버전
    - mapId : 맵 구분
    - participantIdentities : 플레이어 정보 (dict)
    - participants : 플레이어 당 플레이 데이터 (dict)
    - platformId : 서버의 국가
    - queueId : team간의 게임이 성사된 id
    - seasonId : 시즌 번호
    - status.message : --
    - status.status_code : --
    - gameCreation : 듀오나 솔로 등의 상태로 참여 시 생성되는 번호로 유추되나 이번 분석에서는 사용하지 않음

- ***특이사항***
    - gameDuration

        ![gameDuration](https://user-images.githubusercontent.com/54128055/129048446-44c8da58-9aaa-4c28-87a3-43f937346987.png)

        gameDuration (경기가 진행된 시간) 의 분포를 보았을 때 전체적으로는 정규분포의 형태를 띄는 것처럼 보이나, 2 군데의 특이점이 발생한다.

        1. ***400초 이하***

            400초 이하의 그 수가 전체 경기의 1.1% 가량으로 180초 부근에 집중되어 있다.

            이는 네트워크 이상이나 경기 시작 시 포기자의 등장으로 경기가 성립되지 않아 180초부터 선택할 수 있는 '경기 다시하기' 시스템의 결과로 보인다.

            따라서 이 범위의 데이터는 노게임으로 간주하여 분석에서 제외한다.

            ![under420](https://user-images.githubusercontent.com/54128055/129048615-9848ff1b-7ee8-4cc3-8849-7e50fffc179c.png)

        2. ***400초 초과 1020초 이하***

            해당 범위의 경기는 전체 경기의 19.5% 에 해당하며 900초 이후에 그 수가 집중되어 있다.

            이는 경기 포기(패배 선언) 시스템의 흔적으로, 경기 시작 후 900초 부터 팀원 전원의 동의로 패배 선언이 가능하다.

            패배 선언은 팀 전원의 동의가 필요하여, 2~3회까지도 투표 시도가 이루어지고, 1회 당 60초가 소요된다.

            900초에 1020초까지 경기수가 집중되어 있는 것은 패배 선언 투표가 원활하게 이루어지지 않아 여러번 시도한 경우가 포함된 것으로 보인다.

            따라서 이 범주의 데이터를 ***'단기전'***으로 분류한다.

            ![under1020](https://user-images.githubusercontent.com/54128055/129048683-da0190b7-bddf-4c40-8950-075263f09797.png)

        3. ***1020초 초과***

            해당 범위의 데이터는 전체 경기의 80% 가량에 해당하며, ***'중장기전'***으로 분류한다.

    - participants

        participants 컬럼은 경기에 참여한 플레이어들의 행동 데이터를 압축하여 저장하고 있다.

        participants 는 아래와 같이 한 경기 당 10명의 플레이어 데이터를 압축해두었고, 주요 데이터는 다시 stats 로 압축되어 있다. 

        ![participants](https://user-images.githubusercontent.com/54128055/129049211-4fab6c3a-1785-4724-afe8-fb2afc1f99d4.png)
        stats 데이터는 10명의 플레이어의 102개 컬럼의 데이터를 가지고 있어 필요한 데이터를 골라내는 작업이 필요했다.

        해당 데이터 중 종속적인 관계를 갖는 컬럼은 최종 컬럼만을 선택하였고, 가능한 수치형으로 계산할 수 있는 컬럼들을 주로 선택하였다.

        ![stats](https://user-images.githubusercontent.com/54128055/129049260-36adc51c-ca9c-4d8f-a6fa-46e574f5d75c.png)

### Winner / Loser Data

- ***Column***
    - teamId : 팀 구분
    - win : 승리 여부
    - firstBlood : 먼저 적을 사망하게 했는지 여부
    - first___ : 먼저 특정 종류의 오브젝트를 사냥했는지의 여부
    - __ Kills : 특정 오브젝트를 사냥한 숫자
    - ban : 선택 금지 영웅 목록
    - gameId : 게임(매치) 고유값

- ***특이사항***

     winner / loser 데이터는 각 gameId 를 기준으로 승자와 패자로 구분되어 100팀과 200팀의 데이터가 각각 섞여 있는 형태로, 100팀의 승리를 예측한다 는 Binary 예측을 하기에 부적절한 형태였다. 이를 해결하기 위해 두 테이블을 합쳐 정렬한 후 100 / 200 팀의 데이터로 각각 구분하여 다시 정렬할 필요가 있었다.

    ![winner](https://user-images.githubusercontent.com/54128055/129050237-b7d927bc-a8d8-43ef-add7-636fafa8e205.png)

    Winner Data

    ![loser](https://user-images.githubusercontent.com/54128055/129049345-3fc734de-d216-4247-9efb-2e37f4254457.png)

    Loser Data

---

### b. 데이터 전처리

### Match Data

- ***컬럼 선택***

    데이터의 대부분의 차지하는 gameMode : Classic (85%) / mapId : 11(88%) 의 데이터 중

    아래 컬럼만 추출하여 사용한다.

    - gameDuration : 게임 진행 시간
    - gameId : 고유번호
    - participants : 플레이어 행동 데이터

    ![match_columnselection](https://user-images.githubusercontent.com/54128055/129049400-d5566793-030a-4f88-8e56-fc823cad6651.png)
- 압축된 데이터 정리하기
    1. 각 행의 participants 컬럼 데이터를 데이터 프레임 형태로 변경

        ![participants](https://user-images.githubusercontent.com/54128055/129049488-de3940f0-6e5d-45bb-872a-d2bb011e5fda.png)
    2. participants 의 stats 데이터를 컬럼으로 풀어서 participants 프레임에 합친다. (이 과정에서 사용할 컬럼만 선택하여 새로운 프레임을 생성)

        ![stats 1](https://user-images.githubusercontent.com/54128055/129052856-aedac41f-c5ee-4c6a-b774-0dd9a6660200.png)

    3. 플레이어별 데이터를 팀별 데이터로 변환한다.

        ![agg](https://user-images.githubusercontent.com/54128055/129049751-bb84be69-1233-4a42-927a-568f7441b613.png)

    4. 팀별 데이터를 하나의 로우로 치환한다.

        ![team_row](https://user-images.githubusercontent.com/54128055/129049816-5bbcfc5f-4316-4064-9051-3e4c8c0bde33.png)

    5. 일련의 과정을 함수로 만들어, participants 데이터를 모은 하나의 데이터 프레임을 작성한다.

        ```python
        # 한 개의 participants 데이터를 한 줄의 데이터 프레임으로 변환하는 함수
        def get_stats(x) :
        		#pt_cols : participants 데이터 내에서 사용할 컬럼 지정
            pt_df = pd.DataFrame(x)[pt_cols]
            
        		#participants 데이터 내의 stats 데이터를 하나의 데이터 프레임으로 만들고 사용할 컬럼(stat_cols)을 불러온다.
            temp = pt_df['stats'].apply(pd.Series)[stat_cols]
            temp.columns = stat_cols_convert
            
        		#stats 데이터를 participants 데이터 프레임에 붙혀넣는다.
            pt_df = pt_df.merge(temp, how='left', on='participantId')
            
        		#participants 데이터를 팀별로 연산한다.
            pt_agg_df = pt_df.groupby('teamId').agg(teamId_agg)    
           
            agg_cols = []

        		#multi column 으로 되어있는 컬럼을 단일 컬럼으로 변환하기 위해 컬럼명을 취합
            for col in pt_agg_df.columns :
                if col[0] in ['goldE', 'champLv']:
                    agg_cols.append(col[0] + '_' + col[1])
                else :
                    agg_cols.append(col[0]) 
                    
            pt_agg_df.columns = agg_cols
            
        		#팀별로 연산된 데이터를 각 팀별로 분할하여 한 개 row 로 변환한다.
            team_100 = pt_agg_df.loc[[100]].reset_index(drop= True)
            team_100.columns = '100_'+team_100.columns

            team_200 = pt_agg_df.loc[[200]].reset_index(drop= True)
            team_200.columns = '200_'+team_200.columns
            
            return pd.concat([team_100, team_200], axis = 1)

        #위의 함수를 전체 데이터 프레임에 적용하여, 원본 데이터와 같은 행을 가지는 participants 데이터 프레임을 구한다.  
        def get_stats_df(df) :
            
            stats_df = pd.DataFrame()
            
            for i in range(df.shape[0]) :
                stats_df = stats_df.append(get_stats(df['participants'][i]))
                
            stats_df = stats_df.reset_index(drop=True)
                
            return stats_df

        stats_df = get_stats_df(match_df)
        ```

    6. match 데이터 프레임에 함수를 적용하여 플레이어 행동 데이터를 구성한다.

        ![stats_df](https://user-images.githubusercontent.com/54128055/129049861-c25e4e04-3497-458e-bbe9-a99b37fa384e.png)

    7. 위 데이터 프레임을 match 데이터 프레임에 합쳐 match df 전처리를 완료

        ![match_stast](https://user-images.githubusercontent.com/54128055/129049904-53ce8545-1608-4c6d-820b-a885fd82b0e8.png)

### Winner / Loser Data

- 데이터 통합
    1. winner / loser 데이터를 합쳐 100 / 200 구분으로 데이터를 재정렬한다.

        ![win_lose](https://user-images.githubusercontent.com/54128055/129049934-0a5381ad-7840-4679-812e-f50bd1d3efc7.png)

    2. gameId 를 기준으로 100 / 200 팀 별 데이터를 한 개의 row 로 정렬한다.

        ![win_lose2](https://user-images.githubusercontent.com/54128055/129049973-1f23929a-c178-431f-a8a8-74f1a2cf7e27.png)

    3. gameId 를 기준으로 match data 와 winner/loser data 를 합친다.

        ![win_lose3](https://user-images.githubusercontent.com/54128055/129050005-f16e7193-91c9-44a8-9da7-773970881a1d.png)

### Main Data

- gameDuration 을 기준으로 durationType 변수를 생성하여 데이터를 3개의 그룹으로 구분한다.

    durationType 0 : gameDuration ≤ 400 / nogame

    durationType 1 : gameDuration ≤ 1080 / 단기전

    durationType 2 : gameDuration ≥ 1080 / 중장기전

- 모델 연산에서 사용할 컬럼을 줄이기 위해 수치형 변수는 100 과 200의 차이로 변환한다.

    > Ex) killDiff = 100_kill - 200_kill

- T/F 의 불린 데이터는 100 팀과 200 팀이 서로 대척점에 있는 데이터로 완전한 상관성을 가지기 때문에, 100 팀의 데이터만 사용한다.

    > xxx_win, xxx_first___

---

## 3. EDA

EDA 과정을 통해 전반적으로 데이터셋의 모든 변수들은 승리팀과 패배팀의 분포가 명확이 나뉘는 것을 확인 할 수 있었으므로, 해당 데이터셋을 통해 승리 예측 모델을 구축할 경우 정확도가 높을 것으로 예상할 수 있었다.

### a. 수치형 변수

- 총 17개의 Feature
    - 'kill' : 상대팀의 챔피언을 죽인 수
    - 'vision' : 시야 점수
    - 'goldE_sum' : 획득한 골드의 합
    - 'goldE_std' : 획득한 골드의 표준편차
    - 'goldS' : 사용한 골드의 합
    - 'champLv_mean' : 속한 팀 유저들의 평균 레벨
    - 'champLv_std' : 속한 팀 유저 레벨의 표준편차
    - 'towerKills' : 부순 타워의 수
    - 'inhibitorKills' : 부순 억제기의 수
    - 'baronKills' : 처치한 바론의 수
    - 'dragonKills' : 처치한 드래곤의 수
    - 'riftHeraldKills' : 처치한 전령의 수 (게임 내에서 전령은 20분전까지 한번 소환되므로 사실상 0과1)
    - 'killDiff' : 두 팀간의 킬수의 차이
    - 'visionDiff' : 두 팀간의 시야 점수의 차이
    - 'goldDiff' : 두 팀간의 벌어들인 골드 차이
    - 'spandDiff' : 두 팀간의 사용한 골드 차이
    - 'lvDiff' : 두 팀간의 레벨 차이

수치형 변수들에 대해 산점도 그래프를 통해 시각화 해보니, 대부분의 변수들이 승리팀과 패배팀간에 명확한 차이가 있다는 것을 확인 할 수 있다. 

따라서 kill 수, gold 획득량, 챔피언들의 레벨 등 각 변수들에 대해 양팀간의 차이를 구해 분포를 살펴보았다.

### "Kill" feature

- kill feature 는 상대방을 처치한 숫자로, 단기전으로 분류된 경기에서는 승리 팀과 패배 팀의 수치가 명확하게 차이가 난다. 반면, 경기 시간이 길어질 수록 그 차이가 줄어드는 경향을 보여 변수의 영향력이 다소 줄어듦을 보인다.

    ![download](https://user-images.githubusercontent.com/54128055/129050420-ff452185-e22b-4b42-bdfb-c1cdaa05db0b.png)

- 위의 양팀의 "kill의 수" 산점도 그래프를 통해 승리팀과 패배팀의 차이가 분명하게 갈린다고 할 수 있다. 따라서 승리팀과 패배팀의 "Kill 수" 차이에 대해 분포를 확인했으며 이를 durationType 별로 차이가 있는지 살펴보았다.

    $\therefore$  게임 플레이 시간이 장기화될수록 양 팀의 kill 수의 차이가 약간 줄어들지만 명확히 분류가 되는 것을 확인할 수 있다

    ![download_(1)](https://user-images.githubusercontent.com/54128055/129050455-b98b0678-1b34-4476-8fcc-a0baf4096e15.png)

### "vision" feature

- vision feature 는 경기 중 지도에서 확보한 시야의 수치이다. 해당 수치가 높을 수록 맵의 많은 곳을 경기 중에 확인할 수 있다. vision 수치도 승패에 따라 확연한 분류를 보이지만, 경기 시간이 길어질수록 양 팀의 점수 차이가 줄어드는 경향을 보인다.

    ![download_(3)](https://user-images.githubusercontent.com/54128055/129050530-f55ce9a6-d4a1-4e43-a292-27fe08094173.png)

    ![download_(2)](https://user-images.githubusercontent.com/54128055/129050497-9775a0a3-639c-457b-bb08-c7840b718a05.png)

### "level" feature

- level feature 는 플레이어가 조용하는 champ 의 레벨을 의미하며, 전체적으로 승패에 따른 명확한 상관관계를 보이고, 역시 경기 시간이 길어질 수록 그 차이가 줄어든다.

    ![lvl_scatter](https://user-images.githubusercontent.com/54128055/129050602-94a97b4b-2c74-4785-92d6-a1167204064d.png)

    ![lvl_dist](https://user-images.githubusercontent.com/54128055/129050669-79c10782-eaca-4cec-bc74-8d40355068b0.png)

- 분석하기 앞서 heuristic 면에서 승리한 팀원들의 레벨이 패배한 팀원들의 레벨보다 높은 양상을 보일 것이라 생각했으며, 위처럼 실제로 데이터 또한 승리팀의 평균 레벨이 높은 것을 알 수 있다.  하지만 팀 내의 레벨의 편차가 크다면 승패의 불리한 변수가 될 수 있지 않을까 라는 물음으로 팀 별 champ 레벨의 표준 편차를 비교해 보았다. 대체 적으로 편차가 적은 팀이 승리하는 경향을 보이지만, 전혀 반대 되는 케이스도 관측이 되기 때문에 해당 변수는 승부 예측에 명확한 지표로 삼기는 어렵다고 판단했다.

    ![lvl_std](https://user-images.githubusercontent.com/54128055/129050704-45c3f86e-c84d-420b-9ad9-1695e941fa34.png)

### "gold" Features

- gold 란, 게임 내에서 장비나 아이템을 구매하기 위한 재화로, 전반적으로 champ 의 평균 레벨과 비슷한 양상을 보인다.

    ![gold_scatter](https://user-images.githubusercontent.com/54128055/129050790-02b62035-c4ee-45e3-9df8-37ef1f445514.png)

    ![gold_dist](https://user-images.githubusercontent.com/54128055/129050745-5bf84452-0688-4ed5-9d43-ec19bc00c499.png)

- gold 사용량

    "gold 사용량"은 단기전에서는 보다 많이 사용한 팀이 이기는 경향을 보이지만, 경기 시간이 길어질 수록 그 차이가 확연히 줄어든다. 이는 gold 의 사용처가 제한적이고, 경기 후반으로 갈수록 필요가 줄어들고, 경기에 미치는 영향력이 줄어드는 재화라는 점을 알 수 있다.

    ![gold_use](https://user-images.githubusercontent.com/54128055/129050867-0ce81479-1aa4-44b4-9753-836314c2245c.png)

## b. 범주형 변수

- 총 6개의 변수
    - firstBlood - 두 팀 중 먼저 상대팀의 챔피언을 킬했는지 여부
    - firstTower - 두 팀 중 먼저 상대팀의 타워를 부쉈는지의 여부
    - firstinhibitor - 두 팀 중 먼저 상대팀의 억제기를 부쉈는지의 여부
    - firstBaron - 두 팀 중 누가 먼저 바론을 죽였는지의 여부
    - firstDragon - 두 팀 중 누가 먼저 드래곤을 죽였는지의 여부
    - firstRiftHerald - 두 팀 중 누가 먼저 전령을 죽였는지의 여부

범주형 변수 또한 각 변수에 대하여 승리팀과 패배팀간의 빈도수를 확인해 보았다

## 범주형 변수: bool type

### "firstBlood" Feature

- 승리한 팀일수로 "firstblood"를 쟁취한 비율이 20% 정도 높다는 것을 알 수 있다

    ![firtblood](https://user-images.githubusercontent.com/54128055/129050932-df9898c2-a1be-4e0f-bc42-9577c3cebd42.png)

- 따라서 해당 변수의 비율이 게임 시간에 따라 변화 되는지 확인해 보았다. 아래와 같이 게임 지속 시간에 따라 비교를 해보면, 17분 이전에 종료된 게임이 "firstblood"의 쟁취 여부가 중장기적인 게임에 비해 상대적으로 중요하다 판단된다.

    ![fstblood_duration](https://user-images.githubusercontent.com/54128055/129050979-f2570257-3cf9-4135-b66c-3b3edc2faaa7.png)

### "firstTower" Feature

- 두 팀 중 먼저 상대팀의 타워 부순팀이, "firstBlood" 변수와 같이, 승리한 비율이 높으며, 게임이 중장기전으로 들어갈 경우 초반에 끝난 게임에 비해 상대적으로 비율이 낮음을 알 수 있다.

    ![fst_twr](https://user-images.githubusercontent.com/54128055/129051036-28412ed7-9ce9-4445-aec0-1896ad7ca35c.png)

### "firstInhibitor" Feature

- 반면 억제기의 경우 17분 이내에 종료된 게임이든 17분 이상 지속된 게임이든 해당 변수의 쟁취 여부의 중요성은 감소하지 않는다고 할 수 있다.

    ![fst_inhi](https://user-images.githubusercontent.com/54128055/129051069-6593ed40-bafb-4fb0-9929-fbb4c8f1dc38.png)

### "firstBaron" Feature

- 첫 바론 쟁취 여부 또한 성공한 팀의 승리 확률이 높다고 할 수 있지만, 바론은 게임이 시작되고 15분 후에 잡을 수 있기 때문에 17분 이내에 종료된 게임에는 영향이 없음을 확인 할 수 있다.

    ![fst_baron](https://user-images.githubusercontent.com/54128055/129051098-1f894d2b-050f-4224-9a2b-a6ad8adf308e.png)

### "firstDragon" & "firstRiftHerald" Features

- 양팀 중 첫 드래곤과 첫 전령의 사냥 여부의 변수들은 게임이 지속될수록 해당 변수들의 중요도가 점차 낮아짐을 확인 할 수 있었다.
- 따라서 기존의 17분 이상 지속된 게임을 추가적으로 17분 부터 30분 이내에 종료된 게임과 30분 이상 지속된 게임으로 분류하여 "firstDragon" 과 "firstRiftHerald" 변수들의 비율을 확인해보았다.
- 이를 통해 30분 이상의 장기전으로 가는 게임은 해당 범주형 변수의 쟁취 여부가 승패에 영향을 거의 끼치지 않을 것으로 판단 된다.

    "firstDragon" & "firstRiftHerald"





    ![fst_drgon](https://user-images.githubusercontent.com/54128055/129055018-74028171-325a-4fb4-bf44-eb55d3c7f941.png)

    ![fst_herald](https://user-images.githubusercontent.com/54128055/129055112-ee5ccebf-f323-4fa0-bf45-174e113ca39e.png)

### 상관 분석

- Duration Type 1 상관 분석

    단기전에서는 'kill', 'gold 획득량', 'lv', 'gold 사용량' 차이가 승리에 큰 영향력을 끼치는 것을 확인할 수 있고, 해당 변수들은 각 변수끼리도 강한 양의 상관관계를 보여 한 하나의 수치가 높으면 다른 수치들도 같이 높을 것이라고 예상할 수 있다.

    경기 12분까지 분류한 duration Type 1 데이터에서는 Baron 오브젝트가 출현하지 않기 때문에 관련 수치는 공란으로 표시되고 있다.

    ![duration1_corr](https://user-images.githubusercontent.com/54128055/129051672-3d8b0911-991f-4c6f-99a9-913a8c7f1916.png)

- Duration Type 2 상관 분석

    중장기전 상관 계수는 단기전의 상관 계수와 유사한 분포를 보인다.

    하지만 전체적으로 상관 관계가 다소 약하게 나타나는 경향이 있다.

    ![durationt2_corr](https://user-images.githubusercontent.com/54128055/129051684-f4e14524-c4f8-4c95-a4fb-af32e539cf85.png)

- 상관도 비교

    모델링의 타겟 변수로 사용할 '100_win' 을 기준으로 단기전과 중장기전의 상관계수를 비교하였을 때, 중장기전으로 갈 수록 하나의 변수가 가지는 영향력이 다소 줄어듬을 확인할 수 있다.

    ![corr_비교](https://user-images.githubusercontent.com/54128055/129052377-2177d184-9193-4604-8e83-924a3cf496cc.png))

## 4. 승부 예측 모델링

### a. LightGBM Model

- 단기전 (DurationType 1)
    - 예측 정확도

        단기전 승부 예측 모델링의 정확도는 99.9% 로 거의 모든 경기의 결과를 맞출 수 있었다.

        ![confusionMatrix1](https://user-images.githubusercontent.com/54128055/129051773-11da74e1-d383-4cff-b633-8a372ff143aa.png)


    - Feature Importance

        단기전 승부 예측에는 'lv', 'gold 획득량', 'kill' 차이가 가장 영향력 있는 변수로 나타났고, 상위 3개의 변수가 모델에서 대부분의 설명력을 가지는 것으로 보인다.

        ![FI_lgbm_1_rate](https://user-images.githubusercontent.com/54128055/129051831-525d2305-726a-455f-9043-ed7220f2cb7e.png)

- 중장기전 (DurationType 2)
    - 예측 정확도

        중장기전의 예측 정확도도 99% 로 단기전에 비해서 다소 오분류하는 경우가 늘어났지만, 전체적으로 봤을 때 대부분의 결과를 높은 확률로 맞추고 있다.

        ![Untitled](https://user-images.githubusercontent.com/54128055/129051916-f7faa1e8-3022-4b9a-b783-aa9f97e4652a.png)

    - Feature Importance

        중장기전 예측 모델에서는 단기전과 달리 상위 몇개 변수에 몰려있던 변수 중요도가 다소 분산되는 모습을 보인다.

        그럼에도 단기전의 중요 변수는 역시 상당한 영향력을 가지는 것을 확인할 수 있다.

        ![FI_lgbm_2_rate](https://user-images.githubusercontent.com/54128055/129051934-145e1cbb-a882-4e79-ac6c-d5ba354a5667.png)

- Feature Importance 비교

     두 개 모델의 변수 중요도를 비교했을 때, 단기전의 경우 모델의 설명력이 특정 변수에 집중되어 있고, 경기 시간이 길어질 수록 특정 변수가 지닌 설명력이 다소 줄어들면서 변수의 중요도가 분산되는 경향을 확인할 수 있었다.

    ![FI_lgbm_comparision](https://user-images.githubusercontent.com/54128055/129051980-bd5ec87a-151d-4804-9368-45bb4240a9e0.png)

### b. Linear Regression Model

- 단기전 (DurationType 1)
    - 예측 정확도

        선형 회귀 모델로 예측하였을 때도 단기전 승부 예측 모델링의 정확도는 99.9% 로 거의 모든 경기의 결과를 맞출 수 있었다.

        ![confusionMatrix1](https://user-images.githubusercontent.com/54128055/129052037-33632602-3872-4160-a7c5-a8cea66d3071.png)

    - Feature Importance

        단기전 승부 예측에는 'lv', 'gold 획득량', 'kill' 차이가 가장 영향력 있는 변수로 나타났고, 상위 3개의 변수가 모델에서 대부분의 설명력을 가지는 것으로 보인다.

        ![Untitled 1](https://user-images.githubusercontent.com/54128055/129053966-90b0b2b8-b8f7-45cb-a919-e76e16ae09bc.png)
- 중장기전 (DurationType 2)
    - 예측 정확도

        중장기전의 예측 정확도도 99% 로 단기전에 비해서 다소 오분류하는 경우가 늘어났지만, 전체적으로 봤을 때 대부분의 결과를 높은 확률로 맞추고 있다.

        ![Untitled 2](https://user-images.githubusercontent.com/54128055/129054070-0ed2e704-635b-4760-b074-8f31a157f63e.png)

    - Feature Importance

        선형 회귀 모델의 경우도 Light GBM 모델과 거의 같은 결과를 보이고 있다.

        중장기전 예측 모델에서는 단기전과 달리 상위 몇개 변수에 몰려있던 변수 중요도가 다소 분산되며, 단기전의 중요 변수가 상당한 영향력을 가진다.

        ![Untitled 3](https://user-images.githubusercontent.com/54128055/129054120-830520ee-6876-431a-8d32-a1dd4ac180e8.png)

- Feature Importance 비교

     두 개 모델의 변수 중요도를 비교했을 때, 단기전의 경우 모델의 설명력이 특정 변수에 집중되어 있고, 경기 시간이 길어질 수록 특정 변수가 지닌 설명력이 다소 줄어들면서 변수의 중요도가 분산되는 경향을 확인할 수 있었다.

    ![Untitled 4](https://user-images.githubusercontent.com/54128055/129054171-6e5e1fea-d525-4c0b-9a93-60c04efdb18a.png)

### c. LightGBM Model (변수 축소)

- 변수 축소

    단기전 / 중장기전, 양 모델에서 중요도가 0.05 를 기준으로 8개의 변수를 추려, 해당 변수를 기준으로 모델을 축소하여 구성한다.

    ![Untitled 5](https://user-images.githubusercontent.com/54128055/129054202-22ee7fc6-0079-487b-9ac3-e47b0daf80c2.png)

- 단기전 (DurationType 1)
    - 예측 정확도

        단기전의 승부 예측 결과는 변수를 축소하여도 차이가 없었다.

        ![confusionMatrix1](https://user-images.githubusercontent.com/54128055/129052037-33632602-3872-4160-a7c5-a8cea66d3071.png)

    - Feature Importance

        단기전 승부 예측에서 모델링에 사용하는 변수를 줄인 결과 'lv', 'gold 획득량', 두 변수의 영향력이 더 두드러지게 나타났다.

        ![Untitled 6](https://user-images.githubusercontent.com/54128055/129054249-76ab4802-4b8b-4eab-8b27-804be805a4cd.png)

- 중장기전 (DurationType 2)
    - 예측 정확도

        중장기전의 예측 정확도도 99.3% 로 변수 축소 전에 비해서 다소 오분류가 늘기는 했지만, 큰 차이가 없었다.

        변수 축소로 좀 더 가벼운 모델을 운용하여도 정확도 면에서 손해는 거의 없을 것으로 예상한다.

        ![Untitled 7](https://user-images.githubusercontent.com/54128055/129054291-999d603f-3d09-40a6-a310-3ba269a73824.png)

    - Feature Importance

        변수 축소의 결과 이전 모델에서 어느 정도 영향력을 가지던 변수들도 결과에 대한 설명력을 거의 잃어버리는 모습을 보인다. (killDiff)

        상위 변수가 가지는 설명력의 순서는 유지되는 모습을 확인할 수 있다.

        ![Untitled 8](https://user-images.githubusercontent.com/54128055/129054354-4cc9ee44-ea45-489c-88a9-8904f4992deb.png)

- Feature Importance 비교

    변수 중요도의 분포가 변수 축소 이전과 크게 달라지지 않는 모습을 확인할 수 있다.

    극단적으로 상위 변수 3~4개 정도로 축소하였을 경우에도 모델링의 설명력을 충분히 유지할 수 있을 것 같아, 보다 경제적인 모델링이 가능할 것 같다.

    ![Untitled 9](https://user-images.githubusercontent.com/54128055/129054384-f19a136b-3b8e-490d-99d3-aa317b62532f.png)

# 5. 결론

### a. 주제 및 목표

- [x]  **주제 : LOL 경기의 데이터를 바탕으로 해당 경기의 승부를 예측할 수 있다.**
    1. 경기의 종료 후 경기의 데이터를 바탕으로 한 사후 분석이었다,
    2. 양 팀의 데이터를 바탕으로 한 쪽의 승리를 예측하기 때문에 수치 간의 차이가 비교적 명확하게 나타나는 데이터였다.

    LightGBM과 선형 회귀, 2가지의 모델을 사용하고, 변수 축소 후 재모델링까지 진행하였지만, 상기의 사유로 인하여 무척 명확하게 경기 결과를 예측할 수 있는 분석이었다. 당초 분석을 계획하였을 때는 차츰 예측의 정확도를 높혀가는 방향의 분석을 예상하였지만, 그와 반대로 보다 적은 변수로 경제적인 모델링을 하고, 무척 높은 정확도의 사유를 찾아야 하는 분석이 되었다.

    결과적으로 경기의 데이터가 주어졌을 때, 거의 모든 경기의 승부를 정확하게 예측할 수 있음을 확인하였다.

- [x]  목표 1 : 승패 예측이 가능할 때, 주효한 변수를 구분해낼 수 있다.

    승패 예측에 가장 주효한 변수는 champ 의 'Lv', 'gold' 의 획득량이었다. 17분 이전에 결정나는 경기에서는 이 두 변수가 예측 모델에서 대부분의 설명력을 가지며, 경기 시간이 길어질 경우, 'gold' 의 사용량, 'Tower' 오브젝트의 처치 수 등이 차순위의 주요 변수로 떠오르기도 했다.

- [x]  목표 2 : 주요 변수의 분산이 승패에 영향력이 큰 변수인 지 확인할 수 있다.

    팀내의 균형을 확인할 수 있는 champ의 'Lv' 분산, 'gold' 획득량의 분산은 적은 편이 승리하는 경향을 보이기는 하였다. 이는 팀내 플레이어 간의 수준이 비슷하거나, 균형적인 활약을 하고 있을 때, 혹은 눈에 띄게 못 하는 플레이어가 없을 때 승리할 가능성이 높다는 해석도 가능하지만, 반대의 수치도 관측이 되었기 때문에 이를 단순하게 결론을 내긴 어려웠다.

    반대로 팀 내에 유독 잘하는 플레이어가 있는 경우, 한 사람이 전략적으로 활약을 하는 경우도 발생하는 것으로 파악이 되며, 이는 단순히 해당 변수의 분산만으로는 파악이 어렵고, 다른 변수와 연결하여 자세히 살펴볼 필요를 느꼈다.

- [x]  목표 3 : 경기 시간에 따라 승패 예측 모델을 다르게 구성할 수 있는지 확인 할 수 있다.

    경기 시작에 따라 '단기전'과 '중장기전'으로 나눠서 모델링을 하는 것이 유의미함을 확인하였다.

    하지만 두 모델에서 주요한 변수는 차이가 적었으며, 주요한 변수의 차이가 경기 내에서 유의미하게 차이가 날 경우, 경기가 빨리 끝나는 경향이 있다로 결론을 낼 수 있다. 반면 경기가 길어질 경우에는 특정한 변수의 영향력이 줄어들고, 경기 내의 보다 많은 변수의 영향력이 고려되어야 함을 확인하였다.

### b. 아쉬웠던 점과 추가 분석의 가능성

- 경기 후의 데이터를 활용한 사후 분석으로, 결과론적인 분석이 된 것 같아서 아쉬웠다. 경기 데이터를 경기 시간 별로 나타낸 자료를 구할 수 있다면, '일정 시간이 경과한 경기의 승부 예측'과 같이 보다 실용적인 실시간 승부 예측 모델을 구축할 수 있을 것 같다.
- 예상 보다 높은 모델 정확도가 아쉬웠다. 최상위권 경기 데이터를 사용한 영향이 있는 지 확인하고자, 하위권 Silver 랭크의 데이터를 스크랩핑하여 모델에 적용하여 보았지만 결과가 그렇게 다르지 않아, 사후 분석의 특징이자 한계로 판단된다. Silver 랭크 데이터를 바탕으로한 결과는 부록으로 남긴다.

## 부록1. 영웅 선택 순위

경기 시간에 따른 분류 / 승패의 분류에 무관하게 대부분의 조합에서 'Lee Sin' Champ 선택율이 1위 임을 확인할 수 있었고,

1~5 순위의 Champ 선택은 약간의 차이가 있을 뿐 대동소이하며, 10 순위까지 범위를 넓혀보아도 거의 차이가 없음을 확인할 수 있다.

이는 해당 데이터가 상위 0.09% 의 최상위권 경기의 데이터임에 따라 가장 효율이 좋은 Champ 가 선호되는 경향이 반영된 것으로 보인다.

![Untitled 10](https://user-images.githubusercontent.com/54128055/129054429-ee22a7d8-ddff-46f4-aaef-8bdd2f26a176.png)

## 부록2. Silver Rank Data

본문의 승부 예측 결과는 99% 이상의 정확도를 보이며, 대부분의 경기 결과를 정확하게 예측하고 있다.

이는 경기의 사후 데이터를 바탕으로 승부를 예측하기 때문인 것으로 예상되지만, 혹시 대상이 되는 데이터가 최상위권의 경기 데이터이기 때문에 극도로 정제된 형태의 데이터가 발생하는 것이 아닌지 의문을 가지게 되었다.

따라서 기존에 분석에 사용한 데이터와 달리 게임 상에서 하위권에 속하는 Silver Rank 데이터를 사용하여 상기 모델에 맞춰 예측을 해보기로 하였다.

- 예측 정확도

    단기전 (DurationType 1)

    수집한 경기의 표본 수가 무척 적긴 하지만 기존 모델로 모든 경기의 결과를 정확하게 예측하였다.

    ![Untitled 11](https://user-images.githubusercontent.com/54128055/129054499-3acb54ab-27b2-4b48-a4c2-ea09d5ac0fce.png)

    중장기전(DurationType 2)

    중장기전의 경우도 99% 이상의 정확도를 보이고 있다.

    ![Untitled 12](https://user-images.githubusercontent.com/54128055/129054508-5bb70ec2-6a1e-4f95-a509-bfa9427df9b5.png)


Silver Rank Data 를 모델에 적용하였을 경우 기존의 결과와 다른 결과가 도출되는 것을 기대하였으나, 모델의 예측 정확도가 지나치다 싶을 정도로 높은 이유는 Rank 에 따른 정제된 플레이로 인한 것이 아니라 사후 데이터로 결과를 파악하기 때문인 것으로 확인 되었다.
