import math
import random
import re
from difflib import SequenceMatcher
from time import time

import hanja
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from torch import nn, Tensor
from torch.backends import cudnn

from logger_main import logger

# seed 설정
seed = 142
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)

# 전처리
class Preprocessing:
    def __init__(self):
        self.punctuations = '[^\w\s\d]' # 모든 기호
        self.expression = [
            "기획제작",
            "도서출판",
            "사단법인",
            "주식회사",
            "작자미상",
            "작자 미상",
            "일러스트", # 일러스트 추가
            "편집부",
            "편집국",
            "글그림",
            "지은이",
            "출판부",
            "출판사",
            "옮긴이",
            "연구회",
            "[공]",
            "글·사진",
            "기획글",
            "제자들",
            "인터뷰",
            "제작팀",
            "그림",
            "같이",
            "감독",
            "감수",
            "구성",
            "공저",
            "공역",
            "글씀",
            "그림",
            "동화",
            "만화",
            "번역",
            "삽화",
            "사진",
            "시집",
            "엮고",
            "원저",
            "꾸밈",
            "해설",
            "원작",
            "외저",
            "엮음",
            "역주",
            "연출",
            "요리",
            "옮김",
            "집필",
            "지도",
            "교열",
            "작자",
            "미상",
            "지음",
            "지휘",
            "저자",
            "제작",
            "작곡",
            "편저",
            "펴냄",
            "추천",
            "평역",
            "각색",
            "共著",
            "外著",
            "각본",
            "판매",
            "공급",
            "기획",
            "쓰다",
            "쓰고",
            "찍다",
            "찍고",
            "듣고",
            "그리다", # 쓰고, 그리다, 찍다 추가
            "구술",
            "정리", 
            "글",
            "저",
            "편",
            "외",
            "역",
            "譯",
            "編",
            "著",
            "판",
            "공",
            "씀",
            "등",
            "by",
            "written",
            "illustrated",
            "illustration",
            "story",
            "picture",
            "illus.",
            "Publishing",
            "Publisher",
            "Press",
            "ill",
            "song",
            "sung",
            "and",
            "with",
            "illus."
        ] # 도서관 측 제공

    # query와 return_cands의 value는 모두 string이다.

    # 전체 열 공통
    def space_punc(self, text:str) -> str:
        step0 = re.sub('_', " ", text) # underscore는 [^\w\s\d]로 제거 안됨
        step1 = re.sub(self.punctuations, " ", step0)    # 공백 및 기호 정제
        step2 = hanja.translate(step1, "substitution") # 한자-> 한글 변환
        step3 = step2.lower()                          # 영문 대문자 -> 영문 소문자
        final = re.sub(r"\s+", " ", step3)
        return final.strip()

    # 저자
    def author_processing(self,author_text:str) -> str:
        other_expression = ['\s\[[가-힣]+\]']   
        regex = '|'.join([f"\s?{auth_ex}" for auth_ex in self.expression] + other_expression)
        
        # re.sub적용
        step1 = re.sub(regex, "", author_text) # 첫 번째 처리
        final = re.sub(regex, "", step1) # 한 번 더 처리
        return final.strip()

    def process_author(self, text:str) -> str:
        step1 = self.space_punc(text) # 항상 기호 -> space 1개로 바꾸기
        step2 = self.author_processing(step1)
        final = self.space_punc(step2)
        return final.strip()

    # 출판사
    def publisher_processing(self, publisher_text:str) -> str:
        # 처리할 단어 및 정규표현식

        city_name = [
            '서울',
            '파주',
            '고양',
            '수원',
            '용인',
            '성남',
            '부천',
            '화성',
            '안산',
            '안양',
            '평택',
            '시흥',
            '김포',
            '광주',
            '광명',
            '군포',
            '하남',
            '오산',
            '이천',
            '안성',
            '의왕',
            '여주',
            '과천',
            '남양주',
            '의정부',
            '양주',
            '구리',
            '포천',
            '동두천',
            '부산',
            '대구',
            '대전',
            '광주',
            '제주'
        ] # 서울시, 경기도의 모든 시, 광역시 등; 군 제외 ex)서울 [출판사명] 2005 중 앞부분 제거
        city_expression = [f"{a_city}\s+" for a_city in city_name]
        other_expression = ['\(주\)', '[ㄱ-ㅎ]', '\s+\d{4}'] # ex)서울 [출판사명] 2005 중 뒷부분 제거
        regex = '|'.join(self.expression + other_expression + city_expression)
        
        # re.sub적용
        step1 = re.sub(regex, "", publisher_text) # 첫 번째 처리
        final = re.sub(regex, "", step1) # 한 번 더 처리
        return final.strip()

    def process_publisher(self, text:str) -> str:
        step1 = self.space_punc(text) # 항상 기호 -> space 1개로 바꾸기
        step2 = self.publisher_processing(step1)
        final = self.space_punc(step2)
        return final.strip()

    # 검색엔진용 전처리
    def title_forsearch(self, text:str) -> list:
        # # 숫자 및 '(apostrophes) 전처리
        # if re.search(r'\d+', text):
        #     for a_iter in re.finditer(r'\d+', text):
        #         start, end = a_iter.span()
        #         if (end-start) == 1:
        #             text = text[:start] + ' ' + text[start:end] + ' ' + text[end:]
        #         else:
        #             continue

        text = re.sub(r"\'", '', text)

        title_versions:list = []

        # version_1 = 원판 검색용, 그대로 사용
        title_versions.append(text)
        
        # version_2 = 형태소 파서 및 띄어쓰기 일치용, 공백 및 기호제거
        version_2 = re.sub(f"\s+|{self.punctuations}", '', text) 
        title_versions.append(version_2)

        # version_3 = 특수문자 분리, 기호만 제거
        step_3_1 = re.sub(self.punctuations, '', text)
        version_3 = step_3_1.replace(r'\d{1}', r'\s.\d\s.')
        title_versions.append(version_3)

        # version_4 = 1단어 일치용, 그대로 사용
        title_versions.append(text)

        # version_5 = N-GRAM용, 그대로 사용
        title_versions.append(text)

        return title_versions

    def author_forsearch(self, text:str) -> list: 
        punc_forauthorsearch = r'[^\w\s\d;]' # ;를 제외한 모든 기호 제거
        regex = '|'.join([f"\s+{auth_ex}" for auth_ex in self.expression] + ['\s?;+\s?'])
        step1:str = re.sub(punc_forauthorsearch, " ", text)
        step2:str = re.sub(r"\s+", " ", step1)
        step3:str = re.sub(regex, ';', step2)
        step4:str = re.sub(regex, ';', step3)
        step5:str = re.sub(';$', '', step4)
        step6:list = step5.strip().split(';')
        step7:list = [re.sub('|'.join(self.expression), '', s) for s in step6 if s]
        final:list = [s.strip() for s in step7 if s] # 공백을 제외한 값만 반환
        return final

    def isbn_forsearch(self, text:str) -> list: 
        isbn_versions:list = []
        if len(text) != 13 and len(text) != 10:
            return isbn_versions
        else:
            pass

        isbn_versions.append(text) # 그대로 append하거나
        isbn_versions.append(text[:-1]+'X') # Parity Codes를 바꿔 append
        isbn_versions.append(text[:-1]+'0')

        return isbn_versions



# 규칙기반 모델
class Differ:
    def __init__ (self):
        self.pp = Preprocessing()

    def match_ratio(self, cand_processed:str, q_processed:str) -> float:
        # 도서정보에 띄어쓰기가 있으면 조건1) 아니면 조건2)
        if ' ' in q_processed:
            # 조건1) 도서정보의 띄어쓰기 기준 토큰을 서지정보에 match -> match된 문자열 이어붙이기 -> 일치율 match ratio 구하기
            common_tokens:list[str] = [] # 공통문자열 작업
            for a_part in q_processed.split(' '):
                match = SequenceMatcher(
                    None, cand_processed, a_part
                ).get_matching_blocks()
                common_tokens += [cand_processed[m.a:m.a+m.size] for m in match if m.size > 1]
            common_string:str = ''.join(common_tokens)

            rest_cand:str = cand_processed # 공통문자열 이외 문자열 작업
            for rm_token in common_tokens+[' ']:
                rest_cand:str = re.sub(rm_token, '', rest_cand)
            
            ordered_cand_processed:str = common_string + rest_cand 
            match_ratio:float = SequenceMatcher(
                None, ordered_cand_processed, q_processed.replace(' ', '')
            ).ratio()
        else:
            # 조건2) 공백없는 도서정보와 공백없는 서지정보의 일치율 match ratio 구하기; 1글자 저자명, 출판사명 대비
            match_ratio:float = SequenceMatcher(
                None, cand_processed.replace(' ', ''), q_processed.replace(' ', '')
            ).ratio()

        return math.log(match_ratio+1, 2) # 일치율의 낙폭이 커짐을 방지

    def in_lang(self, text:str, lang:str) -> bool:
        if lang == 'eng':
            regex = r'[-a-zA-Z]'
        elif lang == 'kor':
            regex = r'[가-힣]'
        else: 
            raise Exception('in_lang의 lang인자 오류: "eng" 또는 "kor"만 입력 가능')
        temp = re.findall(regex, text)
        return True if len(temp) > 0 else False

    def all_lang(self, text:str, lang:str) -> bool:
        if lang == 'eng':
            regex = r'[-a-zA-Z]'
        elif lang == 'kor':
            regex = r'[가-힣]'
        else:
            raise Exception('in_lang의 lang인자 오류: "eng", "kor"만 입력 가능')
        temp = re.fullmatch(regex, self.pp.space_punc(text).replace(' ', ''))
        return True if temp else False

    def differ_author(self, cand_author:str, q_author:str) -> float:

        q_author_processed = self.pp.process_author(q_author)

        if q_author_processed == '': # 전처리 후 공백일 경우 결측치로 반환
            return -1
        else:
            pass

        cand_author_processed = self.pp.process_author(cand_author)

        # 도서정보:서지 = 한글:영어 = 영어:한글의 관계이면 similarity = 0
        if (self.all_lang(cand_author_processed, 'eng') and self.all_lang(q_author_processed, 'kor')) or (
            self.all_lang(q_author_processed, 'eng') and self.all_lang(cand_author_processed, 'kor')
        ):
            return 0

        # 서지 안에 영문+한글이 있으면 영문 앞 뒤로 띄기
        # SequenceMatcher에서 match할 때 영향을 주지 않게, q_author에서 가져온 하나의 토큰(a_part) 안에 영문+한글이 있게 하지 않기 위함
        if self.in_lang(q_author_processed, 'eng') and self.in_lang(q_author_processed, 'kor'):
            eng_wordlist:list[str] = re.findall(r'[a-zA-Z]+', q_author_processed)
            for a_word in eng_wordlist: # 한글안에 영문이 있을 경우 ex) 해리포터Harrypotter마법사의돌
                firstchar_idx:int = q_author_processed.index(a_word)
                lastchar_idx:int = q_author_processed.index(a_word) + len(a_word)
                q_author_processed:str = q_author_processed[:firstchar_idx] + ' ' + q_author_processed[firstchar_idx:] 
                q_author_processed:str = (
                    q_author_processed[: lastchar_idx + 1] + ' ' + q_author_processed[lastchar_idx + 1 :]  # ex) 결과: 해리포터 Harrypotter 마법사의 돌
                )
            q_author_processed:str = re.sub(r"\s+", ' ', q_author_processed)
        
        # 도서정보에 띄어쓰기가 있으면 조건1) 아니면 조건2)
        match_ratio:float = self.match_ratio(cand_author_processed, q_author_processed)

        return match_ratio


    def differ_publisher(self, cand_publisher:str, q_publisher:str) -> float:
        q_publisher_processed = self.pp.process_publisher(q_publisher)

        if q_publisher_processed == '': # 전처리 후 공백일 경우 결측치로 반환
            return -1
        else:
            pass

        cand_publisher_processed = self.pp.process_publisher(cand_publisher)

        match_ratio = self.match_ratio(cand_publisher_processed, q_publisher_processed)

        return match_ratio


    def differ_isbn(self, cand_isbn:str, q_isbn:str) -> int:

        if (len(q_isbn)!=13 and len(q_isbn)!=10): # or (q_isbn == '') #메서드 밖에서 함
            isbn_similarity = -1 # 도서정보에 출판사가 없는 경우(나중에 -1을 제외하고 평균계산, 나눠주는 col 개수에서 차감)

        elif (cand_isbn == '')  or (
            len(cand_isbn)!=13 and len(cand_isbn)!=10
        ): # 서지에 isbn이 없을 때, 13자리나 10자리가 아닐 때 비교불가
            isbn_similarity = 0 # 공백을 제외한 모든 ut가 다른도서가 됨

        else:
            # 서지 및 cand isbn의 parity code여부
            q_paritybool = False
            cand_paritybool = False
            if (q_isbn[-1] == 'X') or (q_isbn[-1] == '0'):
                q_paritybool = True
            elif (cand_isbn[-1] == 'X') or (cand_isbn[-1] == '0'):
                cand_paritybool = True
            else:
                pass

            # 같은 것을 일단 처리
            if q_isbn == cand_isbn: # 일단 같으면 같은 isbn
                isbn_similarity = 1

            elif len(q_isbn) != len(cand_isbn): # 자리 수가 다르면 무조건 다름
                isbn_similarity = 0

            else: # (q_isbn!=cand_isbn and len(q_isbn)==len(cand_isbn))
                  # 자리 수가 같은데 parity code때문에 달라진 경우
                  # 주의 - 도서관 요청으로 자리수가 달라도 같은 isbn으로 보는 코드 삭제함

                # 1) parity code 처리
                if q_paritybool or cand_paritybool:# 서지 또는 소장 parity 코드 보유 (O) -> 마지막 자리 제거 후 비교
                    condition = (q_isbn[:-1] == cand_isbn[:-1])
                else:                              # 서지, 소장 모두 parity 코드 (X) -> 둘 다 13자리든, 10자리든 통으로 비교
                    condition = (q_isbn == cand_isbn)

                if condition: # 3) 조건 충족 시 같은 isbn
                    isbn_similarity = 1
                else:
                    isbn_similarity = 0

        return isbn_similarity

class SimilarityModule:
    def __init__(self, device_num, model_path):
        self.pp = Preprocessing()
        self.differ = Differ()

        # GPU
        logger.info(f"Load model on GPU:{device_num}")
        device = torch.device(f"cuda:{device_num}")
        torch.cuda.set_device(device)
        self.model = SentenceTransformer(model_path, device=f"cuda:{device_num}")
        self.model.cuda()

        # query = {'title': title,
        #          'author': author,
        #          'publisher': publisher,
        #          'publisher_year': publisher_year,
        #          'isbn': isbn}
        # return_cands = [
        #     {   
        #         'rec_key': id,
        #         'title': title,
        #         'author': author,
        #         'publisher': publisher,
        #         'publisher_year': publisher_year,
        #         'isbn': isbn
        #     }
        # ]

    def compute_with_model(self, query: dict, return_cands: list, sim_threshold: float): # return_cands:list[dict]
        start_time = time()

        # query의 모든 자료형 string화
        for key, val in query.items():
            query[key] = str(val)

        # query 결측치 선처리 및 메서드 딕셔너리 화
        q_missed_keys:list[str] = [query_key for query_key, query_value in query.items() if query_value == '']
        differ_methods = {
            'author': self.differ.differ_author,
            'publisher': self.differ.differ_publisher,
            'isbn': self.differ.differ_isbn
        }

        # 서명 -----------------------------------------------------------------------
        # candidate 선처리
        cand_titles:list[str] = []               
        for cand_dict in return_cands:
            cand_titles.append(self.pp.space_punc(cand_dict['title'])) 

        simil_dict = {}
        # 1차 형태 = {rec_key: temp_dict, rec_key: temp_dict, ...}
        # 2차 형태 = {rec_key: similarity_array, rec_key: similarity_array, ...}

        q_title:str = self.pp.space_punc(query["title"])
        enc = lambda x: self.model.encode(x, convert_to_tensor=True)
        title_cossim:Tensor = util.cos_sim(enc(q_title), enc(cand_titles))
        
        if title_cossim.size()[-1] > 1:
            title_cossim:list = title_cossim.squeeze().tolist() # Tensor -> list
        elif title_cossim.size()[-1] == 1:
            title_cossim:list = [title_cossim.squeeze().item()] # Tensor -> float -> list
        else:
            raise Exception('candidate 개수 0개, 검색결과 개수 확인')

        for cand_idx, cand_dict in enumerate(return_cands):
            a_cossim:float = title_cossim[cand_idx]
            if a_cossim > 1 :
                title_val = 1
            elif a_cossim < 0 :
                title_val = 0
            else: 
                title_val = a_cossim
            temp_dict:dict = { # temp_dict: 나중에 similarity_array로 변환할 예정
                                 'title': title_val,
                                 'title_rule': 0.0,
                                 'author': 0.0,
                                 'publisher': 0.0,
                                 'publisher_year': 0.0,
                                 'isbn': 0.0,
                             } 
                
            simil_dict[cand_dict['rec_key']] = temp_dict

        # 시리즈(1)
        num_in_qtitle:bool = False
        if re.search(r'\d', query['title']):
            num_in_qtitle:bool = True
        else:
            pass

        # 각 cand 당 결측치 수 만큼 점수 하향조정(1)
        # 허점: 쿼리에 결측치가 있고 서지에도 결측치가 있는 경우
        # 결측치를 제외한 나머지 데이터의 수치가 매우 높을 때 하향 조정이 잘 되지 않을 수 있음
        missed_dict:dict = {} # {rec_key: num_missed_keys, rec_key: num_missed_keys, ...}

        # 여기부터 한 sample 단위로 돌기----------------------------------------------------------------------------
        for cand_idx, cand_dict in enumerate(return_cands): # 서지정보 목록에서 서지정보 돌기
            cands_missed_cnt:int = 0
            for cand_key, cand_value in cand_dict.items():
                if (cand_key != 'rec_key') and (cand_value == ''): # 각 cand 당 결측치 수 만큼 점수 하향조정(2)
                    cands_missed_cnt += 1
            missed_dict[cand_dict['rec_key']] = cands_missed_cnt

            # 하나의 cand에 대응하는 similarity_array를 생성할 temp_dict 초기값 불러오기
            temp_dict:dict = simil_dict[cand_dict['rec_key']]
            for cand_key, cand_value in cand_dict.items(): # 각 서지정보의 도서정보 key 돌기

                if cand_key == 'title': # 서명 규칙 추가, 서명에 결측치 없는 것으로 가정
                    # 기본적인 전처리
                    query[cand_key] = self.pp.space_punc(query[cand_key])
                    cand_dict[cand_key] = self.pp.space_punc(cand_dict[cand_key])

                    # 미리 선언
                    cand_nospace:str = cand_dict[cand_key].replace(' ', '')
                    subtitle_issue:bool = False
                    seriesnum_intitle:bool = False
                    second_language:bool = False
                    check_isbn_fortitle:bool = False
                    seclan_isbn:bool = False

                    # 외서일 경우(1)
                    if (cand_dict['isbn'] == '') and (len(cand_dict['isbn'])!=13 or len(cand_dict['isbn'])!=10): # 제대로된 isbn인지 먼저 체크
                        pass
                    else:
                        check_isbn_fortitle:bool = True
                        not_8or9 = lambda x: (x!='8') and (x!='9') # string으로 숫자쓰기 주의
                        len_isbn = len(cand_dict['isbn'])
                        first_digit = cand_dict['isbn'][len_isbn-10]

                        if len_isbn == 13:
                            seclan_isbn = not_8or9(first_digit) and not cand_dict['isbn'][:5] == "97911" # string으로 숫자쓰기 주의

                        elif len_isbn == 10:
                            seclan_isbn = not_8or9(first_digit)
                        else:
                            pass

                    # 외서일 경우(2)
                    if check_isbn_fortitle and seclan_isbn:
                        second_language:bool = True  
                        if query[cand_key].replace(' ', '') == cand_dict[cand_key].replace(' ', ''): # 공백제거 이후 같으면 완전일치 (소문자화는 이미 space_punc로 진행)
                            temp_dict['title_rule'] = 1
                        else:
                            temp_dict['title_rule'] = 0

                    # 한국도서일 경우
                    else:
                        # 딥러닝 유사도가 낮은 이유가 원제부제로 인한 길이차이인 경우(1)
                        q_nospace:str = query[cand_key].replace(' ', '')
                        if temp_dict['title'] < 0.9 and (
                            (q_nospace in cand_nospace) or (cand_nospace in q_nospace)
                        ):
                            subtitle_issue:bool = True
                            title_rule_val:float = 0.9 + temp_dict['title'] * 0.01
                        else:
                            title_rule_val:float = temp_dict['title'] # 원제부제 케이스에 걸리지 않아도 title_rule에서 딥러닝 모델 유사도 값을 복사
                        temp_dict['title_rule'] = title_rule_val
                        
                        # 시리즈 (2)
                        # 조건 부합 시 유사도 5% 추가
                        if re.search(r'\d', cand_dict[cand_key]) or num_in_qtitle:
                            q_ttl_nums:list[int] = [int(a_num) for a_num in re.findall(r'\d+', query['title'])]
                            cand_ttl_nums:list[int] = [int(a_num) for a_num in re.findall(r'\d+', cand_dict[cand_key])]

                            if q_ttl_nums == cand_ttl_nums: # 완전 일치하는 경우만 점수 상승
                                seriesnum_intitle:bool = True
                                seriesnum_cossim:float = temp_dict['title_rule'] * 1.05
                                if seriesnum_cossim > 1:
                                    temp_dict['title_rule'] = 1
                                else:
                                    temp_dict['title_rule'] = seriesnum_cossim
                            else:
                                pass # 시리즈 조건에 해당되지 않는다면 temp_dict['title_rule'] = title_rule_val 그대로 남음
                        else:
                            pass
                else:
                    if cand_key in q_missed_keys: # 쿼리 결측치
                        temp_dict[cand_key] = -1

                    elif cand_value == '': # 서지 결측치
                        temp_dict[cand_key] = 0
                    
                    else: 
                        if cand_key in differ_methods.keys(): # 저자, 출판사, ISBN의 일치율
                            similarity_value:float or int = differ_methods[cand_key](cand_dict[cand_key], query[cand_key])
                            temp_dict[cand_key] = similarity_value

                        elif cand_key == 'publisher_year': # 출판년도 일치율
                            try:
                                if  int(query['publisher_year']) == int(cand_dict["publisher_year"]): # 출판년도 int자료형 변환에러 대비
                                    publisher_year_value = 1
                                else:
                                    try:
                                        diff_year = abs(
                                            int(query['publisher_year']) - int(cand_dict['publisher_year'])
                                        ) # 출판년도 int자료형 변환에러 대비
                                        if diff_year > 100:
                                            publisher_year_value = 0
                                        else:
                                            publisher_year_value = (1 - 0.01 * diff_year)
                                    except:
                                        publisher_year_value = 0
                            except:
                                publisher_year_value = 0

                            temp_dict[cand_key] = publisher_year_value
                        
                        else: # reckey같은 key들은 continue
                            continue

            # temp_dict를 similarity_array로 변환
            similarity_array:list[float] = [0.0]*6
            for array_idx, temp_key in enumerate(temp_dict):
                similarity_array[array_idx] = float(temp_dict[temp_key]) # 유사도 및 일치율 값을 float로 고정

            # 서명 규칙 이어서
            # 딥러닝 유사도가 낮은 이유가 원제부제로 인한 길이차이인 경우(2)
            restof_title:list[float] = [s for s_idx, s in enumerate(similarity_array) if s >= 0 and s_idx > 1]
            if len(restof_title) == 0: # 쿼리에 서명만 있는 경우 ZeroDivisionError 대비, 가공 후 결측치가 되는 경우가 있으므로 if를 남겨둠
                pass

            elif subtitle_issue and sum(restof_title)/len(restof_title) > 0.9: # 원제부제 길이차이 조건(1), (2)를 모두 통과한 경우
                similarity_array[0] = similarity_array[1] # 시리즈에 해당되었다면 이미 similarity_array[1]에 포함됨

            elif second_language: # 외서일 경우(3)
                similarity_array[0] = similarity_array[1]

            else: # 원제부제 길이차이(2) 불충족 또는 원제부제 문제에 해당이 안되면 else
                if seriesnum_intitle: # 그 중 series 조건에 해당하는 경우
                    similarity_array[0] = similarity_array[0]*1.05 if similarity_array[0]*1.05 <= 1 else 1
                else:
                    pass
            similarity_array.pop(1) # 앞에 변수선언 금지
            simil_dict[cand_dict['rec_key']] = similarity_array # 조건을 모두 거친 similarity_array를 simil_dict에 할당


        # 검증용
        self.simil_dict = simil_dict

        # 최종 -----------------------------------------------------------------------------------------------------
        final_simlist:list[dict] = [] # [{'rec_key': val, 'similarity': val}, {'rec_key': val, 'similarity': val}, ...]
        weight:list[float] = [0.3, 0.2, 0.1, 0.2, 0.2] # 가중치
        for_samesimil:list[float] = []

        # 검증용
        self.all_simlist = []

        for key, val in simil_dict.items():
            if sum(val[1:]) == -4: # 쿼리에 title만 있는 경우, 가공 후 결측치가 되는 경우가 있으므로 if를 남겨둠
                avg_simil:float = val[0]
            else:
                posi_weight:list[float] = [weight[i] for i, v in enumerate(val) if v >= 0]
                if len(posi_weight) != len(val): # 결측치 존재 시 Softmax로 가중치 변경
                    sftmx = nn.Softmax(dim=-1)
                    final_weight:list[float] = sftmx(torch.tensor(posi_weight)).tolist()
                else:
                    final_weight:list[float] = weight

                posi_val:list[float] = [v for v in val if v >= 0]

                to_sum = [final_weight[i] * posi_val[i] for i, pv in enumerate(posi_val)] # 가중평균 계산
                avg_simil:float = sum(to_sum)

            # 최대 최소값 조정
            if avg_simil > 1:
                avg_simil = 1
            elif avg_simil < 0:
                avg_simil = 0
            else: pass

            # 검증용
            self.all_simlist.append({'rec_key': key, 'similarity': avg_simil})
                
            if avg_simil > sim_threshold: # 임계값
                final_simlist.append({'rec_key': key, 'similarity': avg_simil})
                for_samesimil.append(avg_simil)
            else:
                continue

         

        for a_idx, a_dict in enumerate(final_simlist):
            # 각 cand 당 결측치 수 만큼 점수 하향조정(3)
            missed_cnts:int = missed_dict[a_dict['rec_key']]
            similarity:float = a_dict['similarity']
            if missed_cnts and similarity > 0.90:
                similarity_missed:float = similarity - 0.2 * missed_cnts
                if similarity_missed > 0.90:
                    temp:float = similarity_missed
                else: # 결측치 때문에 점수가 깎여 threshold에 들지 못함을 방지
                    temp:float = 0.90 + 0.01 * similarity_missed
                a_dict['similarity'] = temp
                for_samesimil[a_idx] = temp
            else:
                pass
            # 같은 일치율값 방지를 위한 미세조정
            similarity:float = a_dict['similarity']
            if (for_samesimil.count(similarity) > 1) and (similarity != 1):
                a_dict['similarity'] += random.uniform(1e-10, 1e-11)
            else:
                pass


        # 전체 일치율 기준으로 sort
        final_simlist:list[dict] = sorted(final_simlist, key=lambda x: x['similarity'], reverse=True)


        logger.info(
            f"Query: {query}, Length of candidates: {len(return_cands):,}, "
            f"Length of results: {len(final_simlist):,}, "
            f"Compute time: {int((time() - start_time) * 1000):,} ms"
        )
        

        return final_simlist
