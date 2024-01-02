from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from FlagEmbedding import FlagModel
from pptx_control import *
# 示例问题-答案知识库
# qa_pairs = [
#     {"question": "天气怎么样？", "answer": "今天天气晴朗。"},
#     {"question": "你是谁？", "answer": "我是一个AI助手。"},
#     {"question": "Python是什么？", "answer": "Python是一种编程语言。"}
# ]
model = FlagModel('/mnt/workspace/kede/kede_knowledge/bge-large-zh-v1.5', use_fp16=True)

qa_pairs = []
file_path = './file/已确认-建设工程项目管理-第一章第一节.pptx'
qa_list = extract_qa_from_pptx(file_path)
for q, a in qa_list:
    # print("question:", q)
    # print("answer:", a)
    qa_pairs.append({"question": q, 'answer': a})


def find_closest_answer(user_question, qa_pairs):
    # 准备文本数据用于TF-IDF向量化
    questions = [user_question] + [pair["question"] for pair in qa_pairs]
    
    embeddings = model.encode(questions)
    # 初始化TF-IDF向量化器
    # 计算输入问题与知识库问题的相似度
    user_question_embedding = embeddings[0]
    qa_embeddings = embeddings[1:]
    similarity = np.dot(qa_embeddings, user_question_embedding)
    
    # 找到最相似的问题的索引
    closest_question_idx = np.argmax(similarity)
    
    # 返回最相似问题的答案
    return qa_pairs[closest_question_idx]["answer"]

if __name__ == '__main__':
    # 使用函数测试
    test_question = "使用建设增值包括哪些？"
    answer = find_closest_answer(test_question, qa_pairs)
    print(f"Question: {test_question}\nAnswer: {answer}")
