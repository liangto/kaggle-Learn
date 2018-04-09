import numpy as np 

#测试数据集
def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec

#创建合集
def creat_vocab_list(data_set):
    vocab_set=set()
    for item in data_set:
        vocab_set=vocab_set | set(item)
    return list(vocab_set)

def set_of_words2vec(vocab_list,input_set):
    result=[0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)]=1
        else:
            pass
    return result

def train_naive_bayes(train_mat,train_category):
    train_doc_num=len(train_mat)
    words_num=len(train_mat[0])
    pos_abusive=np.sum(train_category)/train_doc_num

    p0num=np.ones(words_num)
    p1num=np.ones(words_num)

    p0num_all=2.0
    p1num_all=2.0

    for i in range(train_doc_num):
        if train_category[i]==1:
            p1num+=train_mat[i]
            p1num_all+=np.sum(train_mat[i])
        else:
            p0num+=train_mat[i]
            p0num_all+=np.sum(train_mat[i])
    p1vec=np.log(p1num/p1num_all)
    p0vec=np.log(p0num/p0num_all)

    return p0vec,p1vec,pos_abusive

def classify_naive_bayes(vec2classify,p0vec,p1vec,p_class1):
    p1=np.sum(vec2classify*p1vec)+np.log(p_class1)
    p0=np.sum(vec2classify*p0vec)+np.log(1-p_class1)
    if p1>p0:
        return 1
    else:
        return 0


def bag_words2vec(vocab_list,input_set):
    result=[0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)]+=1
        else:
            print('the word:{} is not in my vocabulary'.format(word))
    return result

def testing_naive_bayes():
    list_post,list_classes=load_data_set()
    vocab_list=creat_vocab_list(list_post)

    train_mat=[]
    for post_in in list_post:
        train_mat.append(set_of_words2vec(vocab_list,post_in))
    p0v, p1v, p_abusive = train_naive_bayes(np.array(train_mat), np.array(list_classes))


    test_one = ['love', 'my', 'dalmation']
    test_one_doc = np.array(set_of_words2vec(vocab_list, test_one))
    print('the result is: {}'.format(classify_naive_bayes(test_one_doc, p0v, p1v, p_abusive)))
    test_two = ['stupid', 'garbage']
    test_two_doc = np.array(set_of_words2vec(vocab_list, test_two))
    print('the result is: {}'.format(classify_naive_bayes(test_two_doc, p0v, p1v, p_abusive)))



testing_naive_bayes()



