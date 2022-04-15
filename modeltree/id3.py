#encoding=utf-8
import math
import json

def compute_ent(data):
    cdict = {}
    num = len(data)
    for e in data:
        if e[-1] in cdict:
            cdict[e[-1]] += 1
        else:
            cdict[e[-1]] = 1
    res = 0
    for key, value in cdict.items():
        res += - (value / num) * math.log2(value / num)
    return res

def split_data_by_feat(data, feat, pos):
    ndata = []
    for e in data:
        if e[pos] == feat:
            ndata.append(e[:pos] + e[pos+1:])
    return ndata


def split_features(data, num_feat):
    ent = compute_ent(data)
    pos = -1
    best = -1
    for i in range(num_feat):
        con_ent = 0
        feats = set(e[i] for e in data)
        for sub in feats:
            ndata = split_data_by_feat(data, sub, i)
            con_ent += len(ndata) / len(data) * compute_ent(ndata)
        gain = ent - con_ent
        if best < gain:
            best = gain
            pos = i
    return pos

def count(data):
    cdict = {}
    for e in data:
        if e[0] in cdict:
            cdict[e[0]] += 1
        else:
            cdict[e[0]] = 1
    cls = ''
    ccount = 0
    for key, value in cdict.items():
        if value > ccount:
            cls = key
    return cls

def ID3Tree(data, names): 
    classes = [e[-1] for e in data]
    if len(set(classes)) == 1:
        return classes[0]
    print(data[0])
    print(data)
    print('----')
    if len(data[0]) == 1:
        return count(data)
    pos  = split_features(data, len(names))
    feat = names[pos]
    tree = {feat:{}}
    del(names[pos])
    feat_values = set([e[pos] for e in data])
    for value in feat_values:
        ndata, subnames = split_data_by_feat(data, value, pos), names[:]
        tree[feat][value] = ID3Tree(ndata, subnames)
    return tree


if __name__ == '__main__':
    names = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    data = [\
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],\
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],\
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],\
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],\
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],\
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],\
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],\
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],\
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],\
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],\
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],\
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],\
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],\
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],\
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],\
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],\
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否',]]
    #ID3Tree(data, names)
    print(json.dumps(ID3Tree(data, names), indent=1, ensure_ascii=False))

