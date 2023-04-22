# ["codeinfo"]]#[[tf,tp,-tf,-tp]]
# data_dict = {
#         "treeDICT": None,
#         "tokenDIC": None,
#         "codeSquence": list(),
#         "node_value": list(),
#         "adjacency_list": list(),
#         "node_order": list(),
#         "edge_order": list(),
#         "seq_label": None,
#         "pro_num": None,
#         "pro_name": None,
#         "sen_label":list(),
#         "position": None,
#         "codecover": None,
#         "codeinfo": None
#
#     }
import pandas as pd

def count_score(s):
    c = s.split(",")
    tf = int(c[0])
    tp = int(c[1])
    Ftf = int(c[2])
    Ftp = int(c[3])
    n = 0
    if tf+tp == 0:
        n = 1000
    else:

        n = (tf+tp)*(tf+Ftf)
        if n == 0:
            n = 1

    score = tf/pow(n,0.5)
    # n = 0
    # if tf + tp == 0:
    #     n = 1000
    # else:
    #     n = tf+tp
    # score = tf/n
    return score

def mao_pao(ll,this_name):
    data_dict1 = {
        "rank_list": None,
        "top_1": None,
        "top_3": None,
        "top_5": None,
        "MAR": None,
        "MFR": None,
        "exam": None

    }
    for i in range(len(ll)):
        for j in range(len(ll)-i-1):
            if ll[j][0]<ll[j+1][0]:
                ll[j],ll[j+1] = ll[j+1],ll[j]
    top1 = None
    top3 = None
    top5 = None
    if int(ll[0][1]) == 1:
        top1 = 1
    if len(ll) > 3:
        if int(ll[0][1]) == 1 or int(ll[1][1]) == 1 or int(ll[2][1]) == 1:
            top3 = 1
    else:
        top3 = 1
    if len(ll) > 5:
        if int(ll[0][1]) == 1 or int(ll[1][1]) == 1 or int(ll[2][1]) == 1 or int(ll[3][1]) == 1 or int(ll[4][1]) == 1:
            top5 = 1
    else:
        top5 = 1
    count = 0
    f_flag = 0
    mar = 0
    data_dict1["exam"] = len(ll)
    for a in range(len(ll)):
        if int(ll[a][1]) == 1 and f_flag == 0:
            data_dict1["MFR"] = a + 1

            mar += a + 1
            f_flag = 1
            count += 1
        elif int(ll[a][1]) == 1 and f_flag != 0:
            mar += a + 1
            count += 1

    try:
        data_dict1["MAR"] = '%.2f' % (mar / count)
    except Exception:
        print(mar)
        print(count)
        print(this_name)
        print(ll)

    data_dict1["rank_list"] = ll
    data_dict1["top_1"] = top1
    data_dict1["top_3"] = top3
    data_dict1["top_5"] = top5
    data_dict1["this_name"] = this_name

    return data_dict1
if __name__ == '__main__':

    filename = r"E:\pycharmworkSpace\StatementContrastive\data\cover\data_new2.json"
    de = pd.read_json(filename, lines=True)

    all_list = list()
    for i in range(len(de)):
        if i % 2 == 1:
            rank_list = list()
            p_num = None
            p_name = None
            codeinfo = None
            sen_label = None
            for pr_nu in de.loc[i, ["codeinfo"]].values:
                codeinfo = pr_nu
            for pr_na in de.loc[i, ["sen_label"]].values:
                sen_label = pr_na
            for pr_nu in de.loc[i, ["pro_num"]].values:
                p_num = pr_nu
            for pr_na in de.loc[i, ["pro_name"]].values:
                p_name = pr_na
            this_name = str(p_name) + str(p_num)
            # if this_name == "Time20":
            #     continue
            for a in range(len(codeinfo)):
                score = count_score(codeinfo[a])
                try:
                    rank_list.append([score, sen_label[a]])
                except:
                    print(codeinfo)
                    print(sen_label)
                    print(len(codeinfo))
                    print(len(sen_label))
                    print(i)
            all_list.append(mao_pao(rank_list,this_name))
            # print(codeinfo)
            # print(sen_label)
    t_top1 = 0
    t_top3 = 0
    t_top5 = 0
    T_MAR = 0
    T_MFR = 0
    exam_d = data_d = {
        "10": 0,
        "20": 0,
        "30": 0,
        "40": 0,
        "50": 0,
        "60": 0,
        "70": 0,
        "80": 0,
        "90": 0,
        "100": 0,

    }
    rrr = 0
    d_list = list()
    for iii in all_list:
        data_d = {
            "this_name": None,
            "top_1": None,
            "top_3": None,
            "top_5": None,
            "exam":None
        }
        # print("projectname: ",iii["this_name"]," |ranklist: ", iii["rank_list"]," |top-1: ",iii["top_1"]," |top-3: ",iii["top_3"]," |top-5: ",iii["top_5"])
        # print("projectname: ", iii["this_name"], " |top-1: ", iii["top_1"]," |top-3: ", iii["top_3"], " |top-5: ", iii["top_5"])
        # print(iii["rank_list"])
        data_d["this_name"] = iii["this_name"]
        flag = 0
        if d_list:
            for a in d_list:
                if a["this_name"] == iii["this_name"]:
                    flag = 1
                    if a["top_1"] == None:
                        a["top_1"] = iii["top_1"]
                    if a["top_3"] == None:
                        a["top_3"] = iii["top_3"]
                    if a["top_5"] == None:
                        a["top_5"] = iii["top_5"]
                    if a["MAR"]>iii["MAR"]:
                        a["MAR"] = iii["MAR"]
                    if a["MFR"]>iii["MFR"]:
                        a["MFR"] = iii["MFR"]
                        a["exam"] = iii["exam"]
                    break
        if flag == 0:
            data_d["this_name"] = iii["this_name"]
            data_d["top_1"] = iii["top_1"]
            data_d["top_3"] = iii["top_3"]
            data_d["top_5"] = iii["top_5"]
            data_d["MAR"] = iii["MAR"]
            data_d["MFR"] = iii["MFR"]
            data_d["exam"] = iii["exam"]
            d_list.append(data_d)
    top1 = 0
    top3 = 0
    top5 = 0
    COUNT = 0
    MAR1 = float(0)
    MFR1 = float(0)
    exam = 0
    # exam_d = data_d = {
    #     "5": 0,
    #     "10": 0,
    #     "15": 0,
    #     "20": 0,
    #     "25": 0,
    #     "30": 0,
    #     "35": 0,
    #     "40": 0,
    #     "45": 0,
    #     "50": 0,
    #     "55": 0,
    #     "60": 0,
    #     "65": 0,
    #     "70": 0,
    #     "75": 0,
    #     "80": 0,
    #     "85": 0,
    #     "90": 0,
    #
    # }

    for iii in d_list:
        # print("projectname: ", iii["this_name"], " |top-1: ", iii["top_1"], " |top-3: ", iii["top_3"], " |top-5: ",
        #       iii["top_5"])
        COUNT+=1
        if iii["top_1"] == 1:
            top1+=1
        if iii["top_3"] == 1:
            top3+=1
        if iii["top_5"] == 1:
            top5+=1
        MAR1 += float(iii["MAR"])
        MFR1 += float(iii["MFR"])
        exam = float(iii["MFR"]/iii["exam"])
        rrr += iii["exam"]
        if 0<exam and exam<=0.10:
            exam_d["10"] += 1
        if 0<exam and exam<=0.20:
            exam_d["20"] += 1
        if 0<exam and exam<=0.30:
            exam_d["30"] += 1
        if 0<exam and exam<=0.40:
            exam_d["40"] += 1
        if 0<exam and exam<=0.50:
            exam_d["50"] += 1
        if 0<exam and exam<=0.60:
            exam_d["60"] += 1
        if 0<exam and exam<=0.70:
            exam_d["70"] += 1
        if 0<exam and exam<=0.80:
            exam_d["80"] += 1
        if 0<exam and exam<=0.90:
            exam_d["90"] += 1
        if 0<exam and exam<=1:
            exam_d["100"]+=1
    t_top1 += top1
    t_top3 += top3
    t_top5 += top5
    T_MAR += MAR1/COUNT
    T_MFR += MFR1/COUNT


    print("total: ")
    print("top1: ",t_top1,"  |top3: ",t_top3,"  |top5: ",t_top5,"  |MAR: ",'%.2f' %(T_MAR),"  |MFR: ",'%.2f' %(T_MFR))
    for k,v in exam_d.items():
        print(k,":  ",float(v/381))
    print("总共")
    print(rrr)