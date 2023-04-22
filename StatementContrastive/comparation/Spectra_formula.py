class formula():
    def __init__(self, formula_name,tf,tp,Ftf,Ftp,l):
        self.formula_name = formula_name
        self.tp = tp
        self.tf = tf
        self.Ftp = Ftp
        self.Ftf = Ftf
        self.mu = l
    def getScore(self):
        if self.formula_name == "Tarantula":
            self.Tarantula(self.tf,self.tp,self.Ftf,self.Ftp)
        elif self.formula_name == "Ample":
            self.Ample(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "SorensenDice":
            self.SorensenDice(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Kulczynski2":
            self.Kulczynski2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "M1":
            self.M1(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Goodman":
            self.Goodman(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Overlap":
            self.Overlap(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "zoltar":
            self.zoltar(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "ER5c":
            self.ER5c(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP13":
            self.GP13(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "DStar2":
            self.DStar2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "DStar":
            self.DStar(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "ER1a":
            self.ER1a(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "ER1b":
            self.ER1b(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Wong3":
            self.Wong3(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP19":
            self.GP19(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP02":
            self.GP02(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Wong1":
            self.Wong1(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Anderberg":
            self.Anderberg(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Hamming":
            self.Hamming(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "M2":
            self.M2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "SimpleMatching":
            self.SimpleMatching(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Dice":
            self.Dice(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "RussellRao":
            self.RussellRao(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Ochiai":
            self.Ochiai(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Jaccard":
            self.Jaccard(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Hamann":
            self.Hamann(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Kulczynski1":
            self.Kulczynski1(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Sokal":
            self.Sokal(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "RogersTanimoto":
            self.RogersTanimoto(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Euclid":
            self.Euclid(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Ochiai2":
            self.Ochiai2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "Wong2":
            self.Wong2(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "GP03":
            self.GP03(self.tf, self.tp, self.Ftf, self.Ftp)
        elif self.formula_name == "SBI":
            self.SBI(self.tf, self.tp, self.Ftf, self.Ftp)
    def getAllScore(self):
        slist = list()
        slist.append(self.Tarantula(self.tf,self.tp,self.Ftf,self.Ftp))
        slist.append(self.Ample(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.SorensenDice(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.M1(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Goodman(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Overlap(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.zoltar(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.ER5c(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP13(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.DStar2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.ER1a(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.ER1b(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Wong3(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP19(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP02(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Wong1(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Anderberg(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Hamming(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.M2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.SimpleMatching(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Dice(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.RussellRao(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Jaccard(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Hamann(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski1(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Sokal(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.RogersTanimoto(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Euclid(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.Wong2(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.GP03(self.tf, self.tp, self.Ftf, self.Ftp))
        slist.append(self.SBI(self.tf, self.tp, self.Ftf, self.Ftp))
        return slist

    def getAllScore1(self,tf,tp):
        slist = list()
        slist.append(self.Tarantula(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Ample(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.SorensenDice(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.M1(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Goodman(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Overlap(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.zoltar(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.ER5c(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP13(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.DStar2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.ER1a(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.ER1b(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Wong3(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP19(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP02(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Wong1(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Anderberg(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Hamming(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.M2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.SimpleMatching(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Dice(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.RussellRao(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Jaccard(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Hamann(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Kulczynski1(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Sokal(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.RogersTanimoto(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Euclid(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Ochiai2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.Wong2(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.GP03(tf, tp, self.Ftf, self.Ftp))
        slist.append(self.SBI(tf, tp, self.Ftf, self.Ftp))
        return slist
    def getMutantScore(self,l):
        list1 = list()
        for i in range(len(l)):
            list1.append(self.getAllScore1(l[i][0],l[i][1]))
        list2 = list()
        for i in range(34):
            num = -10000000
            for j in range(len(l)):
                if list1[j][i]>=num:
                    num = list1[j][i]
            list2.append(num)
        num1 = float(0)
        for i in range(len(l)):
            num1+=self.MUSE(l[i][0],l[i][1],self.Ftf,self.Ftp)
        a = float(num1/float(len(l)))
        list2.append(a)
        return list2




    def MUSE(self,tf,tp,Ftf,Ftp):
        a = tf/(tf+Ftf)-tp/(tp+Ftp)
        return a


    def Tarantula(self,tf,tp,Ftf,Ftp):
        if Ftf+tf == 0:
            a = 0
        else:
            a = tf/(Ftf+tf)
        if Ftp+tp == 0:
            b = 0
        else:
            b = tp/(Ftp+tp)
        if a+b == 0:
            s = 0
        else:
            s = a/(a+b)
        return s

    def Ample(self,tf,tp,Ftf,Ftp):
        if Ftf + tf == 0:
            a = 0
        else:
            a = tf / (Ftf + tf)
        if Ftp + tp == 0:
            b = 0
        else:
            b = tp / (Ftp + tp)
        s = a-b
        return s

    def SorensenDice(self,tf,tp,Ftf,Ftp):
        if tf+tp+Ftf ==0:
            s = 0
        else:
            s = (2*tf)/(2*tf+tp+Ftf)
        return s

    def Kulczynski2(self, tf, tp, Ftf, Ftp):
        if Ftf + tf == 0:
            a = 0
        else:
            a = tf/(Ftf + tf)
        if tf+tp == 0:
            b = 0
        else:
            b = tf/(tf+tp)
        s = 0.5*(a+b)
        return s

    def M1(self, tf, tp, Ftf, Ftp):
        if Ftf+tp == 0:
            s = 0
        else:
            s = (tf+Ftp)/(Ftf+tp)
        return s

    def Goodman(self, tf, tp, Ftf, Ftp):
        if (tf+Ftf+tp) == 0:
            s = 0
        else:
            s = (2*tf-Ftf-tp)/(2*tf+Ftf+tp)
        return s

    def Overlap(self, tf, tp, Ftf, Ftp):
        a = min(tf,tp,Ftf)
        if a == 0:
            s = 0
        else:
            s = tf/a
        return s

    def zoltar(self, tf, tp, Ftf, Ftp):
        if tf == 0:
            return 0
        else:
            a = tf+Ftf+tp+((10000*Ftf*tp)/tf)
            if a == 0:
                return 0
            else:
                return tf/a
    def ER5c(self, tf, tp, Ftf, Ftp):
        if tf<tf+Ftf:
            return 0
        else:
            return 1

    def GP13(self, tf, tp, Ftf, Ftp):
        if tp+tf == 0:
            return 0
        else:
            s = tf*(1+(1/(2*tp+tf)))
            return s

    def DStar2(self, tf, tp, Ftf, Ftp):
        if tp+Ftf == 0:
            return 0
        else:
            return (tf*tf)/tp+Ftf

    def DStar(self, tf, tp, Ftf, Ftp):
        if tp + Ftf == 0:
            return 0
        else:
            return tf / tp + Ftf

    def ER1a(self, tf, tp, Ftf, Ftp):
        if tf<Ftf+tf:
            return -1
        else:
            return Ftp

    def ER1b(self, tf, tp, Ftf, Ftp):
        return tf-tp/(tp+Ftp+1)

    def Wong3(self, tf, tp, Ftf, Ftp):
        if tp<=2:
            return tf-tp
        elif tp<=10 and tp>2:
            return tf-2-0.1*(tp-2)
        else:
            return tf-2.8-0.01*(tp-10)

    def GP19(self, tf, tp, Ftf, Ftp):
        a = tp-tf+tf+Ftf-tp-Ftp
        return tf*pow(a,0.5)

    def GP02(self, tf, tp, Ftf, Ftp):
        return 2*(tf+tp+Ftp)+tp

    def Wong1(self, tf, tp, Ftf, Ftp):
        return tf

    def Anderberg(self, tf, tp, Ftf, Ftp):
        a = tf+2*Ftf+2*tp
        if a == 0:
            return 0
        else:
            return tf/a

    def Hamming(self, tf, tp, Ftf, Ftp):
        return tf+Ftp

    def M2(self, tf, tp, Ftf, Ftp):
        a = tf+Ftp+2*Ftf+2*tp
        if a == 0:
            return 0
        else:
            return tf/a

    def SimpleMatching(self, tf, tp, Ftf, Ftp):
        a = tf+tp+Ftf+Ftp
        if a == 0:
            return 0
        else:
            return (tf+Ftp)/a

    def Dice(self, tf, tp, Ftf, Ftp):
        a = tf+Ftf+tp
        if a == 0:
            return 0
        else:
            return 2*tf/a

    def RussellRao(self, tf, tp, Ftf, Ftp):
        a = tf+tp+Ftf+Ftp
        if a == 0:
            return 0
        else:
            return tf/a

    def Ochiai(self, tf, tp, Ftf, Ftp):
        a = (tf+tp)*(tf+Ftf)
        if a == 0:
            return 0
        else:
            return tf/a

    def Jaccard(self, tf, tp, Ftf, Ftp):
        a = tf+Ftf+tp
        if a == 0:
            return 0
        else:
            return tf/a

    def Hamann(self, tf, tp, Ftf, Ftp):
        a = tf+tp+Ftf+Ftp
        if a == 0:
            return 0
        else:
            return (tf+Ftp-tp-Ftf)/a

    def Kulczynski1(self, tf, tp, Ftf, Ftp):
        a = Ftp+Ftf+tp
        if a == 0:
            return 0
        else:
            return tf/a

    def Sokal(self, tf, tp, Ftf, Ftp):
        a = 2*tf+2*Ftp+Ftp+tp
        if a == 0:
            return 0
        else:
            return (2*tf+2*Ftp)/a


    def RogersTanimoto(self, tf, tp, Ftf, Ftp):
        a = tf+Ftp+2*Ftp+2*tp
        if a == 0:
            return 0
        else:
            return (tf+Ftp)/a

    def Euclid(self, tf, tp, Ftf, Ftp):
        n = tf+Ftp
        return pow(n,0.5)

    def Ochiai2(self, tf, tp, Ftf, Ftp):
        a = (tf+tp)*(Ftf+Ftp)*(tf+Ftp)*(Ftf+tp)
        if a == 0:
            return 0
        else:
            return (tf+tp)/a

    def Wong2(self, tf, tp, Ftf, Ftp):
        return tf-tp

    def GP03(self, tf, tp, Ftf, Ftp):
        a = tf*tf-pow(tp,0.5)
        return pow(a,0.5)

    def SBI(self, tf, tp, Ftf, Ftp):
        a = tf+tp
        if a == 0:
            return 0
        else:
            return tf / a

