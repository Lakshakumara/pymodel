from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class OULAD_data:
    def __init__(self, data_dir='data'):
        self.train_dir = Path(data_dir)
        self.showData = False
    
    def _read_assessments(self):
        return pd.read_csv('./data/assessments.csv', dtype={"weight": float})
    
    def _read_studAss(self):
        return pd.read_csv('./data/studentAssessment.csv')
    
    def _read_studInfo(self):
        return pd.read_csv('./data/studentinfo.csv')
        
    def _read_studInfo_trim(self):
        d=pd.read_csv('./data/studentinfo.csv')
        d=d[["code_module","code_presentation","id_student","num_of_prev_attempts","final_result"]]
        return d
    
    def _read_registration(self):
        return pd.read_csv('./data/studentRegistration.csv')
        
    def _read_std_vle(self):
        return pd.read_csv('./data/studentVle.csv')
    
    def _read_vle(self):
        vle = pd.read_csv('./data/vle.csv')
        vle = vle.drop(vle[vle['week_from'] == '?'].index)
        return vle
        
    def _exam(self, exam=True):
        #pass
        assessments=self._read_assessments()
        if(exam):
            if self.showData:
                print("info: Exams only not included assignment data")
            return assessments[assessments["assessment_type"]=="Exam"]
        else:
            return assessments[assessments["assessment_type"]!="Exam"]
    
    def _no_of_tma(self):
        # Correct
        return self._exam(False).groupby(["code_module","code_presentation"]).count()["id_assessment"].reset_index()
    
    def _stud_ass(self, exam=False):
        #default Correct
        stud_ass = pd.merge(self._read_studAss(), self._exam(exam), how="inner", on=["id_assessment"])
        stud_ass = stud_ass.drop(stud_ass[stud_ass['score'] == '?'].index)

        stud_ass["score"]= stud_ass["score"].astype(int)
        stud_ass["pass"]=(stud_ass["score"] >= 40)
        # only false
        stud_ass["weight"]= stud_ass["weight"].astype(float)
        stud_ass["weight"]=(stud_ass["weight"] == 40)
        stud_ass["weighted_grade"]=stud_ass["score"]*stud_ass["weight"]/100
        
        stud_ass.groupby(["id_student","code_module","code_presentation"]).sum()["weighted_grade"].reset_index()

        #print(stud_ass)
        return stud_ass
    
    def _assement_avg_ps_pm(self):
        # ok
        return self._stud_ass().groupby(["id_student","code_module","code_presentation"]).sum()["weighted_grade"].reset_index()
    
    def _grade(self, status, slice):
        studInfo=self._read_studInfo()
        studInfo= studInfo[studInfo["final_result"]==status].reset_index()
        if slice:
            return self._slice(studInfo)
        else :  
            return studInfo
    
    def _complete(self):
        #Removing the cases where the student has withdrawn their registration to the module
        studInfo=self._read_studInfo()
        studInfo= studInfo[studInfo["final_result"]!="Withdrawn"].reset_index()
        return studInfo
    
    def _pass_rate_ps_pm(self):
        # ok
        #Pass rate per student per module
        stud_ass=self._stud_ass()
        amounts=self._no_of_tma()
        pass_rate=pd.merge((stud_ass[stud_ass["pass"]==True].groupby(["id_student","code_module","code_presentation"]).count()["pass"]).reset_index(),
                           amounts,how="left",on=["code_module","code_presentation"])
        pass_rate["pass_rate"]=pass_rate["pass"]/pass_rate["id_assessment"]
        pass_rate.drop(["pass", "id_assessment"], axis=1,inplace=True)
        pass_rate.head()
        return pass_rate
    
    def _fail_rate_ps_pm(self):
        stud_ass=self._stud_ass()
        amounts=self._no_of_tma()
        fail_rate=pd.merge((stud_ass[stud_ass["pass"]==False].groupby(["id_student","code_module","code_presentation"]).count()["pass"]).reset_index(),amounts,how="left",on=["code_module","code_presentation"])
        fail_rate["fail_rate"]=fail_rate["pass"]/fail_rate["id_assessment"]
        fail_rate.drop(["pass","id_assessment"], axis=1,inplace=True)
        return fail_rate  
    
    def _slice(self, studInfo):
        return studInfo[["code_module","code_presentation","id_student","num_of_prev_attempts","final_result"]]
    
    def _final_exam_score(self):
        stud_exams=self._stud_ass(True)
        stud_exams["exam_score"]=stud_exams["score"]
        stud_exams.drop(["weighted_grade", "id_assessment","date_submitted","is_banked", "score","assessment_type","date","weight","pass"],axis=1,inplace=True)
        return stud_exams

    def _avg_clk_site(self):
        return self._read_std_vle().groupby(["id_student","id_site","code_module","code_presentation"]).mean().reset_index()
    
    def _avg_clk_student(self):
        #General average per student
        return self._avg_clk_site().groupby(["id_student","code_module","code_presentation"]).mean()[["date","sum_click"]].reset_index()

    def _merge(self):
        #merger
        df_1=pd.merge(self._assement_avg_ps_pm(),self._pass_rate_ps_pm(),how="inner",on=["id_student","code_module","code_presentation"])
        assessment_info=pd.merge(df_1, self._stud_ass(True), how="inner", on=["id_student","code_module","code_presentation"])
        
        df_2=pd.merge(self._read_studInfo_trim(),assessment_info,how="inner",on=["id_student","code_module","code_presentation"])
        final_df=pd.merge(df_2,self._avg_clk_student(),how="inner", on=["id_student","code_module","code_presentation"])
        final_df.drop(["id_student","code_module","code_presentation" ],axis=1,inplace=True)
        final_df.drop(["id_assessment","date_submitted","is_banked", "score","assessment_type","weight","date_x","date_y","weighted_grade_y","pass"], axis=1, inplace=True)
        final_df=final_df[final_df["sum_click"]<=10]
        final_df=final_df[final_df["num_of_prev_attempts"]<=4]
        return final_df
    
    def _mergef(self):
        #merger with fail rate
        df_1=pd.merge(self._assement_avg_ps_pm(),self._fail_rate_ps_pm(),how="inner",on=["id_student","code_module","code_presentation"])
        assessment_info=pd.merge(df_1, self._stud_ass(True), how="inner", on=["id_student","code_module","code_presentation"])
        
        df_2=pd.merge(self._read_studInfo_trim(),assessment_info,how="inner",on=["id_student","code_module","code_presentation"])
        final_df=pd.merge(df_2,self._avg_clk_student(),how="inner", on=["id_student","code_module","code_presentation"])
        final_df.drop(["id_student","code_module","code_presentation" ],axis=1,inplace=True)
        final_df.drop(["id_assessment","date_submitted","is_banked", "score","assessment_type","weight","date_x","date_y","weighted_grade_y"], axis=1, inplace=True)
        if self.showData:
            print(final_df.info())
        return final_df
    

    def _heatmapn(self):
        plt.figure(figsize=(8,6))
        sns.heatmap(self._successORdrop().corr(), annot=True, linewidth=.5)
        sns.pairplot(self._successORdrop())
        plt.show()
        return "sucess"

    def weight_balance(self, weight):
        try:
            if (weight)==0:
                return 1
            else:
                return weight
        except ValueError:
            return 1
        
        
    def pass_fail(self, grade):
        try:
            if int(grade)>=40:
                return True
            else:
                return False
        except ValueError:
            return False
    
    def _unreg(self):
        reg=self._read_registration()
        #print("reg\n ", reg)
        reg.drop(["date_registration"], axis=1,inplace=True)
        unreg=reg[reg["date_unregistration"]!="?"]
        #df2=pd.merge(self._read_studInfo(), unreg, how="inner",on=["id_student","code_module","code_presentation"])
        #print("unreg\n ",df2)
        return unreg
    
    def _unregData(self):
        unreg=pd.merge(self._activity(),self._unreg(),how="inner",on=["id_student","code_module","code_presentation"])
        
        #if self.showData:
        #    print("unreg  \n\n\n", unreg)
        #unreg = unreg.drop(unreg[unreg['score'] == '?'].index)
        
        return unreg  
        
    def _activity(self):
        stud_ass = pd.merge(self._read_studAss(), self._read_assessments(), how="inner", on=["id_assessment"])
        stud_ass.drop(["is_banked","date_submitted","date"], axis=1,inplace=True)
        stud_ass=stud_ass[stud_ass["score"]!="?"]
        stud_ass["score"]= stud_ass["score"].astype(float)
        stud_ass["pass"]=stud_ass["score"].apply(self.pass_fail)

        #pass rate code
        totActivity=self._no_of_tma()
        pr=pd.merge((stud_ass[stud_ass["pass"]==True]
                           .groupby(["id_student","code_module","code_presentation"])
                           .count()["pass"]).reset_index()
                          ,totActivity,how="left",on=["code_module","code_presentation"])
        pr["pass_rate"]=pr["pass"]/pr["id_assessment"]
        
        #if self.showData:
            #print("pr  \n", pr)
        
        stud_ass["weight"]= stud_ass["weight"].astype(float)
        stud_ass["weighted_grade"]=stud_ass["score"]*stud_ass["weight"]/100
        stud_ass.drop(["weight"], axis=1,inplace=True)
        #print("weighted_grade  \n", stud_ass)
        avg_wg=stud_ass.groupby(["id_student","code_module","code_presentation"]).sum()["weighted_grade"].reset_index()
        #print("avg_wg\n", avg_wg)
        
        df_1=pd.merge(avg_wg,pr, how="inner",on=["id_student","code_module","code_presentation"])
        #print("\n_activity\n", df_1)
        return df_1
     
    def _rate(self):
        # total score/ no of acivity
        stud_ass=self._activity()
        totActivity=self._no_of_tma()

        score_rate=pd.merge((stud_ass[stud_ass["pass"]==True].groupby(["id_student","code_module","code_presentation"]).count()["pass"]).reset_index(),totActivity,how="left",on=["code_module","code_presentation"])
        score_rate["pass_rate"]=score_rate["pass"]/score_rate["id_assessment"]

        #score_rate.drop(["score", "id_assessment"], axis=1,inplace=True)
        return score_rate
    
        """_summary_ id_student code_module code_presentation  weighted_grade  pass  id_assessment  pass_rate
        """
    
    def _successORdrop(self, drop=False):
        # success and failer data
        if(drop):
            stud_ass=self._unregData()
        else:
            stud_ass=self._activity()
        df=stud_ass.merge(self._avg_clk_student(),how="inner",on=["id_student","code_module","code_presentation"])
        #
        df2=pd.merge(self._read_studInfo_trim(), df, how="inner",on=["id_student","code_module","code_presentation"])
        if(drop):
            df2.drop(["id_student","code_module","code_presentation","id_assessment", "pass","date_unregistration"], axis=1, inplace=True)
            #drop Withdrawn = 0
        #if self.showData:
        #    print("df2\n", df2)
        return df2

    def _heatmap(self):
        df=self._successORdrop()
        df=df[df["sum_click"]<=10]
        df=df[df["num_of_prev_attempts"]<=4]
        #plt.figure(figsize=(8,6))
        #sns.heatmap(df.corr(), annot=True, linewidth=.5)
        #sns.pairplot(df)
        #plt.show()
        return "sucess"
    

    def get_train_set(self, limit=None):
        return self._get_set(limit=limit, directory=self.train_dir)

    def get_test_set(self, limit=None):
        return self._get_set(limit=limit, directory=self.test_dir)