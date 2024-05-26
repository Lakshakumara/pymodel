from slp import oulad as ou
from slp import model as mo 

m = mo.Model()
dset = ou.OULAD_data()
#print(f"\_unregData ==>\n",dset._unregData())
#print(f"\nTMA ==>\n",dset._exam(False))
#print(f"\_successORdrop  ==>\n", dset._successORdrop(True))

#print(f"\nStudent _success  ==>\n", m.train())
print(f"\nStudent _success  ==>\n", m.trainRF())
#print(f"\nprediction  ==>\n", m.predictionRF("rf_w_d", [90, 1, 2.5]))
#print(f"\nStudent Pass Rate  ==>\n",dset._pass_rate_ps_pm())


#print(f"\nStudent Fail Rate  ==>\n",dset._fail_rate_ps_pm())

#print(f"_stud_ass()\n",dset._read_studInfo_trim())
#print(f"_no_of_tma()\n",dset._no_of_tma())
#print(f"_pass_rate_ps_pm()\n",dset._pass_rate_ps_pm())

#print(f"\nTensorFlow  ==>\n",m.tf())