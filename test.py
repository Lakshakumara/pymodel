from slp import oulad as ou

dset = ou.OULAD_data()
#print(f"\nExams ==>\n",dset._exam())
#print(f"\nTMA ==>\n",dset._exam(False))
#print(f"\nNo of TMAS  ==>\n",dset._no_of_tma())

#print(f"\nStudent Assignment  ==>\n",dset._stud_ass())

#print(f"\nStudent Pass Rate  ==>\n",dset._pass_rate_ps_pm())

#print(f"\nStudent Fail Rate  ==>\n",dset._fail_rate_ps_pm())

print(f"_stud_ass()\n",dset._read_studInfo_trim())
#print(f"_no_of_tma()\n",dset._no_of_tma())
#print(f"_pass_rate_ps_pm()\n",dset._pass_rate_ps_pm())