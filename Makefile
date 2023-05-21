KP = kernprof -l
LP = python -m line_profiler 

pa: profile_approx_mc2_core.py
	$(KP) profile_approx_mc2_core.py
	$(LP) profile_approx_mc2_core.py.lprof

pbb: profile_bb.py
	$(KP) profile_bb.py
	$(LP) profile_bb.py.lprof


pcbb: profile_cbb.py
	$(KP) profile_cbb.py
	$(LP) profile_cbb.py.lprof
