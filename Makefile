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

pcbbv2: profile_cbb_v2.py
	$(KP) profile_cbb_v2.py
	$(LP) profile_cbb_v2.py.lprof

picbb: profile_icbb.py
	$(KP) profile_icbb.py
	$(LP) profile_icbb.py.lprof

test:
	pytest tests/test_bounds.py tests/test_bb.py tests/test_cbb.py tests/test_meel.py
