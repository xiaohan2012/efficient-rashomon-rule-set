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

p8: profile_solve_m_eq_12_based_on_8.py
	$(KP) profile_solve_m_eq_12_based_on_8.py
	$(LP) profile_solve_m_eq_12_based_on_8.py.lprof

test_bb:
	pytest tests/test_bounds.py tests/test_bb.py tests/test_cbb.py tests/test_cbb_v2.py
