# Environment Discretization and Deterministic Motion Policies with Discrete Optimal Planning

1. Choose the environment <env> and resolution <res>.
2. Take the Env<env>_res<res>.txt file in Environments folder.
3. Get visibility matrix by running 
	- compute_visibility.py Env<env>_res<res>.txt
4. Use compute_E_parallel.c to get the E table and the workspace W
	- Compile: gcc -o compute_E compute_E_parallel.c -fopenmp -lm
	- Execute: ./compute_E Env<env>_res<res>.txt visual_Env<env>_res<res>.txt
5. Rename E.txt to E_Env<env>_res<res>.txt
6. Rename W.txt to W_Env<env>_res<res>.txt
7. Run 'pursuit-evasion_discrete_domain_autonomousEvader.py Env<env>_res<res>.txt' to simulate a game with an autonomous evader
8. Run 'pursuit-evasion_discrete_domain_default_evader.py Env<env>_res<res>.txt' to simulate a game with an evader with predifinded behaviour
